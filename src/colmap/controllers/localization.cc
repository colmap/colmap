// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/controllers/localization.h"

// feature extraction.
#include "colmap/feature/sift.h"
#include "colmap/scene/database.h"

// mapper.
#include "colmap/estimators/pose.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/scene/projection.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/sensor/bitmap.h"

#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"
#include "colmap/util/opengl_utils.h"
#include "colmap/util/timer.h"

#include <numeric>
#include <array>
#include <fstream>


namespace colmap {

//----------------------------------------------------------------
//  FEATURE EXTRACTOR.
//----------------------------------------------------------------

namespace {

void ScaleKeypoints(const Bitmap& bitmap,
                    const Camera& camera,
                    FeatureKeypoints* keypoints) {
    if (static_cast<size_t>(bitmap.Width()) != camera.width ||
        static_cast<size_t>(bitmap.Height()) != camera.height) {
        const float scale_x = static_cast<float>(camera.width) / bitmap.Width();
        const float scale_y = static_cast<float>(camera.height) / bitmap.Height();
        for (auto& keypoint : *keypoints) {
            keypoint.Rescale(scale_x, scale_y);
        }
    }
}

void MaskKeypoints(const Bitmap& mask,
                   FeatureKeypoints* keypoints,
                   FeatureDescriptors* descriptors) {
    size_t out_index = 0;
    BitmapColor<uint8_t> color;
    for (size_t i = 0; i < keypoints->size(); ++i) {
        if (!mask.GetPixel(static_cast<int>(keypoints->at(i).x),
                           static_cast<int>(keypoints->at(i).y),
                           &color) ||
            color.r == 0) {
            // Delete this keypoint by not copying it to the output.
        } else {
            // Retain this keypoint by copying it to the output index (in case this
            // index differs from its current position).
            if (out_index != i) {
                keypoints->at(out_index) = keypoints->at(i);
                for (int col = 0; col < descriptors->cols(); ++col) {
                    (*descriptors)(out_index, col) = (*descriptors)(i, col);
                }
            }
            out_index += 1;
        }
    }
    
    keypoints->resize(out_index);
    descriptors->conservativeResize(out_index, descriptors->cols());
}

class ImageResizerThread : public Thread {
public:
    ImageResizerThread(int max_image_size,
                       JobQueue<ImageData>* input_queue,
                       JobQueue<ImageData>* output_queue)
    : max_image_size_(max_image_size),
    input_queue_(input_queue),
    output_queue_(output_queue) {}
    
private:
    void Run() override {
        while (true) {
            if (IsStopped()) {
                break;
            }
            
            auto input_job = input_queue_->Pop();
            if (input_job.IsValid()) {
                auto& image_data = input_job.Data();
                
                if (image_data.status == ImageReader::Status::SUCCESS) {
                    if (static_cast<int>(image_data.bitmap.Width()) > max_image_size_ ||
                        static_cast<int>(image_data.bitmap.Height()) > max_image_size_) {
                        // Fit the down-sampled version exactly into the max dimensions.
                        const double scale =
                        static_cast<double>(max_image_size_) /
                        std::max(image_data.bitmap.Width(), image_data.bitmap.Height());
                        const int new_width =
                        static_cast<int>(image_data.bitmap.Width() * scale);
                        const int new_height =
                        static_cast<int>(image_data.bitmap.Height() * scale);
                        
                        image_data.bitmap.Rescale(new_width, new_height);
                    }
                }
                
                output_queue_->Push(std::move(image_data));
            } else {
                break;
            }
        }
    }
    
    const int max_image_size_;
    
    JobQueue<ImageData>* input_queue_;
    JobQueue<ImageData>* output_queue_;
};

class SiftFeatureExtractorThread : public Thread {
public:
    SiftFeatureExtractorThread(const SiftExtractionOptions& sift_options,
                               const std::shared_ptr<Bitmap>& camera_mask,
                               JobQueue<ImageData>* input_queue,
                               JobQueue<ImageData>* output_queue)
    : sift_options_(sift_options),
    camera_mask_(camera_mask),
    input_queue_(input_queue),
    output_queue_(output_queue) {
        THROW_CHECK(sift_options_.Check());
        
#if !defined(COLMAP_CUDA_ENABLED)
        if (sift_options_.use_gpu) {
            opengl_context_ = std::make_unique<OpenGLContextManager>();
        }
#endif
    }
    
private:
    void Run() override {
        if (sift_options_.use_gpu) {
#if !defined(COLMAP_CUDA_ENABLED)
            THROW_CHECK_NOTNULL(opengl_context_);
            THROW_CHECK(opengl_context_->MakeCurrent());
#endif
        }
        
        std::unique_ptr<FeatureExtractor> extractor =
        CreateSiftFeatureExtractor(sift_options_);
        if (extractor == nullptr) {
            LOG(ERROR) << "Failed to create feature extractor.";
            SignalInvalidSetup();
            return;
        }
        
        SignalValidSetup();
        
        while (true) {
            if (IsStopped()) {
                break;
            }
            
            auto input_job = input_queue_->Pop();
            if (input_job.IsValid()) {
                auto& image_data = input_job.Data();
                
                if (image_data.status == ImageReader::Status::SUCCESS) {
                    if (extractor->Extract(image_data.bitmap,
                                           &image_data.keypoints,
                                           &image_data.descriptors)) {
                        ScaleKeypoints(
                                       image_data.bitmap, image_data.camera, &image_data.keypoints);
                        if (camera_mask_) {
                            MaskKeypoints(*camera_mask_,
                                          &image_data.keypoints,
                                          &image_data.descriptors);
                        }
                        if (image_data.mask.Data()) {
                            MaskKeypoints(image_data.mask,
                                          &image_data.keypoints,
                                          &image_data.descriptors);
                        }
                    } else {
                        image_data.status = ImageReader::Status::FAILURE;
                    }
                }
                
                image_data.bitmap.Deallocate();
                
                output_queue_->Push(std::move(image_data));
            } else {
                break;
            }
        }
    }
    
    const SiftExtractionOptions sift_options_;
    std::shared_ptr<Bitmap> camera_mask_;
    
    std::unique_ptr<OpenGLContextManager> opengl_context_;
    
    JobQueue<ImageData>* input_queue_;
    JobQueue<ImageData>* output_queue_;
};

// Feature extraction class to extract features for all images in a directory.
class FeatureExtractorController : public Thread {
public:
    FeatureExtractorController(const ImageReaderOptions& reader_options,
                               const SiftExtractionOptions& sift_options)
    : reader_options_(reader_options),
    sift_options_(sift_options),
    database_(reader_options_.database_path),
    image_reader_(reader_options_, &database_) {
        THROW_CHECK(reader_options_.Check());
        THROW_CHECK(sift_options_.Check());
        
        std::shared_ptr<Bitmap> camera_mask;
        if (!reader_options_.camera_mask_path.empty()) {
            camera_mask = std::make_shared<Bitmap>();
            if (!camera_mask->Read(reader_options_.camera_mask_path,
                                   /*as_rgb*/ false)) {
                LOG(ERROR) << "Cannot read camera mask file: "
                << reader_options_.camera_mask_path
                << ". No mask is going to be used.";
                camera_mask.reset();
            }
        }
        
        const int num_threads = GetEffectiveNumThreads(sift_options_.num_threads);
        THROW_CHECK_GT(num_threads, 0);
        
        // Make sure that we only have limited number of objects in the queue to
        // avoid excess in memory usage since images and features take lots of
        // memory.
        const int kQueueSize = 1;
        resizer_queue_ = std::make_unique<JobQueue<ImageData>>(kQueueSize);
        extractor_queue_ = std::make_unique<JobQueue<ImageData>>(kQueueSize);
        writer_queue_ = std::make_unique<JobQueue<ImageData>>(kQueueSize);
        
        if (sift_options_.max_image_size > 0) {
            for (int i = 0; i < num_threads; ++i) {
                resizers_.emplace_back(
                                       std::make_unique<ImageResizerThread>(sift_options_.max_image_size,
                                                                            resizer_queue_.get(),
                                                                            extractor_queue_.get()));
            }
        }
        
        if (!sift_options_.domain_size_pooling &&
            !sift_options_.estimate_affine_shape && sift_options_.use_gpu) {
            std::vector<int> gpu_indices = CSVToVector<int>(sift_options_.gpu_index);
            THROW_CHECK_GT(gpu_indices.size(), 0);
            
#if defined(COLMAP_CUDA_ENABLED)
            if (gpu_indices.size() == 1 && gpu_indices[0] == -1) {
                const int num_cuda_devices = GetNumCudaDevices();
                THROW_CHECK_GT(num_cuda_devices, 0);
                gpu_indices.resize(num_cuda_devices);
                std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
            }
#endif  // COLMAP_CUDA_ENABLED
            
            auto sift_gpu_options = sift_options_;
            for (const auto& gpu_index : gpu_indices) {
                sift_gpu_options.gpu_index = std::to_string(gpu_index);
                extractors_.emplace_back(
                                         std::make_unique<SiftFeatureExtractorThread>(sift_gpu_options,
                                                                                      camera_mask,
                                                                                      extractor_queue_.get(),
                                                                                      writer_queue_.get()));
            }
        } else {
            if (sift_options_.num_threads == -1 &&
                sift_options_.max_image_size ==
                SiftExtractionOptions().max_image_size &&
                sift_options_.first_octave == SiftExtractionOptions().first_octave) {
                LOG(WARNING)
                << "Your current options use the maximum number of "
                "threads on the machine to extract features. Extracting SIFT "
                "features on the CPU can consume a lot of RAM per thread for "
                "large images. Consider reducing the maximum image size and/or "
                "the first octave or manually limit the number of extraction "
                "threads. Ignore this warning, if your machine has sufficient "
                "memory for the current settings.";
            }
            
            auto custom_sift_options = sift_options_;
            custom_sift_options.use_gpu = false;
            for (int i = 0; i < num_threads; ++i) {
                extractors_.emplace_back(
                                         std::make_unique<SiftFeatureExtractorThread>(custom_sift_options,
                                                                                      camera_mask,
                                                                                      extractor_queue_.get(),
                                                                                      writer_queue_.get()));
            }
        }
    }
    
    ImageData GetImageData() {
        auto job = writer_queue_->Pop();
        if (job.IsValid()) {
            ImageData image_data = job.Data();
            return std::move(image_data);
        }
        return imageDataEmpty;
    }
    
private:
    void Run() override {
        PrintHeading1("Feature extraction");
        Timer run_timer;
        run_timer.Start();
        
        for (auto& resizer : resizers_) {
            resizer->Start();
        }
        
        for (auto& extractor : extractors_) {
            extractor->Start();
        }
        
        for (auto& extractor : extractors_) {
            if (!extractor->CheckValidSetup()) {
                return;
            }
        }
        
        while (image_reader_.NextIndex() < image_reader_.NumImages()) {
            if (IsStopped()) {
                resizer_queue_->Stop();
                extractor_queue_->Stop();
                resizer_queue_->Clear();
                extractor_queue_->Clear();
                break;
            }
            
            ImageData image_data;
            image_data.status = image_reader_.Next(&image_data.camera,
                                                   &image_data.image,
                                                   &image_data.bitmap,
                                                   &image_data.mask);
            
            if (image_data.status != ImageReader::Status::SUCCESS) {
                image_data.bitmap.Deallocate();
            }
            
            if (sift_options_.max_image_size > 0) {
                THROW_CHECK(resizer_queue_->Push(std::move(image_data)));
            } else {
                THROW_CHECK(extractor_queue_->Push(std::move(image_data)));
            }
        }
        
        resizer_queue_->Wait();
        resizer_queue_->Stop();
        for (auto& resizer : resizers_) {
            resizer->Wait();
        }
        
        extractor_queue_->Wait();
        extractor_queue_->Stop();
        for (auto& extractor : extractors_) {
            extractor->Wait();
        }
        
        run_timer.PrintMinutes();
    }
    
    const ImageReaderOptions reader_options_;
    const SiftExtractionOptions sift_options_;
    
    Database database_;
    ImageReader image_reader_;
    ImageData imageDataEmpty;
    
    std::vector<std::unique_ptr<Thread>> resizers_;
    std::vector<std::unique_ptr<Thread>> extractors_;
    
    std::unique_ptr<JobQueue<ImageData>> resizer_queue_;
    std::unique_ptr<JobQueue<ImageData>> extractor_queue_;
    std::unique_ptr<JobQueue<ImageData>> writer_queue_;
};

}  // namespace

std::unique_ptr<Thread> CreateFeatureExtractorController2(
                                                          const ImageReaderOptions& reader_options,
                                                          const SiftExtractionOptions& sift_options) {
    return std::make_unique<FeatureExtractorController>(reader_options,
                                                        sift_options);
}

ImageData GetImageData( std::unique_ptr<Thread> & thread ) {
    std::unique_ptr<FeatureExtractorController> controller(static_cast<FeatureExtractorController*>(thread.release()));
    ImageData imageData = controller->GetImageData();
    return imageData;
}

//----------------------------------------------------------------
//  MAPPER.
//----------------------------------------------------------------

namespace {

void SortAndAppendNextImages(std::vector<std::pair<image_t, float>> image_ranks,
                             std::vector<image_t>* sorted_images_ids) {
    std::sort(image_ranks.begin(),
              image_ranks.end(),
              [](const std::pair<image_t, float>& image1,
                 const std::pair<image_t, float>& image2) {
        return image1.second > image2.second;
    });
    
    sorted_images_ids->reserve(sorted_images_ids->size() + image_ranks.size());
    for (const auto& image : image_ranks) {
        sorted_images_ids->push_back(image.first);
    }
    
    image_ranks.clear();
}

float RankNextImageMaxVisiblePointsNum(const image_t image_id, const class ObservationManager& obs_manager) {
    return static_cast<float>(obs_manager.NumVisiblePoints3D(image_id));
}

float RankNextImageMaxVisiblePointsRatio(const image_t image_id, const class ObservationManager& obs_manager) {
    return static_cast<float>(obs_manager.NumVisiblePoints3D(image_id)) /
    static_cast<float>(obs_manager.NumObservations(image_id));
}

float RankNextImageMinUncertainty(const image_t image_id,
                                  const class ObservationManager& obs_manager) {
    return static_cast<float>(obs_manager.Point3DVisibilityScore(image_id));
}

}  // namespace

bool IncrementalMapper2::Options::Check() const {
    CHECK_OPTION_GT(init_min_num_inliers, 0);
    CHECK_OPTION_GT(init_max_error, 0.0);
    CHECK_OPTION_GE(init_max_forward_motion, 0.0);
    CHECK_OPTION_LE(init_max_forward_motion, 1.0);
    CHECK_OPTION_GE(init_min_tri_angle, 0.0);
    CHECK_OPTION_GE(init_max_reg_trials, 1);
    CHECK_OPTION_GT(abs_pose_max_error, 0.0);
    CHECK_OPTION_GT(abs_pose_min_num_inliers, 0);
    CHECK_OPTION_GE(abs_pose_min_inlier_ratio, 0.0);
    CHECK_OPTION_LE(abs_pose_min_inlier_ratio, 1.0);
    CHECK_OPTION_GE(local_ba_num_images, 2);
    CHECK_OPTION_GE(local_ba_min_tri_angle, 0.0);
    CHECK_OPTION_GE(min_focal_length_ratio, 0.0);
    CHECK_OPTION_GE(max_focal_length_ratio, min_focal_length_ratio);
    CHECK_OPTION_GE(max_extra_param, 0.0);
    CHECK_OPTION_GE(filter_max_reproj_error, 0.0);
    CHECK_OPTION_GE(filter_min_tri_angle, 0.0);
    CHECK_OPTION_GE(max_reg_trials, 1);
    return true;
}

IncrementalMapper2::IncrementalMapper2(std::shared_ptr<const DatabaseCache> database_cache)
: database_cache_(std::move(database_cache)),
reconstruction_(nullptr),
obs_manager_(nullptr),
triangulator_(nullptr),
num_total_reg_images_(0),
num_shared_reg_images_(0) {}

void IncrementalMapper2::BeginReconstruction(const std::shared_ptr<class Reconstruction>& reconstruction) {
    THROW_CHECK(reconstruction_ == nullptr);
    reconstruction_ = reconstruction;
    reconstruction_->Load(*database_cache_);
    // reconstruction_->SetUp();
    obs_manager_ = std::make_shared<class ObservationManager>(
                                                              *reconstruction_, database_cache_->CorrespondenceGraph());
    triangulator_ = std::make_shared<IncrementalTriangulator>(
                                                              database_cache_->CorrespondenceGraph(), *reconstruction_, obs_manager_);
    
    num_shared_reg_images_ = 0;
    num_reg_images_per_camera_.clear();
    for (const image_t image_id : reconstruction_->RegImageIds()) {
        RegisterImageEvent(image_id);
    }
    
    existing_image_ids_ =
    std::unordered_set<image_t>(reconstruction->RegImageIds().begin(),
                                reconstruction->RegImageIds().end());
    
    filtered_images_.clear();
    num_reg_trials_.clear();
}

void IncrementalMapper2::EndReconstruction(const bool discard) {
    THROW_CHECK_NOTNULL(reconstruction_);
    
    if (discard) {
        for (const image_t image_id : reconstruction_->RegImageIds()) {
            DeRegisterImageEvent(image_id);
        }
    }
    
    reconstruction_->TearDown();
    reconstruction_ = nullptr;
    obs_manager_.reset();
    triangulator_.reset();
}

bool IncrementalMapper2::FindInitialImagePair(const Options& options,
                                              TwoViewGeometry& two_view_geometry,
                                              image_t& image_id1,
                                              image_t& image_id2) {
    THROW_CHECK(options.Check());
    
    std::vector<image_t> image_ids1;
    if (image_id1 != kInvalidImageId && image_id2 == kInvalidImageId) {
        // Only image_id1 provided.
        if (!database_cache_->ExistsImage(image_id1)) {
            return false;
        }
        image_ids1.push_back(image_id1);
    } else if (image_id1 == kInvalidImageId && image_id2 != kInvalidImageId) {
        // Only image_id2 provided.
        if (!database_cache_->ExistsImage(image_id2)) {
            return false;
        }
        image_ids1.push_back(image_id2);
    } else {
        // No initial seed image provided.
        image_ids1 = FindFirstInitialImage(options);
    }
    
    // Try to find good initial pair.
    for (size_t i1 = 0; i1 < image_ids1.size(); ++i1) {
        image_id1 = image_ids1[i1];
        
        const std::vector<image_t> image_ids2 =
        FindSecondInitialImage(options, image_id1);
        
        for (size_t i2 = 0; i2 < image_ids2.size(); ++i2) {
            image_id2 = image_ids2[i2];
            
            const image_pair_t pair_id =
            Database::ImagePairToPairId(image_id1, image_id2);
            
            // Try every pair only once.
            if (init_image_pairs_.count(pair_id) > 0) {
                continue;
            }
            
            init_image_pairs_.insert(pair_id);
            
            if (EstimateInitialTwoViewGeometry(
                                               options, two_view_geometry, image_id1, image_id2)) {
                                                   return true;
                                               }
        }
    }
    
    // No suitable pair found in entire dataset.
    image_id1 = kInvalidImageId;
    image_id2 = kInvalidImageId;
    
    return false;
}

std::vector<image_t> IncrementalMapper2::FindNextImages(const Options& options) {
    THROW_CHECK_NOTNULL(reconstruction_);
    THROW_CHECK(options.Check());
    
    std::function<float(image_t, const class ObservationManager&)>
    rank_image_func;
    switch (options.image_selection_method) {
        case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_NUM:
            rank_image_func = RankNextImageMaxVisiblePointsNum;
            break;
        case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_RATIO:
            rank_image_func = RankNextImageMaxVisiblePointsRatio;
            break;
        case Options::ImageSelectionMethod::MIN_UNCERTAINTY:
            rank_image_func = RankNextImageMinUncertainty;
            break;
    }
    
    std::vector<std::pair<image_t, float>> image_ranks;
    std::vector<std::pair<image_t, float>> other_image_ranks;
    
    // Append images that have not failed to register before.
    for (const auto& image : reconstruction_->Images()) {
        // Skip images that are already registered.
        if (image.second.IsRegistered()) {
            continue;
        }
        
        // Only consider images with a sufficient number of visible points.
        if (obs_manager_->NumVisiblePoints3D(image.first) <
            static_cast<size_t>(options.abs_pose_min_num_inliers)) {
            continue;
        }
        
        // Only try registration for a certain maximum number of times.
        const size_t num_reg_trials = num_reg_trials_[image.first];
        if (num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
            continue;
        }
        
        // If image has been filtered or failed to register, place it in the
        // second bucket and prefer images that have not been tried before.
        const float rank = rank_image_func(image.first, *obs_manager_);
        if (filtered_images_.count(image.first) == 0 && num_reg_trials == 0) {
            image_ranks.emplace_back(image.first, rank);
        } else {
            other_image_ranks.emplace_back(image.first, rank);
        }
    }
    
    std::vector<image_t> ranked_images_ids;
    SortAndAppendNextImages(image_ranks, &ranked_images_ids);
    SortAndAppendNextImages(other_image_ranks, &ranked_images_ids);
    
    return ranked_images_ids;
}

void IncrementalMapper2::RegisterInitialImagePair(const Options& options,
                                                  const TwoViewGeometry& two_view_geometry,
                                                  const image_t image_id1,
                                                  const image_t image_id2) {
    THROW_CHECK_NOTNULL(reconstruction_);
    THROW_CHECK_NOTNULL(obs_manager_);
    THROW_CHECK_EQ(reconstruction_->NumRegImages(), 0);
    
    THROW_CHECK(options.Check());
    
    init_num_reg_trials_[image_id1] += 1;
    init_num_reg_trials_[image_id2] += 1;
    num_reg_trials_[image_id1] += 1;
    num_reg_trials_[image_id2] += 1;
    
    const image_pair_t pair_id =
    Database::ImagePairToPairId(image_id1, image_id2);
    init_image_pairs_.insert(pair_id);
    
    Image& image1 = reconstruction_->Image(image_id1);
    const Camera& camera1 = reconstruction_->Camera(image1.CameraId());
    
    Image& image2 = reconstruction_->Image(image_id2);
    const Camera& camera2 = reconstruction_->Camera(image2.CameraId());
    
    //////////////////////////////////////////////////////////////////////////////
    // Estimate two-view geometry
    //////////////////////////////////////////////////////////////////////////////
    
    image1.CamFromWorld() = Rigid3d();
    image2.CamFromWorld() = two_view_geometry.cam2_from_cam1;
    
    const Eigen::Matrix3x4d cam_from_world1 = image1.CamFromWorld().ToMatrix();
    const Eigen::Matrix3x4d cam_from_world2 = image2.CamFromWorld().ToMatrix();
    const Eigen::Vector3d proj_center1 = image1.ProjectionCenter();
    const Eigen::Vector3d proj_center2 = image2.ProjectionCenter();
    
    //////////////////////////////////////////////////////////////////////////////
    // Update Reconstruction
    //////////////////////////////////////////////////////////////////////////////
    
    reconstruction_->RegisterImage(image_id1);
    reconstruction_->RegisterImage(image_id2);
    RegisterImageEvent(image_id1);
    RegisterImageEvent(image_id2);
    
    const FeatureMatches& corrs =
    database_cache_->CorrespondenceGraph()->FindCorrespondencesBetweenImages(
                                                                             image_id1, image_id2);
    
    const double min_tri_angle_rad = DegToRad(options.init_min_tri_angle);
    
    // Add 3D point tracks.
    Track track;
    track.Reserve(2);
    track.AddElement(TrackElement());
    track.AddElement(TrackElement());
    track.Element(0).image_id = image_id1;
    track.Element(1).image_id = image_id2;
    for (const auto& corr : corrs) {
        const Eigen::Vector2d point2D1 =
        camera1.CamFromImg(image1.Point2D(corr.point2D_idx1).xy);
        const Eigen::Vector2d point2D2 =
        camera2.CamFromImg(image2.Point2D(corr.point2D_idx2).xy);
        const Eigen::Vector3d& xyz =
        TriangulatePoint(cam_from_world1, cam_from_world2, point2D1, point2D2);
        const double tri_angle =
        CalculateTriangulationAngle(proj_center1, proj_center2, xyz);
        if (tri_angle >= min_tri_angle_rad &&
            HasPointPositiveDepth(cam_from_world1, xyz) &&
            HasPointPositiveDepth(cam_from_world2, xyz)) {
            track.Element(0).point2D_idx = corr.point2D_idx1;
            track.Element(1).point2D_idx = corr.point2D_idx2;
            obs_manager_->AddPoint3D(xyz, track);
        }
    }
}

bool IncrementalMapper2::RegisterNextImage(const IncrementalMapper::Options& options,
                                           ImageData & imageData) {
    THROW_CHECK_NOTNULL(reconstruction_);
    THROW_CHECK_NOTNULL(obs_manager_);
    THROW_CHECK_GE(reconstruction_->NumRegImages(), 2);
    
    THROW_CHECK(options.Check());
    
    Image& image = imageData.image;
    Camera& camera = imageData.camera;
    
    std::vector<Eigen::Vector2d> tri_points2D;
    std::vector<Eigen::Vector3d> tri_points3D;
    
    for (int i=0; i<imageData.keypoints.size(); i++) {
        const FeatureKeypoint & keypoint = imageData.keypoints[i];
        tri_points2D.push_back( Eigen::Vector2d(keypoint.x, keypoint.y) );
    }
    
    for (const auto& pointId : reconstruction_->Point3DIds()) {
        const Point3D & point3D = reconstruction_->Point3D( pointId );
        tri_points3D.push_back( point3D.xyz );
    }
    
    //////////////////////////////////////////////////////////////////////////////
    // 2D-3D estimation
    //////////////////////////////////////////////////////////////////////////////
    
    // Only refine / estimate focal length, if no focal length was specified
    // (manually or through EXIF) and if it was not already estimated previously
    // from another image (when multiple images share the same camera
    // parameters)
    
    AbsolutePoseEstimationOptions abs_pose_options;
    abs_pose_options.num_threads = options.num_threads;
    abs_pose_options.num_focal_length_samples = 30;
    abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
    abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
    abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
    abs_pose_options.ransac_options.min_inlier_ratio =
    options.abs_pose_min_inlier_ratio;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    abs_pose_options.ransac_options.min_num_trials = 100;
    abs_pose_options.ransac_options.max_num_trials = 10000;
    abs_pose_options.ransac_options.confidence = 0.99999;
    
    AbsolutePoseRefinementOptions abs_pose_refinement_options;
    if (num_reg_images_per_camera_[image.CameraId()] > 0) {
        // Camera already refined from another image with the same camera.
        if (camera.HasBogusParams(options.min_focal_length_ratio,
                                  options.max_focal_length_ratio,
                                  options.max_extra_param)) {
            // Previously refined camera has bogus parameters,
            // so reset parameters and try to re-estimage.
            abs_pose_options.estimate_focal_length = !camera.has_prior_focal_length;
            abs_pose_refinement_options.refine_focal_length = true;
            abs_pose_refinement_options.refine_extra_params = true;
        } else {
            abs_pose_options.estimate_focal_length = false;
            abs_pose_refinement_options.refine_focal_length = false;
            abs_pose_refinement_options.refine_extra_params = false;
        }
    } else {
        // Camera not refined before. Note that the camera parameters might have
        // been changed before but the image was filtered, so we explicitly reset
        // the camera parameters and try to re-estimate them.
        abs_pose_options.estimate_focal_length = !camera.has_prior_focal_length;
        abs_pose_refinement_options.refine_focal_length = true;
        abs_pose_refinement_options.refine_extra_params = true;
    }
    
    if (!options.abs_pose_refine_focal_length) {
        abs_pose_options.estimate_focal_length = false;
        abs_pose_refinement_options.refine_focal_length = false;
    }
    
    if (!options.abs_pose_refine_extra_params) {
        abs_pose_refinement_options.refine_extra_params = false;
    }
    
    size_t num_inliers;
    std::vector<char> inlier_mask;
    
    if (!EstimateAbsolutePose(abs_pose_options,
                              tri_points2D,
                              tri_points3D,
                              &image.CamFromWorld(),
                              &camera,
                              &num_inliers,
                              &inlier_mask)) {
        return false;
    }
    
    if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
        return false;
    }
    
    //////////////////////////////////////////////////////////////////////////////
    // Pose refinement
    //////////////////////////////////////////////////////////////////////////////
    
    if (!RefineAbsolutePose(abs_pose_refinement_options,
                            inlier_mask,
                            tri_points2D,
                            tri_points3D,
                            &image.CamFromWorld(),
                            &camera)) {
        return false;
    }
    
    return true;
}

size_t IncrementalMapper2::TriangulateImage(const IncrementalTriangulator::Options& tri_options,
                                            const image_t image_id) {
    THROW_CHECK_NOTNULL(reconstruction_);
    VLOG(1) << "=> Continued observations: "
    << reconstruction_->Image(image_id).NumPoints3D();
    const size_t num_tris =
    triangulator_->TriangulateImage(tri_options, image_id);
    VLOG(1) << "=> Added observations: " << num_tris;
    return num_tris;
}

size_t IncrementalMapper2::Retriangulate(const IncrementalTriangulator::Options& tri_options) {
    THROW_CHECK_NOTNULL(reconstruction_);
    return triangulator_->Retriangulate(tri_options);
}

size_t IncrementalMapper2::CompleteTracks(const IncrementalTriangulator::Options& tri_options) {
    THROW_CHECK_NOTNULL(reconstruction_);
    return triangulator_->CompleteAllTracks(tri_options);
}

size_t IncrementalMapper2::MergeTracks(const IncrementalTriangulator::Options& tri_options) {
    THROW_CHECK_NOTNULL(reconstruction_);
    return triangulator_->MergeAllTracks(tri_options);
}

size_t IncrementalMapper2::CompleteAndMergeTracks(const IncrementalTriangulator::Options& tri_options) {
    const size_t num_completed_observations = CompleteTracks(tri_options);
    VLOG(1) << "=> Completed observations: " << num_completed_observations;
    const size_t num_merged_observations = MergeTracks(tri_options);
    VLOG(1) << "=> Merged observations: " << num_merged_observations;
    return num_completed_observations + num_merged_observations;
}

IncrementalMapper2::LocalBundleAdjustmentReport
IncrementalMapper2::AdjustLocalBundle(
                                      const Options& options,
                                      const BundleAdjustmentOptions& ba_options,
                                      const IncrementalTriangulator::Options& tri_options,
                                      const image_t image_id,
                                      const std::unordered_set<point3D_t>& point3D_ids) {
    THROW_CHECK_NOTNULL(reconstruction_);
    THROW_CHECK_NOTNULL(obs_manager_);
    THROW_CHECK(options.Check());
    
    LocalBundleAdjustmentReport report;
    
    // Find images that have most 3D points with given image in common.
    const std::vector<image_t> local_bundle = FindLocalBundle(options, image_id);
    
    // Do the bundle adjustment only if there is any connected images.
    if (local_bundle.size() > 0) {
        BundleAdjustmentConfig ba_config;
        ba_config.AddImage(image_id);
        for (const image_t local_image_id : local_bundle) {
            ba_config.AddImage(local_image_id);
        }
        
        // Fix the existing images, if option specified.
        if (options.fix_existing_images) {
            for (const image_t local_image_id : local_bundle) {
                if (existing_image_ids_.count(local_image_id)) {
                    ba_config.SetConstantCamPose(local_image_id);
                }
            }
        }
        
        // Determine which cameras to fix, when not all the registered images
        // are within the current local bundle.
        std::unordered_map<camera_t, size_t> num_images_per_camera;
        for (const image_t image_id : ba_config.Images()) {
            const Image& image = reconstruction_->Image(image_id);
            num_images_per_camera[image.CameraId()] += 1;
        }
        
        for (const auto& camera_id_and_num_images_pair : num_images_per_camera) {
            const size_t num_reg_images_for_camera =
            num_reg_images_per_camera_.at(camera_id_and_num_images_pair.first);
            if (camera_id_and_num_images_pair.second < num_reg_images_for_camera) {
                ba_config.SetConstantCamIntrinsics(camera_id_and_num_images_pair.first);
            }
        }
        
        // Fix 7 DOF to avoid scale/rotation/translation drift in bundle adjustment.
        if (local_bundle.size() == 1) {
            ba_config.SetConstantCamPose(local_bundle[0]);
            ba_config.SetConstantCamPositions(image_id, {0});
        } else if (local_bundle.size() > 1) {
            const image_t image_id1 = local_bundle[local_bundle.size() - 1];
            const image_t image_id2 = local_bundle[local_bundle.size() - 2];
            ba_config.SetConstantCamPose(image_id1);
            if (!options.fix_existing_images ||
                !existing_image_ids_.count(image_id2)) {
                ba_config.SetConstantCamPositions(image_id2, {0});
            }
        }
        
        // Make sure, we refine all new and short-track 3D points, no matter if
        // they are fully contained in the local image set or not. Do not include
        // long track 3D points as they are usually already very stable and adding
        // to them to bundle adjustment and track merging/completion would slow
        // down the local bundle adjustment significantly.
        std::unordered_set<point3D_t> variable_point3D_ids;
        for (const point3D_t point3D_id : point3D_ids) {
            const Point3D& point3D = reconstruction_->Point3D(point3D_id);
            const size_t kMaxTrackLength = 15;
            if (!point3D.HasError() || point3D.track.Length() <= kMaxTrackLength) {
                ba_config.AddVariablePoint(point3D_id);
                variable_point3D_ids.insert(point3D_id);
            }
        }
        
        // Adjust the local bundle.
        BundleAdjuster bundle_adjuster(ba_options, ba_config);
        bundle_adjuster.Solve(reconstruction_.get());
        
        report.num_adjusted_observations =
        bundle_adjuster.Summary().num_residuals / 2;
        
        // Merge refined tracks with other existing points.
        report.num_merged_observations =
        triangulator_->MergeTracks(tri_options, variable_point3D_ids);
        // Complete tracks that may have failed to triangulate before refinement
        // of camera pose and calibration in bundle-adjustment. This may avoid
        // that some points are filtered and it helps for subsequent image
        // registrations.
        report.num_completed_observations =
        triangulator_->CompleteTracks(tri_options, variable_point3D_ids);
        report.num_completed_observations +=
        triangulator_->CompleteImage(tri_options, image_id);
    }
    
    // Filter both the modified images and all changed 3D points to make sure
    // there are no outlier points in the model. This results in duplicate work as
    // many of the provided 3D points may also be contained in the adjusted
    // images, but the filtering is not a bottleneck at this point.
    std::unordered_set<image_t> filter_image_ids;
    filter_image_ids.insert(image_id);
    filter_image_ids.insert(local_bundle.begin(), local_bundle.end());
    report.num_filtered_observations =
    obs_manager_->FilterPoints3DInImages(options.filter_max_reproj_error,
                                         options.filter_min_tri_angle,
                                         filter_image_ids);
    report.num_filtered_observations +=
    obs_manager_->FilterPoints3D(options.filter_max_reproj_error,
                                 options.filter_min_tri_angle,
                                 point3D_ids);
    
    return report;
}

bool IncrementalMapper2::AdjustGlobalBundle(const Options& options, const BundleAdjustmentOptions& ba_options) {
    THROW_CHECK_NOTNULL(reconstruction_);
    THROW_CHECK_NOTNULL(obs_manager_);
    
    const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();
    
    THROW_CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
    "registered for global "
    "bundle-adjustment";
    
    BundleAdjustmentOptions ba_options_tmp = ba_options;
    // Use stricter convergence criteria for first registered images.
    const size_t kMinNumRegImagesForFastBA = 10;
    if (reg_image_ids.size() < kMinNumRegImagesForFastBA) {
        ba_options_tmp.solver_options.function_tolerance /= 10;
        ba_options_tmp.solver_options.gradient_tolerance /= 10;
        ba_options_tmp.solver_options.parameter_tolerance /= 10;
        ba_options_tmp.solver_options.max_num_iterations *= 2;
        ba_options_tmp.solver_options.max_linear_solver_iterations = 200;
    }
    
    // Avoid degeneracies in bundle adjustment.
    obs_manager_->FilterObservationsWithNegativeDepth();
    
    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;
    for (const image_t image_id : reg_image_ids) {
        ba_config.AddImage(image_id);
    }
    
    // Fix the existing images, if option specified.
    if (options.fix_existing_images) {
        for (const image_t image_id : reg_image_ids) {
            if (existing_image_ids_.count(image_id)) {
                ba_config.SetConstantCamPose(image_id);
            }
        }
    }
    
    // Fix 7-DOFs of the bundle adjustment problem.
    ba_config.SetConstantCamPose(reg_image_ids[0]);
    if (!options.fix_existing_images ||
        !existing_image_ids_.count(reg_image_ids[1])) {
        ba_config.SetConstantCamPositions(reg_image_ids[1], {0});
    }
    
    // Run bundle adjustment.
    BundleAdjuster bundle_adjuster(ba_options_tmp, ba_config);
    return bundle_adjuster.Solve(reconstruction_.get());
}

void IncrementalMapper2::IterativeLocalRefinement(const int max_num_refinements,
                                                  const double max_refinement_change,
                                                  const Options& options,
                                                  const BundleAdjustmentOptions& ba_options,
                                                  const IncrementalTriangulator::Options& tri_options,
                                                  const image_t image_id) {
    BundleAdjustmentOptions ba_options_tmp = ba_options;
    for (int i = 0; i < max_num_refinements; ++i) {
        const auto report = AdjustLocalBundle(
                                              options, ba_options_tmp, tri_options, image_id, GetModifiedPoints3D());
        VLOG(1) << "=> Merged observations: " << report.num_merged_observations;
        VLOG(1) << "=> Completed observations: "
        << report.num_completed_observations;
        VLOG(1) << "=> Filtered observations: " << report.num_filtered_observations;
        const double changed =
        report.num_adjusted_observations == 0
        ? 0
        : (report.num_merged_observations +
           report.num_completed_observations +
           report.num_filtered_observations) /
        static_cast<double>(report.num_adjusted_observations);
        VLOG(1) << StringPrintf("=> Changed observations: %.6f", changed);
        if (changed < max_refinement_change) {
            break;
        }
        // Only use robust cost function for first iteration.
        ba_options_tmp.loss_function_type =
        BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
    }
    ClearModifiedPoints3D();
}

void IncrementalMapper2::IterativeGlobalRefinement(const int max_num_refinements,
                                                   const double max_refinement_change,
                                                   const Options& options,
                                                   const BundleAdjustmentOptions& ba_options,
                                                   const IncrementalTriangulator::Options& tri_options,
                                                   const bool normalize_reconstruction) {
    CompleteAndMergeTracks(tri_options);
    VLOG(1) << "=> Retriangulated observations: " << Retriangulate(tri_options);
    for (int i = 0; i < max_num_refinements; ++i) {
        const size_t num_observations = reconstruction_->ComputeNumObservations();
        AdjustGlobalBundle(options, ba_options);
        if (normalize_reconstruction) {
            // Normalize scene for numerical stability and
            // to avoid large scale changes in the viewer.
            reconstruction_->Normalize();
        }
        size_t num_changed_observations = CompleteAndMergeTracks(tri_options);
        num_changed_observations += FilterPoints(options);
        const double changed =
        num_observations == 0
        ? 0
        : static_cast<double>(num_changed_observations) / num_observations;
        VLOG(1) << StringPrintf("=> Changed observations: %.6f", changed);
        if (changed < max_refinement_change) {
            break;
        }
    }
}

size_t IncrementalMapper2::FilterImages(const Options& options) {
    THROW_CHECK_NOTNULL(reconstruction_);
    THROW_CHECK_NOTNULL(obs_manager_);
    THROW_CHECK(options.Check());
    
    // Do not filter images in the early stage of the reconstruction, since the
    // calibration is often still refining a lot. Hence, the camera parameters
    // are not stable in the beginning.
    const size_t kMinNumImages = 20;
    if (reconstruction_->NumRegImages() < kMinNumImages) {
        return {};
    }
    
    const std::vector<image_t> image_ids =
    obs_manager_->FilterImages(options.min_focal_length_ratio,
                               options.max_focal_length_ratio,
                               options.max_extra_param);
    
    for (const image_t image_id : image_ids) {
        DeRegisterImageEvent(image_id);
        filtered_images_.insert(image_id);
    }
    
    const size_t num_filtered_images = image_ids.size();
    VLOG(1) << "=> Filtered images: " << num_filtered_images;
    return num_filtered_images;
}

size_t IncrementalMapper2::FilterPoints(const Options& options) {
    THROW_CHECK_NOTNULL(obs_manager_);
    THROW_CHECK(options.Check());
    const size_t num_filtered_observations = obs_manager_->FilterAllPoints3D(
                                                                             options.filter_max_reproj_error, options.filter_min_tri_angle);
    VLOG(1) << "=> Filtered observations: " << num_filtered_observations;
    return num_filtered_observations;
}

std::shared_ptr<class Reconstruction> IncrementalMapper2::Reconstruction()
const {
    return reconstruction_;
}

class ObservationManager& IncrementalMapper2::ObservationManager() const {
    THROW_CHECK_NOTNULL(obs_manager_);
    return *obs_manager_;
}

IncrementalTriangulator& IncrementalMapper2::Triangulator() const {
    THROW_CHECK_NOTNULL(triangulator_);
    return *triangulator_;
}

const std::unordered_set<image_t>& IncrementalMapper2::FilteredImages() const {
    return filtered_images_;
}

const std::unordered_set<image_t>& IncrementalMapper2::ExistingImageIds() const {
    return existing_image_ids_;
}

const std::unordered_map<camera_t, size_t>&
IncrementalMapper2::NumRegImagesPerCamera() const {
    return num_reg_images_per_camera_;
}

size_t IncrementalMapper2::NumTotalRegImages() const {
    return num_total_reg_images_;
}

size_t IncrementalMapper2::NumSharedRegImages() const {
    return num_shared_reg_images_;
}

const std::unordered_set<point3D_t>& IncrementalMapper2::GetModifiedPoints3D() {
    return triangulator_->GetModifiedPoints3D();
}

void IncrementalMapper2::ClearModifiedPoints3D() {
    triangulator_->ClearModifiedPoints3D();
}

std::vector<image_t> IncrementalMapper2::FindFirstInitialImage(const Options& options) const {
    // Struct to hold meta-data for ranking images.
    struct ImageInfo {
        image_t image_id;
        bool prior_focal_length;
        image_t num_correspondences;
    };
    
    const size_t init_max_reg_trials =
    static_cast<size_t>(options.init_max_reg_trials);
    
    // Collect information of all not yet registered images with
    // correspondences.
    std::vector<ImageInfo> image_infos;
    image_infos.reserve(reconstruction_->NumImages());
    for (const auto& image : reconstruction_->Images()) {
        // Only images with correspondences can be registered.
        if (obs_manager_->NumCorrespondences(image.first) == 0) {
            continue;
        }
        
        // Only use images for initialization a maximum number of times.
        if (init_num_reg_trials_.count(image.first) &&
            init_num_reg_trials_.at(image.first) >= init_max_reg_trials) {
            continue;
        }
        
        // Only use images for initialization that are not registered in any
        // of the other reconstructions.
        if (num_registrations_.count(image.first) > 0 &&
            num_registrations_.at(image.first) > 0) {
            continue;
        }
        
        const struct Camera& camera =
        reconstruction_->Camera(image.second.CameraId());
        ImageInfo image_info;
        image_info.image_id = image.first;
        image_info.prior_focal_length = camera.has_prior_focal_length;
        image_info.num_correspondences =
        obs_manager_->NumCorrespondences(image.first);
        image_infos.push_back(image_info);
    }
    
    // Sort images such that images with a prior focal length and more
    // correspondences are preferred, i.e. they appear in the front of the list.
    std::sort(
              image_infos.begin(),
              image_infos.end(),
              [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
                  if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
                      return true;
                  } else if (!image_info1.prior_focal_length &&
                             image_info2.prior_focal_length) {
                      return false;
                  } else {
                      return image_info1.num_correspondences >
                      image_info2.num_correspondences;
                  }
              });
    
    // Extract image identifiers in sorted order.
    std::vector<image_t> image_ids;
    image_ids.reserve(image_infos.size());
    for (const ImageInfo& image_info : image_infos) {
        image_ids.push_back(image_info.image_id);
    }
    
    return image_ids;
}

std::vector<image_t> IncrementalMapper2::FindSecondInitialImage(const Options& options, const image_t image_id1) const {
    const std::shared_ptr<const CorrespondenceGraph> correspondence_graph =
    database_cache_->CorrespondenceGraph();
    // Collect images that are connected to the first seed image and have
    // not been registered before in other reconstructions.
    const class Image& image1 = reconstruction_->Image(image_id1);
    std::unordered_map<image_t, point2D_t> num_correspondences;
    for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D();
         ++point2D_idx) {
        const auto corr_range =
        correspondence_graph->FindCorrespondences(image_id1, point2D_idx);
        for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
            if (num_registrations_.count(corr->image_id) == 0 ||
                num_registrations_.at(corr->image_id) == 0) {
                num_correspondences[corr->image_id] += 1;
            }
        }
    }
    
    // Struct to hold meta-data for ranking images.
    struct ImageInfo {
        image_t image_id;
        bool prior_focal_length;
        point2D_t num_correspondences;
    };
    
    const size_t init_min_num_inliers =
    static_cast<size_t>(options.init_min_num_inliers);
    
    // Compose image information in a compact form for sorting.
    std::vector<ImageInfo> image_infos;
    image_infos.reserve(reconstruction_->NumImages());
    for (const auto elem : num_correspondences) {
        if (elem.second >= init_min_num_inliers) {
            const class Image& image = reconstruction_->Image(elem.first);
            const struct Camera& camera = reconstruction_->Camera(image.CameraId());
            ImageInfo image_info;
            image_info.image_id = elem.first;
            image_info.prior_focal_length = camera.has_prior_focal_length;
            image_info.num_correspondences = elem.second;
            image_infos.push_back(image_info);
        }
    }
    
    // Sort images such that images with a prior focal length and more
    // correspondences are preferred, i.e. they appear in the front of the list.
    std::sort(
              image_infos.begin(),
              image_infos.end(),
              [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
                  if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
                      return true;
                  } else if (!image_info1.prior_focal_length &&
                             image_info2.prior_focal_length) {
                      return false;
                  } else {
                      return image_info1.num_correspondences >
                      image_info2.num_correspondences;
                  }
              });
    
    // Extract image identifiers in sorted order.
    std::vector<image_t> image_ids;
    image_ids.reserve(image_infos.size());
    for (const ImageInfo& image_info : image_infos) {
        image_ids.push_back(image_info.image_id);
    }
    
    return image_ids;
}

std::vector<image_t> IncrementalMapper2::FindLocalBundle(const Options& options, const image_t image_id) const {
    THROW_CHECK(options.Check());
    
    const Image& image = reconstruction_->Image(image_id);
    THROW_CHECK(image.IsRegistered());
    
    // Extract all images that have at least one 3D point with the query image
    // in common, and simultaneously count the number of common 3D points.
    
    std::unordered_map<image_t, size_t> shared_observations;
    
    std::unordered_set<point3D_t> point3D_ids;
    point3D_ids.reserve(image.NumPoints3D());
    
    for (const Point2D& point2D : image.Points2D()) {
        if (point2D.HasPoint3D()) {
            point3D_ids.insert(point2D.point3D_id);
            const Point3D& point3D = reconstruction_->Point3D(point2D.point3D_id);
            for (const TrackElement& track_el : point3D.track.Elements()) {
                if (track_el.image_id != image_id) {
                    shared_observations[track_el.image_id] += 1;
                }
            }
        }
    }
    
    // Sort overlapping images according to number of shared observations.
    
    std::vector<std::pair<image_t, size_t>> overlapping_images(shared_observations.begin(), shared_observations.end());
    std::sort(overlapping_images.begin(),
              overlapping_images.end(),
              [](const std::pair<image_t, size_t>& image1,
                 const std::pair<image_t, size_t>& image2) {
        return image1.second > image2.second;
    });
    
    // The local bundle is composed of the given image and its most connected
    // neighbor images, hence the subtraction of 1.
    
    const size_t num_images =
    static_cast<size_t>(options.local_ba_num_images - 1);
    const size_t num_eff_images = std::min(num_images, overlapping_images.size());
    
    // Extract most connected images and ensure sufficient triangulation angle.
    
    std::vector<image_t> local_bundle_image_ids;
    local_bundle_image_ids.reserve(num_eff_images);
    
    // If the number of overlapping images equals the number of desired images in
    // the local bundle, then simply copy over the image identifiers.
    if (overlapping_images.size() == num_eff_images) {
        for (const auto& overlapping_image : overlapping_images) {
            local_bundle_image_ids.push_back(overlapping_image.first);
        }
        return local_bundle_image_ids;
    }
    
    // In the following iteration, we start with the most overlapping images and
    // check whether it has sufficient triangulation angle. If none of the
    // overlapping images has sufficient triangulation angle, we relax the
    // triangulation angle threshold and start from the most overlapping image
    // again. In the end, if we still haven't found enough images, we simply use
    // the most overlapping images.
    
    const double min_tri_angle_rad = DegToRad(options.local_ba_min_tri_angle);
    
    // The selection thresholds (minimum triangulation angle, minimum number of
    // shared observations), which are successively relaxed.
    const std::array<std::pair<double, double>, 8> selection_thresholds = {{
        std::make_pair(min_tri_angle_rad / 1.0, 0.6 * image.NumPoints3D()),
        std::make_pair(min_tri_angle_rad / 1.5, 0.6 * image.NumPoints3D()),
        std::make_pair(min_tri_angle_rad / 2.0, 0.5 * image.NumPoints3D()),
        std::make_pair(min_tri_angle_rad / 2.5, 0.4 * image.NumPoints3D()),
        std::make_pair(min_tri_angle_rad / 3.0, 0.3 * image.NumPoints3D()),
        std::make_pair(min_tri_angle_rad / 4.0, 0.2 * image.NumPoints3D()),
        std::make_pair(min_tri_angle_rad / 5.0, 0.1 * image.NumPoints3D()),
        std::make_pair(min_tri_angle_rad / 6.0, 0.1 * image.NumPoints3D()),
    }};
    
    const Eigen::Vector3d proj_center = image.ProjectionCenter();
    std::vector<Eigen::Vector3d> shared_points3D;
    shared_points3D.reserve(image.NumPoints3D());
    std::vector<double> tri_angles(overlapping_images.size(), -1.0);
    std::vector<char> used_overlapping_images(overlapping_images.size(), false);
    
    for (const auto& selection_threshold : selection_thresholds) {
        for (size_t overlapping_image_idx = 0;
             overlapping_image_idx < overlapping_images.size();
             ++overlapping_image_idx) {
            // Check if the image has sufficient overlap. Since the images are ordered
            // based on the overlap, we can just skip the remaining ones.
            if (overlapping_images[overlapping_image_idx].second <
                selection_threshold.second) {
                break;
            }
            
            // Check if the image is already in the local bundle.
            if (used_overlapping_images[overlapping_image_idx]) {
                continue;
            }
            
            const auto& overlapping_image = reconstruction_->Image(
                                                                   overlapping_images[overlapping_image_idx].first);
            const Eigen::Vector3d overlapping_proj_center =
            overlapping_image.ProjectionCenter();
            
            // In the first iteration, compute the triangulation angle. In later
            // iterations, reuse the previously computed value.
            double& tri_angle = tri_angles[overlapping_image_idx];
            if (tri_angle < 0.0) {
                // Collect the commonly observed 3D points.
                shared_points3D.clear();
                for (const Point2D& point2D : overlapping_image.Points2D()) {
                    if (point2D.HasPoint3D() && point3D_ids.count(point2D.point3D_id)) {
                        shared_points3D.push_back(
                                                  reconstruction_->Point3D(point2D.point3D_id).xyz);
                    }
                }
                
                // Calculate the triangulation angle at a certain percentile.
                const double kTriangulationAnglePercentile = 75;
                tri_angle = Percentile(CalculateTriangulationAngles(proj_center, overlapping_proj_center, shared_points3D), kTriangulationAnglePercentile);
            }
            
            // Check that the image has sufficient triangulation angle.
            if (tri_angle >= selection_threshold.first) {
                local_bundle_image_ids.push_back(overlapping_image.ImageId());
                used_overlapping_images[overlapping_image_idx] = true;
                // Check if we already collected enough images.
                if (local_bundle_image_ids.size() >= num_eff_images) {
                    break;
                }
            }
        }
        
        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
            break;
        }
    }
    
    // In case there are not enough images with sufficient triangulation angle,
    // simply fill up the rest with the most overlapping images.
    
    if (local_bundle_image_ids.size() < num_eff_images) {
        for (size_t overlapping_image_idx = 0;
             overlapping_image_idx < overlapping_images.size();
             ++overlapping_image_idx) {
            // Collect image if it is not yet in the local bundle.
            if (!used_overlapping_images[overlapping_image_idx]) {
                local_bundle_image_ids.push_back(overlapping_images[overlapping_image_idx].first);
                used_overlapping_images[overlapping_image_idx] = true;
                
                // Check if we already collected enough images.
                if (local_bundle_image_ids.size() >= num_eff_images) {
                    break;
                }
            }
        }
    }
    
    return local_bundle_image_ids;
}

void IncrementalMapper2::RegisterImageEvent(const image_t image_id) {
    const Image& image = reconstruction_->Image(image_id);
    size_t& num_reg_images_for_camera =
    num_reg_images_per_camera_[image.CameraId()];
    num_reg_images_for_camera += 1;
    
    size_t& num_regs_for_image = num_registrations_[image_id];
    num_regs_for_image += 1;
    if (num_regs_for_image == 1) {
        num_total_reg_images_ += 1;
    } else if (num_regs_for_image > 1) {
        num_shared_reg_images_ += 1;
    }
}

void IncrementalMapper2::DeRegisterImageEvent(const image_t image_id) {
    const Image& image = reconstruction_->Image(image_id);
    size_t& num_reg_images_for_camera =
    num_reg_images_per_camera_.at(image.CameraId());
    THROW_CHECK_GT(num_reg_images_for_camera, 0);
    num_reg_images_for_camera -= 1;
    
    size_t& num_regs_for_image = num_registrations_[image_id];
    num_regs_for_image -= 1;
    if (num_regs_for_image == 0) {
        num_total_reg_images_ -= 1;
    } else if (num_regs_for_image > 0) {
        num_shared_reg_images_ -= 1;
    }
}

bool IncrementalMapper2::EstimateInitialTwoViewGeometry(const Options& options,
                                                        TwoViewGeometry& two_view_geometry,
                                                        const image_t image_id1,
                                                        const image_t image_id2) {
    const Image& image1 = database_cache_->Image(image_id1);
    const Camera& camera1 = database_cache_->Camera(image1.CameraId());
    
    const Image& image2 = database_cache_->Image(image_id2);
    const Camera& camera2 = database_cache_->Camera(image2.CameraId());
    
    const FeatureMatches matches =
    database_cache_->CorrespondenceGraph()->FindCorrespondencesBetweenImages(
                                                                             image_id1, image_id2);
    
    std::vector<Eigen::Vector2d> points1;
    points1.reserve(image1.NumPoints2D());
    for (const auto& point : image1.Points2D()) {
        points1.push_back(point.xy);
    }
    
    std::vector<Eigen::Vector2d> points2;
    points2.reserve(image2.NumPoints2D());
    for (const auto& point : image2.Points2D()) {
        points2.push_back(point.xy);
    }
    
    TwoViewGeometryOptions two_view_geometry_options;
    two_view_geometry_options.ransac_options.min_num_trials = 30;
    two_view_geometry_options.ransac_options.max_error = options.init_max_error;
    two_view_geometry = EstimateCalibratedTwoViewGeometry(
                                                          camera1, points1, camera2, points2, matches, two_view_geometry_options);
    
    if (!EstimateTwoViewGeometryPose(
                                     camera1, points1, camera2, points2, &two_view_geometry)) {
                                         return false;
                                     }
    
    if (static_cast<int>(two_view_geometry.inlier_matches.size()) >=
        options.init_min_num_inliers &&
        std::abs(two_view_geometry.cam2_from_cam1.translation.z()) <
        options.init_max_forward_motion &&
        two_view_geometry.tri_angle > DegToRad(options.init_min_tri_angle)) {
        return true;
    }
    
    return false;
}

}  // namespace colmap

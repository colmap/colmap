//
//  localize.cpp
//  colmap
//  Created by Lukasz Karluk on 15/6/2024.
//

#include "colmap/exe/localize.h"
#include "colmap/exe/feature.h"
#include "colmap/exe/gui.h"

#include "colmap/controllers/localization.h"
#include "colmap/controllers/option_manager.h"

#include "colmap/util/opengl_utils.h"

namespace colmap {

int RunLocalizer(int argc, char** argv) {
    std::string image_to_localize;
    std::string sparse_path;
    int camera_mode = -1;
    std::string descriptor_normalization = "l1_root";
    
    OptionManager options;
    options.AddDatabaseOptions();
    options.AddImageOptions();
    options.AddDefaultOption("camera_mode", &camera_mode);
    options.AddDefaultOption("image_to_localize", &image_to_localize);
    options.AddDefaultOption("sparse_path", &sparse_path);
    options.AddDefaultOption("descriptor_normalization",
                             &descriptor_normalization,
                             "{'l1_root', 'l2'}");
    options.AddExtractionOptions();
    options.AddMapperOptions();
    options.Parse(argc, argv);
    
    if (image_to_localize.empty()) {
        LOG(ERROR) << "Missing arg image_to_localize`";
        return EXIT_FAILURE;
    }
    
    if (!ExistsDir(sparse_path)) {
      LOG(ERROR) << "`sparse_path` is not a directory";
      return EXIT_FAILURE;
    }
    
    ImageReaderOptions reader_options = *options.image_reader;
    reader_options.database_path = *options.database_path;
    reader_options.image_path = *options.image_path;
    reader_options.image_list.push_back(image_to_localize);
    
    if (camera_mode >= 0) {
        UpdateImageReaderOptionsFromCameraMode(reader_options,
                                               (CameraMode)camera_mode);
    }
    
    StringToLower(&descriptor_normalization);
    if (descriptor_normalization == "l1_root") {
        options.sift_extraction->normalization =
        SiftExtractionOptions::Normalization::L1_ROOT;
    } else if (descriptor_normalization == "l2") {
        options.sift_extraction->normalization =
        SiftExtractionOptions::Normalization::L2;
    } else {
        LOG(ERROR) << "Invalid `descriptor_normalization`";
        return EXIT_FAILURE;
    }
    
    if (!ExistsCameraModelWithName(reader_options.camera_model)) {
        LOG(ERROR) << "Camera model does not exist";
    }
    
    if (!VerifyCameraParams(reader_options.camera_model,
                            reader_options.camera_params)) {
        return EXIT_FAILURE;
    }
    
    if (!VerifySiftGPUParams(options.sift_extraction->use_gpu)) {
        return EXIT_FAILURE;
    }
    
    std::unique_ptr<QApplication> app;
    if (options.sift_extraction->use_gpu && kUseOpenGL) {
        app.reset(new QApplication(argc, argv));
    }
    
    auto feature_extractor = CreateFeatureExtractorController2(reader_options, *options.sift_extraction);
    
    if (options.sift_extraction->use_gpu && kUseOpenGL) {
        RunThreadWithOpenGLContext(feature_extractor.get());
    } else {
        feature_extractor->Start();
        feature_extractor->Wait();
    }
    
    ImageData imageData = GetImageData( feature_extractor );
    
    PrintHeading1("Loading database");  

    std::shared_ptr<DatabaseCache> database_cache;
    {
      Timer timer;
      timer.Start();
      const size_t min_num_matches =
          static_cast<size_t>(options.mapper->min_num_matches);
      database_cache = DatabaseCache::Create(Database(*options.database_path),
                                             min_num_matches,
                                             options.mapper->ignore_watermarks,
                                             options.mapper->image_names);
      timer.PrintMinutes();
    }

    auto reconstruction = std::make_shared<Reconstruction>();
    reconstruction->Read(sparse_path);

    IncrementalMapper2 mapper(database_cache);
    mapper.BeginReconstruction(reconstruction);
    const auto mapper_options = options.mapper->Mapper();
    mapper.RegisterNextImage(mapper_options, imageData);
    mapper.EndReconstruction(/*discard=*/false);

    return EXIT_SUCCESS;
}

} // namespace colmap

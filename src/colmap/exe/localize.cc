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
    int camera_mode = -1;
    std::string descriptor_normalization = "l1_root";
    
    OptionManager options;
    options.AddDatabaseOptions();
    options.AddImageOptions();
    options.AddDefaultOption("camera_mode", &camera_mode);
    options.AddDefaultOption("image_to_localize", &image_to_localize);
    options.AddDefaultOption("descriptor_normalization",
                             &descriptor_normalization,
                             "{'l1_root', 'l2'}");
    options.AddExtractionOptions();
    options.Parse(argc, argv);
    
    if (image_to_localize.empty()) {
        LOG(ERROR) << "Missing arg image_to_localize`";
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
    
    auto feature_extractor = CreateFeatureExtractorController2(
                                                               reader_options, *options.sift_extraction);
    
    if (options.sift_extraction->use_gpu && kUseOpenGL) {
        RunThreadWithOpenGLContext(feature_extractor.get());
    } else {
        feature_extractor->Start();
        feature_extractor->Wait();
    }
    
    ImageData & imageData = GetImageData( feature_extractor );
    
    return EXIT_SUCCESS;
}

} // namespace colmap

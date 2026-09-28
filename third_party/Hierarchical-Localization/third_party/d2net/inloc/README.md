# InLoc evaluation instructions

Start by downloading the [InLoc_demo](https://github.com/HajimeTaira/InLoc_demo) code. Once it is up and running according to the official instruction, you can copy and paste all the files available here overwriting the `Features_WUSTL` and `parfor_sparseGV` functions. `generate_list.m` will generate `image_list.txt` containing the queries and top 100 database matches (run `sort -u image_list.txt > image_list_unique.txt` to remove the duplicates). After extracting features for all the images in `image_list_unique.txt`, you can run `custom_demo` directly. 

The feature extraction part for D2-Net can be done using the following command: `python extract_features.py --image_list_file /path/to/image_list_unique.txt --multiscale --output_format .mat`.

In case you plan on using your own features, don't forget to change the extension in `Features_WUSTL.m`. The local features are supposed to be stored in the `mat` format with two fields:

- `keypoints` - `N x 3` matrix with `x, y, scale` coordinates of each keypoint in COLMAP format (the `X` axis points to the right, the `Y` axis to the bottom),

- `descriptors` - `N x D` matrix with the descriptors.

The evaluation pipeline is live at [visuallocalization.net](https://www.visuallocalization.net/). In order to generate a submission file, please use the provided [ImgList2text](https://github.com/HajimeTaira/InLoc_demo/blob/master/functions/utils/ImgList2text.m) function.

We have also provided the `merge_files` MATLAB script that was used to merge the solutions of D2-Net Multiscale and Dense InLoc based on the view synthesis score. It can be used as follows `merge_files('output/densePV_top10_shortlist_method1.mat', 'outputs/densePV_top10_shortlist_method2.mat')`.
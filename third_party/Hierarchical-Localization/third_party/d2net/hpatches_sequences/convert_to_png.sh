# DELF Extraction script doesn't support .ppm images.
current_dir=`pwd`
echo $current_dir
for dir in `ls hpatches-sequences-release`; do
    echo $dir
    cd hpatches-sequences-release/$dir
    mogrify -format png *.ppm
    cd $current_dir
done

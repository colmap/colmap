

DATA=/home/GGS/workspace/test/relo_loop/r1_p2_1/data/
# DATA=/home/GGS/20220907175843/cam0/
start=`date +%s`
run_time="spend time\n"

image_path=$DATA/rgb
if [ ! -d $image_path ];then
    echo "$image_path Error"
    exit 1
fi


cur_time=$(date "+%Y%m%d%H%M%S")

[ -z ${NAME+x} ] && NAME=$cur_time
work_path=$DATA/clmp_pose_$NAME
mkdir $work_path

shell_path="$( cd "$( dirname "$BASH_SOURCE[0]" )" && pwd )"
CLMP=$shell_path/../../build/src/exe/colmap
VOCAB=$shell_path/vocab_tree_flickr100K_words32K.bin

# python $shell_path/generate_mvs_project.py --project_path $work_path --dataset_path $image_path/..

temp_time=`date +%s`
$CLMP feature_extractor --database_path $work_path/database.db --image_path $image_path --image_list $DATA/KeyFrameTrajectory.txt \
--ImageReader.camera_model PINHOLE --SiftExtraction.num_threads $(nproc) | tee -a $work_path/featurer_$cur_time.log
end=`date +%s`
dif=$[ end - temp_time ]
run_time=$run_time"featurer $dif s\n"
temp_time=`date +%s`

# --SequentialMatching.loop_detection_num_images 200 --SequentialMatching.loop_detection_num_checks 500 \
$CLMP sequential_matcher --database_path $work_path/database.db --SequentialMatching.loop_detection 1  \
--SequentialMatching.output_index_path $DATA/index.vcb \
--SiftMatching.num_threads $(nproc) --SequentialMatching.vocab_tree_path $VOCAB | tee -a $work_path/matcher_$cur_time.log
end=`date +%s`
dif=$[ end - temp_time ]
run_time=$run_time"matcher $dif s\n"
temp_time=`date +%s`

mkdir $work_path/sparse
$CLMP point_triangulator --database_path $work_path/database.db --image_path $image_path --Mapper.num_threads $(nproc) \
--image_list_path $DATA/KeyFrameTrajectory.txt \
--camera_file_path  $DATA/camera.txt  --output_path $work_path/sparse | tee -a $work_path/triangulator_$cur_time.log
end=`date +%s`
dif=$[ end - temp_time ]
run_time=$run_time"triangulator $dif s\n"
temp_time=`date +%s`


run_time=$run_time"all $dif s\n"

echo -e $run_time | tee -a $work_path/spend.log


exit 0
#!/bin/bash

# 脚本名称和用法提示
script_name=$(basename "$0")
usage() {
    echo "Usage: $script_name [colmap_command] [arguments...]"
    exit 1
}

# 检查至少有一个参数（colmap_command）
if [ "$#" -lt 1 ]; then
    echo "Error: No Colmap command specified."
    usage
fi

# 第一个参数是 Colmap 命令
colmap_command="$1"

# 移除第一个参数，剩下的是传递给 Colmap 的参数
shift

./build/src/colmap/exe/colmap "$colmap_command" "$@"

# 检查 Colmap 命令的执行状态
if [ $? -ne 0 ]; then
    echo "Error: Colmap command failed."
    usage
fi


# ./build/src/colmap/exe/colmap gui # bash develop/run.sh gui
# ./build/src/colmap/exe/colmap feature_extractor --image_path */images/ --database_path */database.db
# ./build/src/colmap/exe/colmap exhaustive_matcher --database_path */database.db
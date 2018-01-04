#!/bin/bash

check_command() {
    for c in "${commands[@]}"; do
        if [ "$command" == "$c" ]; then
            command_exists=1
            return
        fi
    done
    command_exists=0
}

show_help() {
    echo "COLMAP -- Structure-from-Motion and Multi-View Stereo"
    echo ""
    echo "Usage:"
    echo "  colmap [command] [options]"
    echo ""
    echo "Example usage:"
    echo "  colmap help [ -h, --help ]"
    echo "  colmap gui"
    echo "  colmap gui -h [ --help ]"
    echo "  colmap feature_extractor --image_path IMAGES --database_path DATABASE"
    echo "  colmap exhaustive_matcher --database_path DATABASE"
    echo "  colmap mapper --image_path IMAGES --database_path DATABASE --export_path EXPORT"
    echo "  ..."
    echo ""
    echo "Documentation:"
    echo "  https://colmap.github.io/"
    echo ""
    echo "Available commands:"
    for c in "${commands[@]}"
    do
        if [[ ! $c = *_test ]]; then
            echo "  $c"
        fi
    done
    exit
}

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_path=$script_path/../lib/colmap
command=$1
commands=("help" "gui" $(ls $lib_path))

check_command

if [ "$command_exists" == 0 ] || \
   [ "$command" == "" ] || \
   [ "$command" == "help" ] || \
   [ "$command" == "-h" ] || \
   [ "$command" == "--help" ]; then
    show_help
elif [ "$command" == "gui" ]; then
    command="colmap"
fi

shift
$lib_path/$command $@

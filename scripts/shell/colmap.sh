#!/bin/bash

array_contains() {
    array=$1
    value=$2
    for entry in "${array[@]}"; do
        if [ "$entry" == "$value" ]; then
            echo "1"
            return
        fi
    done
    echo "0"
}

show_help() {
    echo "Example usage:"
    echo "  colmap -h [ --help ]"
    echo "  colmap gui"
    echo "  colmap gui -h [ --help ]"
    echo "  colmap feature_extractor --project_path PROJECT"
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
colmap_internal_path=$script_path/colmap-internal
commands=("gui" $(ls $colmap_internal_path))

command=$1
command_exists=$(array_contains $commands $command)

if [ "$command_exists" == "0" ] || [ "$command" == "" ] || [ "$command" == "-h" ] || [ "$command" == "--help" ]; then
    show_help
elif [ "$command" == "gui" ]; then
    command="colmap"
fi

shift
$colmap_internal_path/$command $@

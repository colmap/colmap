set -e
PYTHON_EXEC=$1
OUTPUT=$2
echo "Building stubs with $PYTHON_EXEC to $OUTPUT"
$PYTHON_EXEC -m pybind11_stubgen pycolmap -o $OUTPUT \
        --numpy-array-use-type-var \
        --enum-class-locations=.+:pycolmap \
        --ignore-invalid-expressions "ceres::*" \
        --print-invalid-expressions-as-is \
        --print-safe-value-reprs "[a-zA-Z]+Options\(\)"
mv "$OUTPUT/pycolmap" "$OUTPUT/pycolmap-stubs"
FILES="$OUTPUT/pycolmap-stubs/*.pyi"

perl -i -pe's/: ceres::([a-zA-Z]|::)+//g' $FILES
perl -i -pe's/ -> ceres::([a-zA-Z]|::)+:$/:/g' $FILES

COLMAP_DIR=$(dirname $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ))
$PYTHON_EXEC -m ruff format --config ${COLMAP_DIR}/ruff.toml ${FILES}

set -e
PYTHON_EXEC=$1
OUTPUT=$2
PACKAGE_NAME="_core"
echo "Building stubs with $PYTHON_EXEC to $OUTPUT"
$PYTHON_EXEC -m pybind11_stubgen $PACKAGE_NAME -o $OUTPUT \
        --numpy-array-use-type-var \
        --enum-class-locations=.+:$PACKAGE_NAME \
        --ignore-invalid-expressions "ceres::*" \
        --print-invalid-expressions-as-is \
        --print-safe-value-reprs "[a-zA-Z]+Options\(\)"
FILES=$(find $OUTPUT/$PACKAGE_NAME/ -name '*.pyi' -type f)

perl -i -pe's/\b_core\b/pycolmap/g' $FILES
perl -i -pe's/: ceres::([a-zA-Z]|::)+//g' $FILES
perl -i -pe's/ -> ceres::([a-zA-Z]|::)+:$/:/g' $FILES

# pybind issue, will not be fixed: https://github.com/pybind/pybind11/pull/2277
perl -i -pe's/(?<=\b__(eq|ne)__\(self, )arg0: [a-zA-Z0-9_]+\)/other: object)/g' $FILES

# mypy bug: https://github.com/python/mypy/issues/4266
perl -i -pe's/(__hash__:? .*= None)$/\1  # type: ignore/g' $FILES

COLMAP_DIR=$(dirname $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ))
ruff format --config ${COLMAP_DIR}/ruff.toml ${FILES}

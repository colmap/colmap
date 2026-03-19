#!/bin/bash

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

# native_enum stubs: convert enum member annotations to assignments for mypy.
# pybind11-stubgen generates `MEMBER: typing.ClassVar[EnumType]  # value = ...`
# but mypy requires `MEMBER = value` for enum members.
# Handles both top-level and nested enum classes at any indentation level.
perl -i -pe'
    if (/^(\s*)class \w+\(enum\.IntEnum\)/) { $in_enum = 1; $indent = $1 . "    " }
    elsif ($in_enum && /^(\s*)\S/ && length($1) <= length($indent) - 4) { $in_enum = 0 }
    if ($in_enum) { s/^($indent\w+): typing\.ClassVar\[.*?\].*$/$1 = .../ }
' $FILES

# pybind issue, will not be fixed: https://github.com/pybind/pybind11/pull/2277
perl -i -pe's/(?<=\b__(eq|ne)__\(self, )arg0: [a-zA-Z0-9_]+\)/other: object)/g' $FILES

# mypy bug: https://github.com/python/mypy/issues/4266
perl -i -pe's/(__hash__:? .*= None)$/\1  # type: ignore/g' $FILES

# pybind issue: parameter and key types should not be cast to the more generic types.
perl -i -pe's/Mapping\[typing\.Supports(Int|Float)/Mapping\[\L\1/g' $FILES
perl -i -pe's/Mapping\[(int|float) \| typing\.SupportsIndex/Mapping\[$1/g' $FILES
perl -i -pe's/typing\.SupportsInt \| typing\.SupportsIndex/int/g' $FILES
perl -i -pe's/typing\.SupportsFloat \| typing\.SupportsIndex/float/g' $FILES

COLMAP_DIR=$(dirname $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ))
ruff format --config ${COLMAP_DIR}/ruff.toml ${FILES}

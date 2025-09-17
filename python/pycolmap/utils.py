from collections.abc import MutableSequence
from types import ModuleType
from typing import Any, Optional


def import_module_symbols(
    dst_vars: dict[str, Any],
    src_module: ModuleType,
    exclude: Optional[set[str]] = None,
) -> MutableSequence[str]:
    symbols = {}
    for n, s in vars(src_module).items():
        if n.startswith("_"):
            continue
        if exclude is not None and n in exclude:
            continue
        symbols[n] = s
    dst_vars.update(symbols)
    return list(symbols)

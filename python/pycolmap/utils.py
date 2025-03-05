from types import ModuleType
from typing import Any, Dict, MutableSequence, Optional, Set


def import_module_symbols(
    dst_vars: Dict[str, Any],
    src_module: ModuleType,
    exclude: Optional[Set[str]] = None,
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

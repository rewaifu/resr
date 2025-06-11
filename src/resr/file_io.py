import os
from typing import Iterator, Any


def scandir(dir_path: str, suffix: str | None = None, recursive: bool = False) -> Iterator[Any]:
    if (suffix is not None) and not isinstance(suffix, str | tuple):
        raise TypeError('"suffix" must be a string or tuple of strings')

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if suffix is None or entry.path.endswith(suffix):
                    yield entry.path
            elif recursive:
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
            else:
                continue

    for path in sorted(_scandir(dir_path, suffix=suffix, recursive=recursive)):
        yield path

import os
from collections.abc import Iterator
from pathlib import Path


def scandir(dir_path: str | Path, suffix: str | tuple[str, ...] | None = None, recursive: bool = False) -> Iterator[Path]:
    if suffix is not None and not isinstance(suffix, (str, tuple)):
        msg = '"suffix" must be a string or tuple of strings'
        raise TypeError(msg)

    base = str(dir_path)

    def _gen(path: str):
        with os.scandir(path) as it:
            for entry in it:
                name = entry.name
                if name.startswith('.'):
                    continue
                if entry.is_file(follow_symlinks=False):
                    if suffix is None or name.endswith(suffix):
                        yield Path(entry.path)
                elif recursive and entry.is_dir(follow_symlinks=False):
                    yield from _gen(entry.path)

    yield from sorted(_gen(base), key=lambda p: p.name)

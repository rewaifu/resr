from collections.abc import Iterator
from pathlib import Path


def scandir(dir_path: str | Path, suffix: str | tuple[str, ...] | None = None, recursive: bool = False) -> Iterator[Path]:
    if suffix is not None and not isinstance(suffix, (str, tuple)):
        msg = '"suffix" must be a string or tuple of strings'
        raise TypeError(msg)

    dir_path = Path(dir_path)

    def _scandir(current_path: Path, suffix, recursive):
        for entry in current_path.iterdir():
            if entry.name.startswith('.'):
                continue
            if entry.is_file():
                if suffix is None or entry.name.endswith(suffix):
                    yield entry
            elif recursive and entry.is_dir():
                yield from _scandir(entry, suffix, recursive)

    yield from sorted(_scandir(dir_path, suffix=suffix, recursive=recursive))

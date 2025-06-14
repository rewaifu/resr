from abc import ABC, abstractmethod

Size = tuple[int, int]


class Tiler(ABC):
    @abstractmethod
    def starting_tile_size(self, width: int, height: int, channels: int) -> Size:
        pass

    def split(self, tile_size: Size) -> Size:
        w, h = tile_size
        assert w >= 16
        assert h >= 16
        return max(16, w // 2), max(16, h // 2)


class NoTiling(Tiler):
    def starting_tile_size(self, width: int, height: int, _channels: int) -> Size:
        size = max(width, height)
        return size, size

    def split(self, _tile_size: Size) -> Size:
        msg = 'Image cannot be upscaled with No Tiling mode.'
        raise ValueError(msg)


class MaxTileSize(Tiler):
    def __init__(self, tile_size: int = 2**31) -> None:
        self.tile_size = tile_size

    def starting_tile_size(self, width: int, height: int, _channels: int) -> Size:
        max_tile_size = max(width + 10, height + 10)
        size = min(self.tile_size, max_tile_size)
        return size, size


class ExactTileSize(Tiler):
    def __init__(self, size: Size | int) -> None:
        self.exact_size = (size, size) if isinstance(size, int) else size

    def starting_tile_size(self, _width: int, _height: int, _channels: int) -> Size:
        return self.exact_size

    def split(self, _tile_size: Size) -> Size:
        msg = f'Splits not supported for exact size ({self.exact_size[0]}x{self.exact_size[1]}px). Not enough VRAM to run the current model.'
        raise ValueError(
            msg
        )

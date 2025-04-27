from dataclasses import dataclass

import numpy as np
import torch

from .tile_blender import BlendDirection, TileOverlap, TileBlender
from .tiler import Tiler
from ..img_util import get_h_w_c, image2tensor, tensor2image


@dataclass
class Segment:
    start: int
    end: int
    start_padding: int
    end_padding: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def padded_length(self) -> int:
        return self.end + self.end_padding - (self.start - self.start_padding)


def split_into_segments(length: int, tile_size: int, overlap: int) -> list[Segment]:
    if length <= tile_size:
        return [Segment(0, length, 0, 0)]

    assert tile_size > overlap * 2

    result = [Segment(0, tile_size - overlap, 0, overlap)]

    while result[-1].end < length:
        start = result[-1].end
        end = start + tile_size - overlap * 2

        if end + overlap >= length:
            result.append(Segment(start, length, overlap, 0))
        else:
            result.append(Segment(start, end, overlap, overlap))

    return result


def process_tiles(
    img: np.ndarray,
    model: torch.nn.Module,
    scale: int,
    tiler: Tiler,
    channels: int = 3,
    overlap: int = 32,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cuda'),
    amp: bool = True,
) -> np.ndarray:
    if len(img.shape) != 3 or img.shape[2] != channels:
        if channels == 3:
            img = np.stack((img,) * 3, axis=-1)
        elif channels == 1:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = np.mean(img, axis=2, keepdims=True)
            elif len(img.shape) == 2:
                img = img[..., np.newaxis]

    h, w, c = get_h_w_c(img)
    tile_size = tiler.starting_tile_size(w, h, c)
    model = model.to(device, dtype=dtype).eval()

    while True:
        try:
            result_blender = TileBlender(h * scale, w * scale, c, BlendDirection.Y)

            for y_seg in split_into_segments(h, tile_size[1], overlap):
                y_start, y_end = y_seg.start - y_seg.start_padding, y_seg.end + y_seg.end_padding
                row_blender = TileBlender((y_end - y_start) * scale, w * scale, c, BlendDirection.X)

                for x_seg in split_into_segments(w, tile_size[0], overlap):
                    x_start, x_end = x_seg.start - x_seg.start_padding, x_seg.end + x_seg.end_padding

                    with torch.autocast(device_type=str(device), dtype=dtype, enabled=amp):
                        with torch.inference_mode():
                            tensor = model(image2tensor(img[y_start:y_end, x_start:x_end, :], dtype=dtype).to(device))

                    row_blender.add_tile(tensor2image(tensor), TileOverlap(start=x_seg.start_padding * scale, end=x_seg.end_padding * scale))
                    del tensor

                result_blender.add_tile(row_blender.get_result(), TileOverlap(start=y_seg.start_padding * scale, end=y_seg.end_padding * scale))
                del row_blender

            return result_blender.get_result()

        except torch.cuda.OutOfMemoryError:
            tile_size = tiler.split(tile_size)
            torch.cuda.empty_cache()

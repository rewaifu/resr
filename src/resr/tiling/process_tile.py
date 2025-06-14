import gc
from dataclasses import dataclass

import numpy as np
import torch

from resr.img_util import get_h_w_c, image2tensor, tensor2image

from .tile_blender import BlendDirection, TileBlender, TileOverlap
from .tiler import Tiler


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
    device: torch.device | None = None,
    amp: bool = True,
) -> np.ndarray:
    if device is None:
        device = torch.device('cuda')

    if img.ndim == 2:
        img = img[..., np.newaxis]
    if img.shape[2] != channels:
        if channels == 3:
            img = np.repeat(img, 3, axis=2)
        elif channels == 1:
            img = np.mean(img, axis=2, keepdims=True)

    h, w, c = get_h_w_c(img)
    tile_size = tiler.starting_tile_size(w, h, c)
    model = model.to(device, dtype=dtype).eval()

    autocast_ctx = torch.autocast(device_type=str(device), dtype=dtype, enabled=amp)

    while True:
        try:
            result_blender = TileBlender(h * scale, w * scale, c, BlendDirection.Y)

            for y_seg in split_into_segments(h, tile_size[1], overlap):
                y_start, y_end = y_seg.start - y_seg.start_padding, y_seg.end + y_seg.end_padding
                row_blender = TileBlender((y_end - y_start) * scale, w * scale, c, BlendDirection.X)

                for x_seg in split_into_segments(w, tile_size[0], overlap):
                    x_start, x_end = x_seg.start - x_seg.start_padding, x_seg.end + x_seg.end_padding
                    img_tile = img[y_start:y_end, x_start:x_end, :]

                    with autocast_ctx, torch.inference_mode():
                        tensor = image2tensor(img_tile, dtype=dtype).to(device).unsqueeze(0)
                        tensor = model(tensor)

                    img_tile = tensor2image(tensor)
                    row_blender.add_tile(img_tile, TileOverlap(start=x_seg.start_padding * scale, end=x_seg.end_padding * scale))

                    del img_tile, tensor

                img_row = row_blender.get_result()
                result_blender.add_tile(img_row, TileOverlap(start=y_seg.start_padding * scale, end=y_seg.end_padding * scale))

                del img_row, row_blender
                gc.collect()

            return result_blender.get_result()

        except torch.cuda.OutOfMemoryError:  # noqa: PERF203
            tile_size = tiler.split(tile_size)
            torch.cuda.empty_cache()
            gc.collect()

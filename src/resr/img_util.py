import cv2
import numpy as np
import torch
from torch import Tensor


def get_h_w_c(img: np.ndarray):
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    return h, w, c


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def median_blur_and_normalize(image, kernel_size=13):
    blurred_image = cv2.medianBlur((image * 255).astype(np.uint8), kernel_size).astype(np.float32) / 255.0
    min_val = np.min(blurred_image)
    return (image.astype(np.float32) - min_val) / (np.max(blurred_image) - min_val)


def image2tensor(value: list[np.ndarray] | np.ndarray, dtype: torch.dtype = torch.float32) -> list[Tensor] | Tensor:
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(img[None, ...]) if len(img.shape) == 2 else torch.from_numpy(img).permute(2, 0, 1)

        if tensor.dtype != dtype:
            tensor = tensor.to(dtype, non_blocking=True)

            if img.dtype == np.uint8 and dtype.is_floating_point:
                tensor.div_(255)

        return tensor

    if isinstance(value, list):
        return [_to_tensor(i) for i in value]
    return _to_tensor(value)


def tensor2image(value: list[torch.Tensor] | torch.Tensor, dtype=np.float32) -> list[np.ndarray] | np.ndarray:
    def _to_ndarray(tensor: torch.Tensor) -> np.ndarray:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        tensor = tensor.detach().cpu()

        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        img = tensor.numpy() if len(tensor.shape) == 2 else tensor.permute(1, 2, 0).numpy()

        if tensor.dtype.is_floating_point and dtype == np.uint8:
            img = (img * 255.0).round()

        return img.astype(dtype, copy=False)

    if isinstance(value, list):
        return [_to_ndarray(i) for i in value]
    return _to_ndarray(value)

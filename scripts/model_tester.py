import logging
import os

import typer
from pepeline import ImgFormat, read, save
from resselt import load_from_file
from tqdm import tqdm

from resr.file_io import scandir
from resr.img_util import median_blur_and_normalize
from resr.tiling import ExactTileSize, process_tiles

logger = logging.getLogger(__name__)


def main(
    input_folder: str,
    output_folder: str,
    model_folder: str,
    model_names: list[str],
    tile_size: int = typer.Option(default=512),
    normalize: bool = typer.Option(default=False),
) -> None:
    tiler = ExactTileSize(tile_size)

    os.makedirs(output_folder, exist_ok=True)

    model_paths = list(scandir(model_folder, recursive=True, suffix='.pth'))
    image_paths = list(scandir(input_folder))

    total = len(model_names) * len(image_paths)
    with tqdm(total=total) as pbar:
        for model_name in model_names:
            # Определяем путь к модели
            if os.path.isabs(model_name):
                model_path_list = [model_name]
                model_name_str = os.path.basename(model_name)
            else:
                model_path_list = [
                    p for p in model_paths if model_name in os.path.basename(p)
                ]
                model_name_str = model_name

            if not model_path_list:
                logger.warning(
                    "Invalid model name: %s. File doesn't exist in %s",
                    model_name_str,
                    model_folder,
                )
                continue

            base_model_name = os.path.splitext(model_name_str)[0]
            model = load_from_file(model_path_list[0])

            for img_path in image_paths:
                img_name = os.path.basename(img_path)
                pbar.set_description(
                    f'Model: {model_name_str} | Image: {img_name}'
                )
                img = read(img_path, img_format=ImgFormat.F32)
                if normalize:
                    img = median_blur_and_normalize(img)
                img = process_tiles(
                    img,
                    tiler=tiler,
                    model=model,
                    scale=model.parameters_info.upscale,
                )
                basename = os.path.splitext(img_name)[0]
                output_path = os.path.join(
                    output_folder,
                    f'{basename}_{base_model_name}.png',
                )
                save(img, output_path)
                pbar.update(1)


if __name__ == '__main__':
    typer.run(main)

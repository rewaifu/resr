import logging
import os

import typer
from pepeline import read, ImgFormat, ImgColor, save
from resselt import load_from_file
from tqdm import tqdm

from pepetorch.tiling import process_tiles, ExactTileSize
from pepetorch.utils.misc import scandir


def main(input_folder: str, output_folder: str, model_folder: str, model_names: list[str], tile_size: int = typer.Option(default=512)):
    tiler = ExactTileSize(tile_size)

    os.makedirs(output_folder, exist_ok=True)
    model_paths = list(scandir(model_folder, recursive=True, suffix='.pth'))
    image_paths = list(scandir(input_folder))

    with tqdm(total=len(model_names) * len(image_paths)) as pbar:
        for model_name in model_names:
            model_path = list(filter(lambda x: model_name in x, model_paths))
            if len(model_path) == 0:
                logging.warn(f"Invalid model name: {model_name}. File doesn't exists in {model_folder}")
            base_model_name, _ = os.path.splitext(model_name)

            model = load_from_file(model_path[0])

            for img_path in image_paths:
                pbar.set_description(f'Model: {model_name} | Image: {os.path.basename(img_path)}')
                img = read(img_path, format=ImgFormat.F32, mode=ImgColor.GRAY)
                img = process_tiles(img, tiler=tiler, model=model, scale=model.parameters_info.upscale)
                basename, _ = os.path.splitext(os.path.basename(img_path))
                save(img, os.path.join(output_folder, f'{basename}_{base_model_name}.png'))
                pbar.update(1)


if __name__ == '__main__':
    typer.run(main)

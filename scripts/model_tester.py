import logging
from pathlib import Path

import typer
from pepeline import ImgFormat, read, save
from resselt import load_from_file
from tqdm import tqdm

from resr.file_io import scandir
from resr.tiling import ExactTileSize, process_tiles

logger = logging.getLogger(__name__)


def main(
    input_folder: Path,
    output_folder: Path,
    model_folder: str,
    model_names: list[Path],
    tile_size: int = typer.Option(default=512),
) -> None:
    tiler = ExactTileSize(tile_size)

    output_folder.mkdir(exist_ok=True)

    model_paths = list(scandir(model_folder, recursive=True, suffix=".pth"))
    image_paths = list(scandir(input_folder))

    with tqdm(total=len(model_names) * len(image_paths)) as pbar:
        for model_name_path in model_names:
            if model_name_path.is_absolute():
                model_path_list = [model_name_path]
                model_name_str = model_name_path.name
            else:
                model_path_list = list(filter(lambda x: model_name_path.name in x.name, model_paths))
                model_name_str = model_name_path.name

            if not model_path_list:
                logger.warning("Invalid model name: %s. File doesn't exist in %s", model_name_str, model_folder)
                continue

            base_model_name = Path(model_name_str).stem
            model = load_from_file(model_path_list[0])

            for img_path in image_paths:
                pbar.set_description(f"Model: {model_name_str} | Image: {img_path.name}")
                img = read(img_path, format=ImgFormat.F32)
                img = process_tiles(
                    img,
                    tiler=tiler,
                    model=model,
                    scale=model.parameters_info.upscale,
                )
                basename = img_path.stem
                output_path = output_folder / f"{basename}_{base_model_name}.png"
                save(img, output_path)
                pbar.update(1)


if __name__ == "__main__":
    typer.run(main)

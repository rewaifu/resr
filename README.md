## Usage

```python
from resselt import load_from_file
from resr.tiling import process_tiles, ExactTileSize
from pepeline import read, save

model = load_from_file('4x_spanplus.pth')

tiler = ExactTileSize(512)

img = read('image.png')
img = process_tiles(img, tiler=tiler, model=model, scale=model.parameters_info.upscale)
save(img, 'output.png')
```
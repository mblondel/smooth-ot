To compute and plot color transfer, call the script `plot_color_transfer.py`. Pass in `--img1` and `--img2` as valid images in the `data/` directory. For instance, we can call the following:
```
python plot_color_transfer.py --n_colors 256 --method l2_sd --gamma 1 --img1 comunion --img2 autumn
```
Full usage is as follows:
```
usage: plot_color_transfer.py [-h] [--n_colors N_COLORS] [--method METHOD]
                              [--gamma GAMMA] [--max_iter MAX_ITER]
                              [--img1 IMG1] [--img2 IMG2]

optional arguments:
  -h, --help           show this help message and exit
  --n_colors N_COLORS  number of color clusters
  --method METHOD      OT method
  --gamma GAMMA        regularization parameter
  --max_iter MAX_ITER
  --img1 IMG1
  --img2 IMG2
```

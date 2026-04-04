Below are instructions for creating the SMPL-based dataset used in our reposing and human segmentation experiments. First, we walk through how to setup SMPL-X and the MOYO pose dataset. Then we will use `/smplx_data/bake_moyo_data.py` to create the dataset as it appears in our paper.

## Setup

### SMPL-X Body Model:
First ensure the [smplx](https://github.com/vchoutas/smplx) Python package is installed:

```bash
pip install smplx[all]
```

Then download the SMPL-X human body models from:

https://smpl-x.is.tue.mpg.de/download.php


Click the button that says: "Download SMPLX-X v1.1 (NPZ+PKL, 830 MB) - Use this for SMPL-X Python codebase"
Then unzip and populate the PoissonNet directory`smplx_data/models/` with the `.pkl` and `.npz` files. Your directory should look like:

```
poissonnet
├── smplx_data
│   ├── models
│   │   ├── SMPLX_FEMALE.npz
│   │   ├── SMPLX_FEMALE.pkl
│   │   ├── SMPLX_MALE.npz
│   │   ├── SMPLX_MALE.pkl
│   │   ├── SMPLX_NEUTRAL.npz
│   │   └── SMPLX_NEUTRAL.pkl
```

### MOYO Dataset:

First clone the `moyo` repository into our `smplx_data` directory:

```bash
cd smplx_data
git clone https://github.com/sha2nkt/moyo.git
```

Follow instructions to download the MOYO dataset, their repo suggests:

```bash
cd moyo_toolkit
bash ./moyo/bash/download_moyo.sh -o ./data/ -u
```

Now you should have:
```
poissonnet/
├── smplx_data/
│   ├── models/
│   ├── moyo_toolkit/
│   └── ...
```

## Creating human mesh dataset

To create the mesh dataset exactly as it appears in our paper, run the following from the `poissonnet` root directory. If your MOYO or SMPL-X directories are different, you can change the CLI arguments below:
```bash
python -m smplx_data.bake_moyo_data --moyo_dir ./smplx_data/moyo/data --smplx_dir ./smplx_data/models --train_samples 32000 --test_samples 2000 --num_dupes 1 --body_shape_std 5.0
```
For more context, please refer to the `bake_moyo_data.py` script.

Now you should be left with two PyTorch `.pt` files inside of `smplx_data/`:
```
poissonnet/
├── smplx_data/
│   ├── models/
│   ├── moyo_toolkit/
│   ├── train_baked_hands_32000_5.0.pt
│   ├── test_baked_hands_2000_5.0.pt
│   └── ...
```

The dictionaries are sturctured as:
```python
{
    'src_verts': (N, V, 3)  -- source vertices
    'src_poses': (N, d)     -- source poses parameters
    'src_betas': (N, 10)    -- source body shape parameters
    'tar_verts': (N, V, 3)  -- target vertices
    'tar_poses': (N, d)     -- target poses parameters
    'tar_betas': (N, 10)    -- target body shape parameters
    'faces': (F, 3)         -- SMPLX face indices (shared across all shapes)
    'genders': (N,)         -- gender of each shape [0, 1, 2] -> [neutral, male, female]
    'body_shape_std': float -- standard deviation used for sampling body shape parameters
}
```

Now you should be ready to run both the segmentation and reposing experiments!
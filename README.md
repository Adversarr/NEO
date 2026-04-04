<h1 align="center">Learning Laplacian Eigenspace with<br>Mass-Aware Neural Operators on Point Clouds</h1>

<p align="center">
  <a href="https://github.com/Adversarr">Zherui Yang</a><sup>1</sup> &nbsp;&nbsp;
  <a href="https://people.iiis.tsinghua.edu.cn/~taodu">Tao Du</a><sup>2,3</sup> &nbsp;&nbsp;
  <a href="http://staff.ustc.edu.cn/~lgliu/">Ligang Liu</a><sup>1&dagger;</sup>
</p>
<p align="center">
  <sup>1</sup>University of Science and Technology of China &nbsp;&nbsp;
  <sup>2</sup>Tsinghua University &nbsp;&nbsp;
  <sup>3</sup>Shanghai Qi Zhi Institute
</p>
<p align="center">
  <b>SIGGRAPH 2026 (Conditional Accept)</b>
</p>
<p align="center">
  <!-- <a href="#"><img src="https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b?logo=arxiv" alt="arXiv"></a> -->
  <a href="https://adversarr.github.io/NEO"><img src="https://img.shields.io/badge/Project-Page-blue?logo=googlechrome" alt="Project Page"></a>
  <a href="https://github.com/Adversarr/NEO"><img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="GitHub"></a>
</p>

<p align="center">
  <img src="figures/teaser.jpg" width="100%" alt="NEO teaser"/>
</p>

**NEO (Neural Eigenspace Operator)** is a feed-forward framework that predicts the low-frequency Laplace-Beltrami eigenspace directly from point clouds, replacing expensive iterative eigensolvers with rapid neural inference and Rayleigh-Ritz refinement. It achieves **88x speedup** over ARPACK at 512k points with **near-linear** runtime scaling and **zero-shot** generalization from 2k training to 512k+ inference.



## Highlights

- **Subspace Learning:** Reformulates eigenvector regression as invariant subspace prediction, resolving sign-flip and rotation ambiguities.
- **Mass-Aware Attention:** Injects per-point area weights into cross-attention, enabling robust handling of non-uniform sampling densities.
- **Near-Linear Scaling:** O(N) inference for fixed target modes, compared to O(N^1.16) for ARPACK.
- **Zero-Shot Resolution Transfer:** Trained on 2k-point clouds, generalizes to 512k+ points without fine-tuning.
- **Dual Utility:** Serves as both a fast eigensolver replacement and an effective intrinsic point embedding.



## Method Overview

<p align="center">
  <img src="figures/pipeline.jpg" width="100%" alt="NEO pipeline"/>
</p>

NEO takes a point cloud with per-point mass weights as input. A mass-aware neural operator predicts redundant basis functions in a single forward pass. These are M-orthonormalized, then the discrete Laplacian is projected into the low-dimensional subspace. A small dense eigenproblem yields the final LBO eigenfunctions via Rayleigh-Ritz refinement.



## Results

### Runtime Scaling

<p align="center">
  <img src="figures/runtime.jpg" width="60%" alt="Runtime comparison"/>
</p>


### Accuracy

<p align="center">
  <img src="figures/accuracy.jpg" width="60%" alt="Accuracy"/>
</p>


On the ShapeNet test set (k=96): mean span loss of 3.35e-3 (>99.7% spectral energy capture), stable under FP16 mixed precision.

### Robustness to Non-Uniform Sampling

<p align="center">
  <img src="figures/non-uniform-robustness.jpg" width="100%" alt="Non-uniform robustness"/>
</p>

Mass-aware attention prevents catastrophic degradation under biased sampling, while the mass-agnostic baseline fails.

### Cross-Resolution & Discretization Transfer

<p align="center">
  <img src="figures/cross-resolution.jpg" width="100%" alt="Cross-resolution transfer"/>
</p>

Strong zero-shot transfer from 2k training to 1.6M points across different Laplacian discretizations (intrinsic Delaunay mesh vs. k-NN graph).

### Gallery

<p align="center">
  <img src="figures/gallery-teaser.jpg" width="100%" alt="Gallery"/>
</p>

NEO predictions on diverse out-of-distribution shapes spanning organic creatures, classic graphics models, and man-made CAD parts.

## Applications

### Shape Matching via Functional Maps

<p align="center">
  <img src="figures/fmap.jpg" width="60%" alt="Functional maps"/>
</p>

### Segmentation (NEO + PointNet)

<p align="center">
  <img src="figures/segmentation-compare.jpg" width="60%" alt="Segmentation comparison"/>
</p>


### Heat-Based Geodesic Distance

<p align="center">
  <img src="figures/heat-geodesic.jpg" width="60%" alt="Heat geodesic"/>
</p>



## Installation

### Automatic

```bash
./setup_conda.sh
```

### Manual

```bash
conda env create --name neo "python=3.12"
conda activate neo
pip install -r requirements.txt -r requirements-pyg.txt
pip install -e .
```

## Reproducing Experiments

All paper results can be reproduced by following the steps below. Scripts live under `exp/launch/` and assume the `neo` conda environment is active and `G2PT_DATA_ROOT` (or the relevant per-dataset variable) points to your data directory.

### 🌙 0. Bonus 
Beyond the core paper pipeline, this repo also contains a few side utilities and exploratory artifacts that were useful during development but are not part of the main reproduction path.

- **Blender rendering utilities:** `renders/` includes reusable Blender scripts and command notes for reproducing teaser/gallery-style point-cloud and mesh visualizations from generated outputs.
- **Additional development baselines:** besides the main NEO checkpoints, the codebase still includes PointNet2, PointTransformer, position-only, RoPE, Transolver-variants used in internal comparisons.
- **Extra exploratory results:** the repo also contains omitted figures and analysis scripts for failure cases, subspace-size ablations, downstream summary plots, and alternative eigensolver backends beyond the README's headline results.

### 1. Data Preparation

**Pre-training data (ShapeNet).**
Download ShapeNet and run the preprocessing script to compute per-sample LBO eigenvectors and pack everything into HDF5 files:

```bash
python exp/pretrain_datagen/shapenet/preprocess.py \
    --input-dir  /path/to/ShapeNet \
    --output-dir /path/to/preprocessed_shapenet_h5 \
    --k 96 --n-samples 1024 --per-mesh-count 4 --num-workers 8
```

Set the output path as `G2PT_DATA_SHAPENET` (or export `G2PT_DATA_ROOT` and the script will derive it automatically).

**SHREC16 (classification).**  Download the SHREC 2016 dataset and preprocess it:

```bash
python exp/downstream/preprocess_shrec16_classification.py \
    --root-dir /path/to/shrec_16 \
    --output-dir /path/to/preprocessed_shrec16
```

**Correspondence (functional maps).**  Download the correspondence dataset and run:

```bash
python exp/downstream/preprocess_corr.py \
    --root-dir /path/to/functional_mapping \
    --output-dir /path/to/preprocessed_corr \
    --k 128
```

**Human segmentation.**  Download the Human Body Segmentation Benchmark (SIG17) and preprocess it:

```bash
python exp/downstream/preprocess_human_sig17_seg_benchmark.py \
    --root-dir /path/to/human_seg \
    --output-dir /path/to/preprocessed_human_seg
```

---

### 2. Pre-training

Train the base NEO model on ShapeNet:

```bash
export G2PT_DATA_SHAPENET=/path/to/preprocessed_shapenet_h5
bash exp/launch/pretrain-base.sh
```

### 3. Downstream Tasks

Each downstream script loads a frozen pre-trained backbone (`freeze_pretrained=true`) and fine-tunes only the task head. Run them after pre-training is complete.

**3a. Shape Classification (SHREC16)**

```bash
bash exp/launch/downstream_classification_shrec16.sh
```

This sweeps over training-set sizes `{30, 90, 120, 300, 480}` and runs three conditions in parallel for each: NEO (with pre-training), PointNet baseline, and PointTransformer baseline. Results are logged per `run_name`.

**3b. Shape Correspondence (Functional Maps)**

```bash
bash exp/launch/downstream_correspondence.sh
```

Runs two conditions: NEO with the pre-trained backbone (`freeze_pretrained=true`, `model_depth=6`) and a position-embedding-only baseline. Both use `batch_size=32`.

**3c. Human Body Segmentation**

```bash
bash exp/launch/downstream_segment_human.sh
```

Runs `exp/downstream/seg.py` with and without the pre-trained backbone. The `run_name` distinguishes results (`with-pretrain` vs. `no-pretrain`).

---

### 4. Inference & Evaluation

The `exp/pretrain/infer_*.py` family of scripts evaluates a trained model against ground-truth LBO eigenspaces and records timing, span loss, and cosine-similarity scores for each sample. Use `infer.py` for point-cloud inputs (with robust Laplacian) and `infer_mesh.py` for triangle-mesh inputs:

```bash
# Point-cloud inference (robust Laplacian)
python exp/pretrain/infer.py \
    --ckpt  /path/to/model.ckpt \
    --data_dir /path/to/test_samples \
    --glob  "*" \
    --device cuda

# Mesh inference
python exp/pretrain/infer_mesh.py \
    --ckpt  /path/to/model.ckpt \
    --data_dir /path/to/test_samples \
    --device cuda

# k-NN graph Laplacian (discretization transfer experiment)
python exp/pretrain/infer_knn.py \
    --ckpt  /path/to/model.ckpt \
    --data_dir /path/to/test_samples \
    --device cuda
```

Each script writes per-sample results (eigenvectors, timing, scores) under `<sample_dir>/inferred/` and a `results.json` summary. Pass `--no-mass` to ablate the mass-aware attention at inference time. For mixed-precision evaluation, FP16 results are produced automatically alongside FP32 when a CUDA device is available.

Runtime statistics (Table 1 in the paper) can be collected and plotted with:

```bash
python exp/pretrain/get_stats_from_validation.py
python exp/pretrain/plot_stats.py
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{yang2026neo,
  title={Learning Laplacian Eigenspace with Mass-Aware Neural Operators on Point Clouds},
  author={Yang, Zherui and Du, Tao and Liu, Ligang},
  journal={ACM Transactions on Graphics (Proc. SIGGRAPH)},
  year={2026}
}
```

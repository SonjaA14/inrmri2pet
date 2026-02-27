# MRI → SUV Prediction via Voxelwise SIREN Network

*Code for MICCAI paper submission 2026*

A per-voxel neural network that predicts **PET SUV (Standardized Uptake Value)** from a set of co-registered MRI sequences. Each voxel is treated as an independent sample; the network learns the mapping from MRI intensities to metabolic activity without any spatial convolution.

---

## Goal

Given a set of MRI modalities and a co-registered PET/CT scan for the same patient, train a network that predicts the SUV value at every soft-tissue voxel purely from the local MRI signal. This allows investigation of which MRI-derived features carry information about PSMA uptake.

---

## Input Data

All images are expected as NIfTI (`.nii.gz`) files. The dense displacement field (DDF) used for registration is an `.mha`.

| Key | Category | Description |
|-----|----------|-------------|
| `t1` | Structural | T1-weighted MRI |
| `t2` | Structural | T2-weighted MRI |
| `adc` | Diffusion | Apparent Diffusion Coefficient map |
| `ttp` | Perfusion | Time-to-peak from dynamic contrast-enhanced MRI |
| `inr` | Perfusion | INR components (4D NIfTI, 3 components: Ktrans, ve, vp) |
| `ct` | Reference | CT image defining the reference voxel grid |
| `suv` | Target | PET SUV image (prediction target, already in CT space) |
| `ddf` | Registration | Dense displacement field (MRI → CT registration) |

Paths per patient are configured in `paths.json`.

---

## Preprocessing (`dataset.py` — `preprocess_and_save`)

All preprocessing is run once per patient and outputs are cached to disk (controlled by the `recompute_data` flag in `main.py`).

### 1. Load reference grid
The CT image defines the target voxel grid (spacing, origin, orientation). The SUV image is resampled onto the CT grid using linear interpolation.

### 2. Warp MRI images to CT space
Each 3D MRI modality (T1, T2, ADC, TTP) is warped into CT space by applying the provided dense displacement field (DDF) via `SimpleITK.DisplacementFieldTransform` + linear resampling.

The INR feature map is 4D (x, y, z, 3 components). Each component is warped independently and the results are stacked back into a 4D volume.

### 3. Crop to MRI overlap region
After warping, a non-zero intersection mask is computed across all modalities. The bounding box of this mask is used to crop all images (CT, SUV, and all warped MRI volumes) to the region where all inputs have valid data. The affine origin is updated accordingly to keep physical coordinates correct.

### 4. Save processed images
All cropped images are saved as `*_processed.nii.gz` in the patient output directory for reuse across ablation runs.

---

## Dataset (`dataset.py` — `VoxelwiseMRIDataset`)

### HU gating
A CT-based soft-tissue mask is applied: only voxels with Hounsfield Units in **[-300, 300]** are kept. This excludes air (very negative HU) and bone (high positive HU), retaining only soft tissue. The CT mask also drives the per-slice evaluation plot (see below).

### Normalization
NaN / Inf values in TTP and INR are replaced with 0 before normalization. Each feature is then normalized as follows, with statistics computed over the valid (soft-tissue) voxels only:

| Feature | Method |
|---------|--------|
| T1, T2, ADC | Per-patient min–max → **[0, 1]** |
| TTP | Divided by fixed cohort-wide maximum of **240 s** → [0, ~1] |
| Ktrans, v_e, v_p | Per-component **z-score**, then divided by **20** to bring the range in line with [0, 1] features |
| SUV (target) | Per-patient min–max → **[0, 1]** |

The fixed TTP divisor (240 s) corresponds to the longest dynamic acquisition in the cohort, ensuring consistent scaling across patients. The INR z-score divisor of 20 is a fixed rescaling factor applied after standardization to keep all input features on a comparable scale.

### Samples
Each dataset item is a single voxel: a 1D feature vector and a scalar SUV target. With a batch size of **4096 voxels** and shuffling, the full soft-tissue volume is seen each epoch.

### Configurable feature set
The active feature set can be any subset of `{t1, t2, adc, ttp, Ktrans, ve, vp}`, enabling the ablation study in `main.py`.

---

## Model Architecture (`model.py` — `T2ADCEnvelope`)

### Fourier feature lifting
Before entering the network, the raw input features (N scalar values per voxel) are lifted to a high-dimensional Fourier representation:

1. A fixed random Gaussian matrix **B** of shape `(128, N)` with scale 10 is sampled once at initialisation and kept constant throughout training.
2. For each voxel `x` of shape `(N,)`, compute `2π · x · B^T` → shape `(128, N)`.
3. Apply `sin` and `cos` → `(256, N)`.
4. Flatten → `(256 · N,)` input to the network.

This gives the network access to high-frequency components of the input signal (random Fourier features / positional encoding style).

### SIREN network
The lifted features pass through a **SIREN** (Sinusoidal Representation Network):
- **1 first layer**: `SineLayer` with `ω₀ = 30`, weights initialized uniformly in `[-1/in, 1/in]`.
- **3 hidden layers**: `SineLayer` with `ω₀ = 30`, weights initialized to preserve signal variance: `U[-√(6/in)/ω₀, √(6/in)/ω₀]`.
- **Output layer**: linear, 512 → 1, same variance-preserving initialization.

Each `SineLayer` computes `sin(ω₀ · W x + b)`.

Configuration: `in_features = 256 · N`, `hidden_features = 512`, `hidden_layers = 3`, `out_features = 1`.

### Loss: OrthoPar decomposition
The standard MSE loss is augmented with a structured error term. Within each batch the error `e = ŷ - y_norm` is decomposed using SVD into:

- **Parallel component** `r_‖`: the projection of the error onto the column space of the (batch-normalised) feature matrix `X_n`. Errors here are "explainable" by the input features — the model is making a mistake it could in principle fix.
- **Orthogonal component** `r_⊥`: the residual. Errors here are not captured by the linear span of the features.

Total loss:
```
L = MSE(e) + λ_parallel · MSE(r_‖)
```
with `λ_parallel = 1.0` and a ridge regularization parameter `ridge = 1e-3`.

This encourages the network to minimise errors that lie in the direction the features can explain, putting more pressure on reducing the "avoidable" error.

Within-batch z-score normalization is applied to features and target before loss computation (features clamped to ±10).

### Optimizer
Adam with AMSGrad, `lr = 1e-5`. Learning rate is halved (`factor = 0.5`) if `train/loss` does not improve for 5 epochs (`ReduceLROnPlateau`, `patience = 5`).

---

## Training Setup (`train.py`)

- **Framework**: PyTorch Lightning
- **Batch size**: 4096 voxels
- **Epochs**: 75 (configurable)
- **Accelerator**: MPS (Apple Silicon) → CUDA → CPU, auto-detected
- **DataLoader**: 4 workers, shuffle=True, persistent workers
- **Logging**: Weights & Biases (`wandb`), project `inrmri`, run named `{patient_id}_{feature_set_name}`

Logged W&B config fields include learning rate, epochs, patient ID, feature set name, and boolean flags for each active modality.

---

## Evaluation Callback (`model.py` — `EvaluationPlotCallback`)

At epoch 1 and every 5 epochs thereafter, the callback:

1. Selects the **middle axial slice** of the volume as the evaluation slice.
2. Runs inference on all valid soft-tissue voxels in that slice (no gradient).
3. Computes the OrthoPar error decomposition for those voxels.
4. Reconstructs five 2D images at full slice resolution (invalid voxels → NaN):

| Panel | Content |
|-------|---------|
| Original SUV | Normalized ground-truth SUV for the eval slice |
| Predicted SUV | Model prediction for the same slice |
| Difference | `|prediction − target|` with MAE annotation |
| Parallel Error | Squared parallel error component `r_‖²` with MSE annotation |
| Orthogonal Error | Squared orthogonal error component `r_⊥²` with MSE annotation |

All panels use `origin='lower'` orientation. The figure (1×5, 25×5 inches) is logged to W&B as an image alongside scalar metrics `eval_slice/mae`, `eval_slice/mse`, and `eval_slice/rmse`.

---

## Ablation Study (`main.py`)

The entry point runs a systematic leave-one-out and group-ablation study. For each patient the preprocessing is run once (`recompute_data=True`) and then reused for all feature subsets.

| Ablation name | Features used |
|---------------|---------------|
| `full` | t1, t2, adc, ttp, Ktrans, ve, vp |
| `minus_t1` | t2, adc, ttp, Ktrans, ve, vp |
| `minus_t2` | t1, adc, ttp, Ktrans, ve, vp |
| `minus_adc` | t1, t2, ttp, Ktrans, ve, vp |
| `minus_ttp` | t1, t2, adc, Ktrans, ve, vp |
| `minus_Ktrans` | t1, t2, adc, ttp, ve, vp |
| `minus_ve` | t1, t2, adc, ttp, Ktrans, vp |
| `minus_vp` | t1, t2, adc, ttp, Ktrans, ve |
| `minus_structural` | adc, ttp, Ktrans, ve, vp (no T1/T2) |
| `minus_inr_series` | t1, t2, adc, ttp (no INR components) |

Final metrics per run are appended to a timestamped CSV at `outputs/ablation_results_<timestamp>.csv`.

---

## Running

```bash
python main.py
```

Paths to patient data are read from `paths.json`. Only patients with all required files present (`.nii.gz` or `.mha`) are processed; others are skipped with a warning.

Output files per patient are written to `outputs/<patient_id>/`.

import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

# Cohort-wise TTP normalization constant: maximum acquisition time across the cohort
TTP_MAX_S = 240


def compute_physical_bbox(nii_image):
    """
    Compute the axis-aligned bounding box of a NIfTI image in world coordinates.

    Args:
        nii_image: A nibabel Nifti1Image

    Returns:
        bbox_min: np.ndarray of shape (3,) -- minimum world coordinates
        bbox_max: np.ndarray of shape (3,) -- maximum world coordinates
    """
    shape = nii_image.shape[:3]
    affine = nii_image.affine

    # All 8 corners in voxel space
    corners_voxel = np.array([
        [i, j, k]
        for i in [0, shape[0] - 1]
        for j in [0, shape[1] - 1]
        for k in [0, shape[2] - 1]
    ])  # (8, 3)

    # Transform to world coordinates
    corners_homo = np.hstack([corners_voxel, np.ones((8, 1))])  # (8, 4)
    corners_world = (affine @ corners_homo.T).T[:, :3]  # (8, 3)

    return corners_world.min(axis=0), corners_world.max(axis=0)


def compute_overlap_bbox(images):
    """
    Compute the intersection of axis-aligned bounding boxes for multiple NIfTI images.

    Args:
        images: List of nibabel Nifti1Image objects

    Returns:
        overlap_min: np.ndarray of shape (3,)
        overlap_max: np.ndarray of shape (3,)

    Raises:
        ValueError: If images do not overlap in all three dimensions
    """
    all_mins = []
    all_maxs = []

    for img in images:
        bbox_min, bbox_max = compute_physical_bbox(img)
        all_mins.append(bbox_min)
        all_maxs.append(bbox_max)

    overlap_min = np.max(all_mins, axis=0)
    overlap_max = np.min(all_maxs, axis=0)

    if np.any(overlap_min >= overlap_max):
        raise ValueError(
            f"Images do not overlap in all dimensions. "
            f"Overlap min: {overlap_min}, Overlap max: {overlap_max}"
        )

    return overlap_min, overlap_max


def load_nifti_as_sitk(path):
    """Load NIfTI via nibabel into SimpleITK, matching warp_mri_sequence.py convention."""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    img = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    aff = nii.affine
    sp = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))  # column norms = true voxel spacing
    img.SetSpacing(sp.tolist())
    img.SetOrigin(aff[:3, 3].tolist())
    img.SetDirection((aff[:3, :3] / sp).flatten().tolist())
    return img


def _resample_sitk_to_ct(moving_sitk, ct_sitk, ddf_transform):
    """Resample a SimpleITK image to CT space; return numpy array (x,y,z) and CT affine."""
    warped = sitk.Resample(
        moving_sitk, ct_sitk, ddf_transform,
        sitk.sitkLinear, 0.0, moving_sitk.GetPixelID(),
    )
    arr = sitk.GetArrayFromImage(warped).transpose(2, 1, 0)
    sp = np.array(ct_sitk.GetSpacing())
    origin = np.array(ct_sitk.GetOrigin())
    direction = np.array(ct_sitk.GetDirection()).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction * sp
    affine[:3, 3] = origin
    return arr, affine


def warp_mri_to_ct(mri_path, ct_sitk, ddf_transform):
    """
    Warp an MRI image to CT space using a displacement field.

    Args:
        mri_path: Path to MRI NIfTI file
        ct_sitk: SimpleITK image defining the CT reference grid
        ddf_transform: SimpleITK DisplacementFieldTransform

    Returns:
        warped_nii: nibabel Nifti1Image on the CT grid
    """
    moving = load_nifti_as_sitk(mri_path)
    arr, affine = _resample_sitk_to_ct(moving, ct_sitk, ddf_transform)
    return nib.Nifti1Image(arr, affine)


def warp_mri_4d_to_ct(mri_path, ct_sitk, ddf_transform):
    """
    Warp a 4D MRI image to CT space by warping each component (last axis) independently.

    Returns:
        warped_nii: nibabel Nifti1Image of shape (x, y, z, n_components) on the CT grid
    """
    nii = nib.load(mri_path)
    data = nii.get_fdata().astype(np.float32)
    aff = nii.affine
    sp = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))  # column norms = true voxel spacing

    n_components = data.shape[3] if data.ndim == 4 else 1
    warped_components = []
    for c in range(n_components):
        comp = data[..., c] if data.ndim == 4 else data
        sitk_img = sitk.GetImageFromArray(comp.transpose(2, 1, 0))
        sitk_img.SetSpacing(sp.tolist())
        sitk_img.SetOrigin(aff[:3, 3].tolist())
        sitk_img.SetDirection((aff[:3, :3] / sp).flatten().tolist())
        arr, affine = _resample_sitk_to_ct(sitk_img, ct_sitk, ddf_transform)
        warped_components.append(arr)

    stacked = np.stack(warped_components, axis=-1)  # (x, y, z, n_components)
    return nib.Nifti1Image(stacked, affine)


def resample_nii_to_grid(nii_image, ref_grid):
    """
    Resample a nibabel NIfTI image onto a SimpleITK reference grid
    using identity transform (physical-space alignment).

    Args:
        nii_image: nibabel Nifti1Image
        ref_grid: SimpleITK Image defining output geometry

    Returns:
        result_nii: nibabel Nifti1Image on the reference grid
    """
    data = nii_image.get_fdata().astype(np.float32)
    affine = nii_image.affine

    # Convert nibabel (x,y,z) -> SimpleITK (z,y,x)
    sitk_img = sitk.GetImageFromArray(data.transpose(2, 1, 0))

    # Extract spacing, origin, direction from the nibabel affine
    spacing_nib = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    origin_nib = affine[:3, 3]
    direction_nib = (affine[:3, :3] / spacing_nib).flatten()

    sitk_img.SetSpacing(spacing_nib.tolist())
    sitk_img.SetOrigin(origin_nib.tolist())
    sitk_img.SetDirection(direction_nib.tolist())

    # Resample onto the reference grid
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_grid)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())  # identity
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    resampled_sitk = resampler.Execute(sitk_img)

    # Convert back: SimpleITK (z,y,x) -> nibabel (x,y,z)
    resampled_np = sitk.GetArrayFromImage(resampled_sitk)
    resampled_np = np.transpose(resampled_np, (2, 1, 0))

    # Build output affine from reference grid metadata
    out_spacing = np.array(ref_grid.GetSpacing())
    out_origin = np.array(ref_grid.GetOrigin())
    out_direction = np.array(ref_grid.GetDirection()).reshape(3, 3)

    out_affine = np.eye(4)
    out_affine[:3, :3] = out_direction * out_spacing
    out_affine[:3, 3] = out_origin

    return nib.Nifti1Image(resampled_np, out_affine)


def resample_to_shape(image, target_shape):
    """
    Resample/resize image to target shape using interpolation.
    This preserves all information by scaling rather than cropping.

    Args:
        image: 3D numpy array
        target_shape: Target shape (x, y, z)

    Returns:
        Resampled image with target shape
    """
    from scipy.ndimage import zoom

    current_shape = np.array(image.shape)
    target_shape = np.array(target_shape)

    # Calculate zoom factors for each dimension
    zoom_factors = target_shape / current_shape

    # Resample using linear interpolation
    resampled = zoom(image, zoom_factors, order=1)  # order=1 is linear interpolation

    return resampled


def preprocess_and_save(t2_path, adc_path, t1_path, ct_path, suv_path,
                        ddf_path, ttp_path, inr_path, output_dir):
    """
    Warp MRI images to CT space using the provided DDF, crop to the MRI
    overlap region, and save all images on the same grid.

    The SUV and CT are already in the same space and only need cropping.

    Args:
        t2_path: Path to T2 NIfTI image
        adc_path: Path to ADC NIfTI image
        t1_path: Path to T1 NIfTI image
        ct_path: Path to CT NIfTI image (reference grid)
        suv_path: Path to SUV NIfTI image (prediction target, already in CT space)
        ddf_path: Path to dense displacement field (.mha or .nii.gz)
        ttp_path: Path to time-to-peak NIfTI image
        inr_path: Path to inr components NIfTI image
        output_dir: Directory for output files

    Returns:
        Dictionary with paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    # 1. Load CT reference, SUV, and DDF
    print("\n1. Loading CT reference, SUV, and displacement field...")
    ct_sitk = load_nifti_as_sitk(ct_path)
    ct_nii = nib.load(ct_path)
    print(f"   CT:  size={ct_sitk.GetSize()}, "
          f"spacing=({ct_sitk.GetSpacing()[0]:.3f}, {ct_sitk.GetSpacing()[1]:.3f}, {ct_sitk.GetSpacing()[2]:.3f}) mm")

    # Resample SUV to CT grid (same physical space but may differ in resolution)
    suv_sitk = load_nifti_as_sitk(suv_path)
    print(f"   SUV: size={suv_sitk.GetSize()}, "
          f"spacing=({suv_sitk.GetSpacing()[0]:.3f}, {suv_sitk.GetSpacing()[1]:.3f}, {suv_sitk.GetSpacing()[2]:.3f}) mm")
    suv_resampled = sitk.Resample(
        suv_sitk, ct_sitk, sitk.Transform(),
        sitk.sitkLinear, 0.0, suv_sitk.GetPixelID(),
    )
    suv_data = sitk.GetArrayFromImage(suv_resampled).transpose(2, 1, 0)
    suv_nii = nib.Nifti1Image(suv_data, ct_nii.affine)
    print(f"   SUV resampled to CT grid: {suv_nii.shape[:3]}")

    print(f"   Loading DDF: {ddf_path}")
    ddf = sitk.ReadImage(ddf_path, sitk.sitkVectorFloat64)
    ddf_transform = sitk.DisplacementFieldTransform(ddf)

    # 2. Warp each MRI image to CT space
    print("\n2. Warping MRI images to CT space...")
    mri_paths_3d = {"T2": t2_path, "ADC": adc_path, "T1": t1_path, "TTP": ttp_path}
    warped = {}
    for name, path in mri_paths_3d.items():
        print(f"   - Warping {name}...")
        warped[name] = warp_mri_to_ct(path, ct_sitk, ddf_transform)
        print(f"     {nib.load(path).shape[:3]} -> {warped[name].shape}")

    # INR is 4D: warp each component separately
    if inr_path is not None:
        print(f"   - Warping INR (4D, per-component)...")
        warped["INR"] = warp_mri_4d_to_ct(inr_path, ct_sitk, ddf_transform)
        n_inr = warped["INR"].shape[3]
        print(f"     -> {warped['INR'].shape[:3]} x {n_inr} components")

    # All warped images are now on the CT grid (same shape and affine)
    ct_shape = warped["T2"].shape
    print(f"\n   All images now on CT grid: {ct_shape}")

    # 3. Crop to the region where all MRI images have data (non-zero)
    print("\n3. Cropping to MRI overlap region...")
    nonzero_mask = np.ones(ct_shape, dtype=bool)
    for name in ["T2", "ADC", "T1", "TTP"]:
        nonzero_mask &= warped[name].get_fdata() != 0
    if "INR" in warped:
        # Require all INR components to be non-zero
        nonzero_mask &= (warped["INR"].get_fdata() != 0).all(axis=-1)

    # Find bounding box of the non-zero intersection
    coords = np.argwhere(nonzero_mask)
    if len(coords) == 0:
        raise ValueError("No overlapping non-zero voxels found across T1, T2, ADC")
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0) + 1  # +1 for exclusive upper bound

    slices = tuple(slice(lo, hi) for lo, hi in zip(bbox_min, bbox_max))
    crop_shape = tuple(bbox_max - bbox_min)
    print(f"   CT grid:     {ct_shape}")
    print(f"   Crop region: x=[{bbox_min[0]}:{bbox_max[0]}], "
          f"y=[{bbox_min[1]}:{bbox_max[1]}], z=[{bbox_min[2]}:{bbox_max[2]}]")
    print(f"   Crop shape:  {crop_shape}")

    # Update affine origin to reflect the crop offset
    ct_affine = warped["T2"].affine
    crop_offset_world = ct_affine[:3, :3] @ bbox_min
    crop_affine = ct_affine.copy()
    crop_affine[:3, 3] = ct_affine[:3, 3] + crop_offset_world

    # Crop all warped MRI images (numpy slices on first 3 dims work for both 3D and 4D)
    for name in ["T2", "ADC", "T1", "TTP"]:
        data = warped[name].get_fdata()[slices]
        warped[name] = nib.Nifti1Image(data, crop_affine)
    if "INR" in warped:
        data = warped["INR"].get_fdata()[slices]  # slices apply to first 3 dims; 4th is kept
        warped["INR"] = nib.Nifti1Image(data, crop_affine)

    # Crop CT and SUV (both already in CT space)
    ct_data_cropped = ct_nii.get_fdata()[slices]
    ct_nii_cropped = nib.Nifti1Image(ct_data_cropped, crop_affine)

    suv_data_cropped = suv_nii.get_fdata()[slices]
    suv_nii_cropped = nib.Nifti1Image(suv_data_cropped, crop_affine)

    output_shape = crop_shape
    print(f"   Final shape: {output_shape}")

    # 4. Save
    print(f"\n4. Saving processed images to {output_dir}...")
    output_paths = {}

    for name, key in [("T2", "t2"), ("ADC", "adc"), ("T1", "t1"), ("TTP", "ttp")]:
        out_path = os.path.join(output_dir, f"{key}_processed.nii.gz")
        nib.save(warped[name], out_path)
        output_paths[key] = out_path
        print(f"   Saved: {key}_processed.nii.gz")

    if "INR" in warped:
        inr_out = os.path.join(output_dir, "inr_processed.nii.gz")
        nib.save(warped["INR"], inr_out)
        output_paths['inr'] = inr_out
        print(f"   Saved: inr_processed.nii.gz ({warped['INR'].shape[3]} components)")

    ct_out = os.path.join(output_dir, "ct_processed.nii.gz")
    nib.save(ct_nii_cropped, ct_out)
    output_paths['ct'] = ct_out
    print(f"   Saved: ct_processed.nii.gz")

    suv_out = os.path.join(output_dir, "suv_processed.nii.gz")
    nib.save(suv_nii_cropped, suv_out)
    output_paths['suv'] = suv_out
    print(f"   Saved: suv_processed.nii.gz")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nAll images cropped to MRI overlap: {output_shape}")

    return output_paths


class VoxelwiseMRIDataset(Dataset):
    """
    PyTorch dataset for voxel-wise prediction.
    Each sample is a single voxel with:
        - Features: [T1, T2, ADC] values at that voxel (normalized to [0, 1])
        - Target: SUV value at that voxel (normalized to [0, 1])

    When a CT image is provided, voxels outside the soft-tissue HU range
    (air and bone) are excluded from the dataset.
    """

    def __init__(self, t2_path, adc_path, t1_path, suv_path, ct_path=None, ttp_path=None, inr_path=None,
                 hu_min=-300, hu_max=300, features=None):
        """
        Args:
            t2_path: Path to processed T2 image
            adc_path: Path to processed ADC image (feature)
            t1_path: Path to processed T1 image
            suv_path: Path to processed SUV image (prediction target)
            ct_path: Path to processed CT image (for HU gating). If None,
                     all voxels are included.
            ttp_path: Path to processed TTP image (feature)
            inr_path: Path to processed INR image (feature) has shape (x, y, z, n_components)
                      with 0 -> Ktrans, 1 -> Ve, 2 -> Vp
            normalize: If True, normalize features (MRI to [0,1], INR to z-score)
            hu_min: Minimum HU value to include (default: -300, excludes air)
            hu_max: Maximum HU value to include (default: 300, excludes bone)
            features: List of feature names to use. Available: 't1', 't2', 'adc',
                      'ttp', 'Ktrans', 've', 'vp'. Defaults to all available.
        """
        print("Loading processed images...")

        # Load feature images
        self.t2_data = nib.load(t2_path).get_fdata()[:, :, ::-1].copy()
        self.adc_data = nib.load(adc_path).get_fdata()[:, :, ::-1].copy()
        self.t1_data = nib.load(t1_path).get_fdata()[:, :, ::-1].copy()
        ttp_raw = nib.load(ttp_path).get_fdata()[:, :, ::-1].copy()
        self.ttp_data = np.nan_to_num(ttp_raw, nan=0.0, posinf=0.0, neginf=0.0)
        inr_raw = nib.load(inr_path).get_fdata()[:, :, ::-1].copy()  # inr_data shape: (x, y, z, )
        self.inr_data = np.nan_to_num(inr_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Load target image
        self.suv_data = nib.load(suv_path).get_fdata()[:, :, ::-1].copy()

        # Verify all have the same shape
        assert self.t2_data.shape == self.adc_data.shape == self.t1_data.shape == self.suv_data.shape, \
            "All images must have the same shape"

        self.shape = self.t1_data.shape
        self.features = features
        print(f"Active features: {self.features}")

        # HU gating: build index of valid (soft-tissue) voxels
        print(f"Applying HU gating (keeping {hu_min} <= HU <= {hu_max})...")
        ct_data = nib.load(ct_path).get_fdata()[:, :, ::-1].copy()
        assert ct_data.shape == self.shape, \
            f"CT shape {ct_data.shape} does not match image shape {self.shape}"
        self.hu_mask = (ct_data >= hu_min) & (ct_data <= hu_max)
        # Flatten in Fortran order (x varies fastest) to match __getitem__ decoding
        self.valid_indices = np.flatnonzero(self.hu_mask.ravel(order='F'))
        total_voxels = np.prod(self.shape)
        print(f"  Total voxels: {total_voxels}")
        print(f"  Valid (soft tissue): {len(self.valid_indices)} "
                f"({100 * len(self.valid_indices) / total_voxels:.1f}%)")
        print(f"  Excluded: {total_voxels - len(self.valid_indices)} "
                f"(air/bone/outside)")

        print("Normalizing data to [0, 1]...")
        mask = self.hu_mask
        self.t1_min, self.t1_max = self.t1_data[mask].min(), self.t1_data[mask].max()
        self.t2_min, self.t2_max = self.t2_data[mask].min(), self.t2_data[mask].max()
        self.adc_min, self.adc_max = self.adc_data[mask].min(), self.adc_data[mask].max()
        self.suv_min, self.suv_max = self.suv_data[mask].min(), self.suv_data[mask].max()
        # TTP: cohort-wise normalization by fixed acquisition maximum (203.5 s, mp_0100 -> 240)
        self.ttp_max = TTP_MAX_S
        # INR: per-component z-score; inr_data[mask] -> (n_valid, n_components)
        self.inr_mean = self.inr_data[mask].mean(axis=0)  # (n_components,)
        self.inr_std = self.inr_data[mask].std(axis=0)    # (n_components,)

        self.t1_data = (self.t1_data - self.t1_min) / (self.t1_max - self.t1_min + 1e-8)
        self.t2_data = (self.t2_data - self.t2_min) / (self.t2_max - self.t2_min + 1e-8)
        self.adc_data = (self.adc_data - self.adc_min) / (self.adc_max - self.adc_min + 1e-8)
        self.suv_data = (self.suv_data - self.suv_min) / (self.suv_max - self.suv_min + 1e-8)
        self.ttp_data = self.ttp_data / TTP_MAX_S
        # INR: inr_mean/std are (n_components,); broadcasting over (x,y,z,n_components)
        # Divide by 20 to rescale z-scores to a range comparable to [0, 1] features
        self.inr_data = (self.inr_data - self.inr_mean) / (self.inr_std + 1e-8) / 20.0

        print(f"  T1:  [{self.t1_min:.4f}, {self.t1_max:.4f}] -> [0, 1]")
        print(f"  T2:  [{self.t2_min:.4f}, {self.t2_max:.4f}] -> [0, 1]")
        print(f"  ADC: [{self.adc_min:.4f}, {self.adc_max:.4f}] -> [0, 1]")
        print(f"  TTP: cohort-wise / {TTP_MAX_S} s -> [0, ~1]")
        for c, (mu, sigma) in enumerate(zip(self.inr_mean, self.inr_std)):
            print(f"  INR[{c}]: mean={mu:.4f}, std={sigma:.4f} -> z-score")
        print(f"  SUV (target): [{self.suv_min:.4f}, {self.suv_max:.4f}] -> [0, 1]")

        print(f"Image shape: {self.shape}")
        print(f"Dataset samples: {len(self.valid_indices)}")

    @property
    def n_features(self):
        return len(self.features)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        flat_idx = self.valid_indices[idx]

        # Convert flat index to 3D coordinates
        x = flat_idx % self.shape[0]
        y = (flat_idx / self.shape[0]) % self.shape[1]
        z = flat_idx / (self.shape[0] * self.shape[1])

        feat_vals = []
        for f in self.features:
            if f == 't1':
                feat_vals.append(float(self.t1_data[x, y, z]))
            elif f == 't2':
                feat_vals.append(float(self.t2_data[x, y, z]))
            elif f == 'adc':
                feat_vals.append(float(self.adc_data[x, y, z]))
            elif f == 'ttp':
                feat_vals.append(float(self.ttp_data[x, y, z]))
            elif f == 'Ktrans':
                feat_vals.append(float(self.inr_data[x, y, z, 0]))
            elif f == 've':
                feat_vals.append(float(self.inr_data[x, y, z, 1]))
            elif f == 'vp':
                feat_vals.append(float(self.inr_data[x, y, z, 2]))

        features = torch.tensor(feat_vals, dtype=torch.float32)
        target = torch.tensor([self.suv_data[x, y, z]], dtype=torch.float32)

        return features, target

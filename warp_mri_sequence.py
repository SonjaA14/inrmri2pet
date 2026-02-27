#!/usr/bin/env python
"""
Warp any MRI sequence to CT space using an existing displacement field.

The DDF maps CT-space coordinates to MRI-space coordinates in physical (mm)
units, so it works for any image in the same physical space as the registered
MRI, regardless of resolution or spacing.

Usage:
    python warp_mri_sequence.py \
        --input  mp_0008/20231204/T2_TSE_3mm_tra_401.nii.gz \
        --ref    mp_0008/20240221/CT.nii.gz \
        --ddf    out/mp_0008/dense_displacement_field_bspline.mha \
        --output out/mp_0008/T2_TSE_warped.nii.gz \
        [--interp linear]
"""

import argparse
import nibabel as nib
import numpy as np
import SimpleITK as sitk


def load_nifti_as_sitk(path):
    """Load NIfTI via nibabel, matching register_per_bone.py convention."""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    if data.ndim == 4:
        print(f"  4D image ({list(data.shape)}), extracting first frame...")
        data = data[..., 0]
    img = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    aff = nii.affine
    sp = np.abs([aff[0, 0], aff[1, 1], aff[2, 2]])
    img.SetSpacing(sp.tolist())
    img.SetOrigin(aff[:3, 3].tolist())
    img.SetDirection((aff[:3, :3] / sp).flatten().tolist())
    return img


def save_with_ref_affine(sitk_image, ref_nifti_path, output_path):
    """Save SITK image as NIfTI using the reference image's affine.

    The custom loader (load_nifti_as_sitk) uses a different direction convention
    than native NIfTI. To ensure the output matches the reference orientation,
    we convert back to nibabel array order and save with the reference affine.
    """
    ref_nii = nib.load(ref_nifti_path)
    # SITK array is (z, y, x); undo the (2,1,0) transpose to get nibabel order
    arr = sitk.GetArrayFromImage(sitk_image).transpose(2, 1, 0)
    out_nii = nib.Nifti1Image(arr, ref_nii.affine)
    nib.save(out_nii, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Warp an MRI sequence to CT space using an existing DDF."
    )
    parser.add_argument("--input", required=True,
                        help="MRI sequence to warp (NIfTI).")
    parser.add_argument("--ref", required=True,
                        help="Reference image defining output grid (e.g. CT).")
    parser.add_argument("--ddf", required=True,
                        help="Displacement field (.mha).")
    parser.add_argument("--output", required=True,
                        help="Output warped image path.")
    parser.add_argument("--interp", default="linear",
                        choices=["linear", "bspline", "nearest"],
                        help="Interpolation method (default: linear). "
                             "Use nearest for segmentations.")
    args = parser.parse_args()

    interp_map = {
        "linear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
        "nearest": sitk.sitkNearestNeighbor,
    }

    print(f"Loading input: {args.input}")
    moving = load_nifti_as_sitk(args.input)
    print(f"  Size: {moving.GetSize()}, Spacing: {[round(s,3) for s in moving.GetSpacing()]}")

    print(f"Loading reference: {args.ref}")
    ref = load_nifti_as_sitk(args.ref)
    print(f"  Size: {ref.GetSize()}, Spacing: {[round(s,3) for s in ref.GetSpacing()]}")

    print(f"Loading DDF: {args.ddf}")
    ddf = sitk.ReadImage(args.ddf, sitk.sitkVectorFloat64)
    tx = sitk.DisplacementFieldTransform(ddf)

    print(f"Resampling with {args.interp} interpolation...")
    warped = sitk.Resample(
        moving, ref, tx,
        interp_map[args.interp], 0.0, moving.GetPixelID(),
    )

    print(f"Saving to: {args.output} (matching reference orientation)")
    save_with_ref_affine(warped, args.ref, args.output)
    print("Done.")


if __name__ == "__main__":
    main()

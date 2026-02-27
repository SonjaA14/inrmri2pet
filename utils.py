import numpy as np
import nibabel as nib
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt


def plot_results(dataset, slice_number):
    # Plot original vs predicted
    fig, axes = plt.subplots(1, 1, figsize=(15, 5))

    # Original SUV - transpose to match DICOM viewer orientation
    # suv_data is (x, y, z) but imshow expects (rows, cols) = (y, x)
    suv_slice = dataset.dataset.suv_data[:, :, slice_number].T
    im0 = axes.imshow(suv_slice, cmap='hot', origin='lower')
    axes.set_title(f'Original SUV (Slice {slice_number})')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    plt.colorbar(im0, ax=axes, label='Normalized SUV')
    # Print slice statistics for debugging
    print(f"\nSlice {slice_number} statistics:")
    print(f"  Shape: {suv_slice.shape}")
    print(f"  Min: {suv_slice.min():.4f}, Max: {suv_slice.max():.4f}")
    print(f"  Mean: {suv_slice.mean():.4f}, Std: {suv_slice.std():.4f}")


    plt.tight_layout()

    # Save figure
    plt.show()

    plt.close(fig)


def plot_overlap_check(output_paths):
    """Plot a middle slice of each resampled image for visual inspection."""
    t1 = nib.load(output_paths['t1']).get_fdata()
    t2 = nib.load(output_paths['t2']).get_fdata()
    adc = nib.load(output_paths['adc']).get_fdata()
    suv = nib.load(output_paths['suv']).get_fdata()

    z_mid = t1.shape[2] / 2

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, data, name in zip(axes, [t1, t2, adc, suv], ["T1", "T2", "ADC", "SUV"]):
        im = ax.imshow(data[:, :, z_mid].T, cmap='gray', origin='lower')
        ax.set_title(f"{name} (slice {z_mid})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax)

    fig.suptitle("Overlap region check — all images should be spatially aligned")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


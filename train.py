import nibabel as nib
import numpy as np
import json
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import matplotlib.pyplot as plt
from dataset import VoxelwiseMRIDataset, preprocess_and_save
from model import EvaluationPlotCallback, T2ADCEnvelope

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def train(recompute_data, suv_path, t1_path, t2_path, adc_path, ct_path, ddf_path, ttp_path, inr_path, patient_id,
            output_dir, epochs, learning_rate, features=None, feature_set_name=None):
    # Step 1: Preprocess and save
    print("="*60)
    print("STEP 1: Preprocessing and saving images")
    print("="*60)
    if recompute_data:
        print("Recomputing data.")
        output_paths = preprocess_and_save(
            t1_path=t1_path,
            t2_path=t2_path,
            adc_path=adc_path,
            ct_path=ct_path,
            suv_path=suv_path,
            ddf_path=ddf_path,
            ttp_path=ttp_path,
            inr_path=inr_path,
            output_dir=output_dir,
        )
    else:
        print("Reusing previously computed data.")
        output_paths = {
            't1': os.path.join(output_dir, 't1_processed.nii.gz'),
            't2': os.path.join(output_dir, 't2_processed.nii.gz'),
            'adc': os.path.join(output_dir, 'adc_processed.nii.gz'),
            'ct': os.path.join(output_dir, 'ct_processed.nii.gz'),
            'suv': os.path.join(output_dir, 'suv_processed.nii.gz'),
            'ddf': os.path.join(output_dir, 'ddf_processed.nii.gz'),
            'ttp': os.path.join(output_dir, 'ttp_processed.nii.gz'),
            'inr': os.path.join(output_dir, 'inr_processed.nii.gz')
        }

    # plot_overlap_check(output_paths)

    # Step 2: Create dataset
    print("\n" + "="*60)
    print("STEP 2: Creating PyTorch dataset")
    print("="*60)

    dataset = VoxelwiseMRIDataset(
        t1_path=output_paths['t1'],
        t2_path=output_paths['t2'],
        adc_path=output_paths['adc'],
        suv_path=output_paths['suv'],
        ct_path=output_paths['ct'],
        ttp_path=output_paths['ttp'],
        inr_path=output_paths.get('inr'),
        features=features,
    )

    print(f"\nDataset length: {len(dataset)}")

    # Split data: identify slice in the middle for plotting
    slice_number = min(dataset.shape[2] / 2, dataset.shape[2] - 1)
    img_shape = dataset.shape
    voxels_per_slice = img_shape[0] * img_shape[1]

    # Find which dataset indices (into valid_indices) belong to the eval slice
    slice_flat_start = slice_number * voxels_per_slice
    slice_flat_end = (slice_number + 1) * voxels_per_slice
    is_eval = ((dataset.valid_indices >= slice_flat_start) &
               (dataset.valid_indices < slice_flat_end))
    eval_indices = np.where(is_eval)[0].tolist()

    print(f"Training samples: {len(dataset)} (all slices)")
    print(f"Plot / Evaluation samples: {len(eval_indices)} (slice {slice_number})")

    # Eval subset for plotting; training uses the full dataset
    eval_dataset = torch.utils.data.Subset(dataset, eval_indices)

    # Create DataLoader for training (all voxels, including eval slice)
    train_loader = DataLoader(
        dataset,
        batch_size=4096,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True
    )

    # Step 3: Create model
    print("\n" + "="*60)
    print("STEP 3: Creating SIREN MRI→SUV model")
    print("="*60)
    mapping_size = 256
    n_features = dataset.n_features
    B_gauss = torch.randn((mapping_size / 2, n_features)).to(DEVICE) * 10

    model = T2ADCEnvelope(
        in_features=mapping_size * n_features,
        hidden_features=512,
        hidden_layers=3,
        out_features=1,  # SUV prediction
        B=B_gauss,
        lam_parallel=1,  # Weight for parallel loss component
        ridge=1e-3,
        learning_rate=learning_rate,
    )

    print(f"\nModel architecture:")
    print(model)

    # Step 4: Train model
    print("\n" + "="*60)
    print(f"STEP 4: Training for {epochs} epochs")
    print("="*60)


    run = wandb.init(
        entity="",
        project="inrmri",
        name=f"{patient_id}_{feature_set_name}",
        config={
            "learning_rate": learning_rate,
            "dataset": "",
            "epochs": epochs,
            "patient_id": patient_id,
            "features": features,
            "feature_set_name": feature_set_name,
            "feat_t1": "t1" in dataset.features,
            "feat_t2": "t2" in dataset.features,
            "feat_adc": "adc" in dataset.features,
            "feat_ttp": "ttp" in dataset.features,
            "feat_Ktrans": any(f.startswith('Ktrans') for f in dataset.features),
            "feat_ve": any(f.startswith('ve') for f in dataset.features),
            "feat_vp": any(f.startswith('vp') for f in dataset.features),
            "n_features": dataset.n_features,
            "inr_norm_zscore": True,
        },
    )

    wandb_logger = WandbLogger(experiment=run, log_model=True)

    # Build 2D mask for the eval slice (which pixels have valid data)
    if dataset.hu_mask is not None:
        eval_slice_mask = dataset.hu_mask[:, :, slice_number]
    else:
        eval_slice_mask = np.ones((img_shape[0], img_shape[1]), dtype=bool)

    eval_callback = EvaluationPlotCallback(
        eval_dataset=eval_dataset,
        img_shape=img_shape,
        slice_number=slice_number,
        slice_mask=eval_slice_mask,
        device=DEVICE,
        every_n_epochs=15
    )

    checkpoint_dir = os.path.join(output_dir, "checkpoints", f"{patient_id}_{feature_set_name}")
    ckpt_best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="train/loss",
        mode="min",
        save_top_k=1,
    )
    ckpt_last = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="last",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[eval_callback, ckpt_best, ckpt_last],
    )

    trainer.fit(model, train_loader)

    final_metrics = {
        k: v.item()
        for k, v in trainer.callback_metrics.items()
        if isinstance(v, torch.Tensor)
    }

    run.finish()

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    return {
        "patient_id": patient_id,
        "feature_set_name": feature_set_name,
        "features": "|".join(features or []),
        "run_path": run.path,
        **final_metrics,
    }


if __name__ == '__main__':
    main()

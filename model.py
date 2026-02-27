import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback

import torch
import pytorch_lightning as pl


# =============================================================================
# Evaluation Plot Callback
# =============================================================================

class EvaluationPlotCallback(Callback):
    """Callback to generate and log evaluation plots every N epochs."""

    def __init__(self, eval_dataset, img_shape, slice_number, slice_mask, device, every_n_epochs=5):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.img_shape = img_shape
        self.slice_number = slice_number
        self.slice_mask = slice_mask  # 2D bool array (W, H) — valid pixels in eval slice
        self.device = device
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1  # 1-indexed for display

        # Run every N epochs (and on the first epoch)
        if current_epoch % self.every_n_epochs != 0 and current_epoch != 1:
            return

        # Create a temporary DataLoader for evaluation
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=4096,
            shuffle=False,
            num_workers=0,  # Use 0 workers to avoid cessing issues in callback
            pin_memory=False
        )

        # Run inference
        pl_module.eval()
        predictions = []
        targets = []
        features = []

        with torch.no_grad():
            for batch in eval_loader:
                x, y = batch
                x = x.to(self.device)
                y_hat = pl_module(x)
                predictions.append(y_hat.cpu())
                targets.append(y)
                features.append(x.cpu())

        pl_module.train()

        # Concatenate as tensors for ortho/parallel decomposition
        predictions_t = torch.cat(predictions, dim=0)
        targets_t = torch.cat(targets, dim=0)
        features_t = torch.cat(features, dim=0)

        # Compute ortho/parallel error decomposition
        targets_t = targets_t.squeeze(-1).unsqueeze(1) if targets_t.ndim > 1 else targets_t.unsqueeze(1)
        X_n = pl_module._normalise(features_t, pl_module._normalise_features).clamp(-10, 10)
        projection_n = pl_module._normalise(targets_t, pl_module._normalise_target)
        e = predictions_t - projection_n

        r_parallel, r_ortho = pl_module.svd_ortho_parallel(X_n, e)

        # Convert to numpy and scatter into full 2D slice.
        # slice_mask is (W, H); transpose to (H, W) for imshow display.
        H, W = self.img_shape[1], self.img_shape[0]
        mask_2d = self.slice_mask.T  # (H, W)
        valid_y, valid_x = np.where(mask_2d)

        def _scatter(values):
            full = np.full((H, W), np.nan)
            full[valid_y, valid_x] = values.numpy().squeeze()
            return full

        pred_slice = _scatter(predictions_t)
        target_slice = _scatter(projection_n)
        diff_slice = pred_slice - target_slice
        parallel_slice = _scatter(r_parallel)
        ortho_slice = _scatter(r_ortho)

        # Extract raw data slices directly from the eval dataset
        def _get_masked_slice(arr_2d):
            """arr_2d is (W, H); apply mask and return (H, W) for imshow."""
            out = arr_2d.copy().astype(float)
            out[~self.slice_mask] = np.nan
            return out.T  # (H, W)

        ds = self.eval_dataset.dataset if hasattr(self.eval_dataset, 'dataset') else self.eval_dataset
        t1_slice  = _get_masked_slice(ds.t1_data[:, :, self.slice_number])
        t2_slice  = _get_masked_slice(ds.t2_data[:, :, self.slice_number])
        adc_slice = _get_masked_slice(ds.adc_data[:, :, self.slice_number])
        ttp_slice = _get_masked_slice(ds.ttp_data[:, :, self.slice_number])
        suv_gt_slice  = _get_masked_slice(ds.suv_data[:, :, self.slice_number])
        ktrans_slice  = _get_masked_slice(ds.inr_data[:, :, self.slice_number, 0])
        ve_slice      = _get_masked_slice(ds.inr_data[:, :, self.slice_number, 1])
        vp_slice      = _get_masked_slice(ds.inr_data[:, :, self.slice_number, 2])

        # Create matplotlib figure
        vmax = max(np.nanmax(np.abs(diff_slice)), 1e-6)
        vmax_par = max(np.nanmax(np.abs(parallel_slice)), 1e-6)
        vmax_ort = max(np.nanmax(np.abs(ortho_slice)), 1e-6)
        mae = np.nanmean(np.abs(diff_slice))
        mse = np.nanmean(diff_slice ** 2)

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        # --- Row 1: input modalities ---
        im = axes[0, 0].imshow(t1_slice, cmap='gray', origin='lower')
        axes[0, 0].set_title(f'T1')
        plt.colorbar(im, ax=axes[0, 0], label='Norm. intensity [a.u.]')

        im = axes[0, 1].imshow(t2_slice, cmap='gray', origin='lower')
        axes[0, 1].set_title('T2')
        plt.colorbar(im, ax=axes[0, 1], label='Norm. intensity [a.u.]')

        im = axes[0, 2].imshow(adc_slice, cmap='gray', origin='lower')
        axes[0, 2].set_title('ADC')
        plt.colorbar(im, ax=axes[0, 2], label='Norm. ADC [a.u.]')

        im = axes[0, 3].imshow(ttp_slice, cmap='gray', origin='lower')
        axes[0, 3].set_title('TTP')
        plt.colorbar(im, ax=axes[0, 3], label='TTP / 240 s')

        # --- Row 2: targets / pharmacokinetics ---
        im = axes[1, 0].imshow(suv_gt_slice, cmap='hot', origin='lower')
        axes[1, 0].set_title('SUV (Ground Truth)')
        plt.colorbar(im, ax=axes[1, 0], label='Norm. SUV [a.u.]')

        im = axes[1, 1].imshow(ktrans_slice, cmap='viridis', origin='lower')
        axes[1, 1].set_title(r'$K_\mathrm{trans}$')
        plt.colorbar(im, ax=axes[1, 1], label=r'$K_\mathrm{trans}$ z-score / 20 [min⁻¹]')

        im = axes[1, 2].imshow(ve_slice, cmap='viridis', origin='lower')
        axes[1, 2].set_title(r'$v_e$')
        plt.colorbar(im, ax=axes[1, 2], label=r'$v_e$ z-score / 20 [a.u.]')

        im = axes[1, 3].imshow(vp_slice, cmap='viridis', origin='lower')
        axes[1, 3].set_title(r'$v_p$')
        plt.colorbar(im, ax=axes[1, 3], label=r'$v_p$ z-score / 20 [a.u.]')

        # --- Row 3: predictions and errors ---
        im = axes[2, 0].imshow(pred_slice, cmap='hot', origin='lower')
        axes[2, 0].set_title(f'SUV (Predicted) (Epoch {current_epoch})')
        plt.colorbar(im, ax=axes[2, 0], label='Norm. SUV [a.u.]')

        im = axes[2, 1].imshow(np.abs(diff_slice), cmap='magma', origin='lower', vmin=0, vmax=vmax)
        axes[2, 1].set_title(f'Difference')
        plt.colorbar(im, ax=axes[2, 1], label='|ΔSUV| [a.u.]')

        im = axes[2, 2].imshow(parallel_slice**2, cmap='magma', origin='lower', vmin=0, vmax=vmax_par**2)
        axes[2, 2].set_title(f'Parallel Error')
        plt.colorbar(im, ax=axes[2, 2], label='Squared error [a.u.²]')

        im = axes[2, 3].imshow(ortho_slice**2, cmap='magma', origin='lower', vmin=0, vmax=vmax_ort**2)
        axes[2, 3].set_title(f'Orthogonal Error')
        plt.colorbar(im, ax=axes[2, 3], label='Squared error [a.u.²]')

        for ax in axes.flat:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        plt.tight_layout()

        # Log to wandb - pass matplotlib figure directly, wandb auto-converts to interactive Plotly
        if trainer.logger:
            trainer.logger.experiment.log({
                f'evaluation/slice_{self.slice_number}': fig,
                'eval_slice/mae': mae,
                'eval_slice/mse': mse,
                'eval_slice/rmse': np.sqrt(mse),
            }, step=trainer.global_step)

        plt.close(fig)

# =============================================================================
# SIREN model with OrthoPall Loss for MRI → SUV prediction
# =============================================================================

class T2ADCEnvelope(pl.LightningModule):
    def __init__(
        self,
        in_features: int = 512,
        hidden_features: int = 512,
        hidden_layers: int = 7,
        out_features: int = 1,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30.,
        B=None,
        lam_parallel: float = 0.5,
        ridge: float = 1e-3,
        eps: float = 1e-12,
        centre_target: bool = True,
        centre_and_scale_feats: bool = True,
        learning_rate: float = 1e-5,
    ):
        super().__init__()
        self.lam_parallel = lam_parallel
        self.ridge = ridge
        self.eps = eps
        self.centre_target = centre_target
        self.centre_and_scale_feats = centre_and_scale_feats
        self.learning_rate = learning_rate
        self._normalise_features = "zscore" if centre_and_scale_feats else "none"
        self._normalise_target = "zscore" if centre_target else "none"

        # Fourier feature mapping matrix — registered as a buffer so it is
        # saved in the checkpoint state dict and moved with model.to(device).
        self.register_buffer('B', B)

        # Build SIREN network
        layers = [SineLayer(in_features, hidden_features,
                            is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                    is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*layers)

        # Output layer (linear for regression)
        self.output_layer = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            self.output_layer.weight.uniform_(
                -np.sqrt(6 / hidden_features) / hidden_omega_0,
                 np.sqrt(6 / hidden_features) / hidden_omega_0)

        self.save_hyperparameters(ignore=['B'])

    @staticmethod
    def _zscore(x: torch.Tensor, eps: float = 1e-12):
        mu = x.mean(dim=0, keepdim=True)
        sd = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - mu) / (sd + 1e-6)

    def _normalise(self, x: torch.Tensor, mode: str):
        if mode == "none":
            return x
        if mode == "zscore":
            return self._zscore(x, eps=self.eps)
        raise ValueError(f"Unknown normalisation mode: {mode}")

    @staticmethod
    def svd_ortho_parallel(X_n: torch.Tensor, e: torch.Tensor):
        """Decompose error e into components parallel and orthogonal to col(X_n)."""
        U, S, Vh = torch.linalg.svd(X_n, full_matrices=False)
        tol = 1e-6 * S.max().clamp_min(1e-12)
        r = (S > tol).sum().item()
        U_r = U[:, :max(r, 1)]
        r_parallel = U_r @ (U_r.transpose(0, 1) @ e)
        r_ortho = e - r_parallel
        return r_parallel, r_ortho

    def training_step(self, batch, batch_idx):
        """
        B: Batch size (number of voxels in this batch)
        N: Number of features (T1, T2, ADC) per voxel
        Expected batch: tuple of (features, target)
          features: [B, N]  = [T1, T2, ADC] values for B voxels
          target:   [B, 1]  = SUV values for B voxels
        """
        x, projection = batch
        
        projection = projection.squeeze(-1).unsqueeze(1) if projection.ndim > 1 else projection.unsqueeze(1)  # [B, 1] or [B] -> [B, 1]

        N = x.shape[1]

        # Normalise within this voxel-batch (per-patient step); 
        # clamp to avoid extreme values from constant features
        X_n = self._normalise(x, self._normalise_features).clamp(-10, 10)  # [B,N]

        projection_n = self._normalise(projection, self._normalise_target)    # [B,1]

        # Predict SUV
        y_hat = self(x)

        # Error field; trained to predict the normalised target
        e = y_hat - projection_n  # [B,1]

        # --- SVD-based projection ---
        r_parallel, r_ortho = self.svd_ortho_parallel(X_n, e)

        # Loss terms (mean squared)
        mse_total = (e ** 2).mean()
        mse_parallel = (r_parallel ** 2).mean()
        mse_ortho = (r_ortho ** 2).mean()
 
        loss = mse_total + self.lam_parallel * mse_parallel

        # Logs
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mse_total", mse_total)
        self.log("train/mse_parallel", mse_parallel)
        self.log("train/mse_ortho", mse_ortho)
        self.log("train/mse_total_std", (e ** 2).std())
        self.log("train/mse_parallel_std", (r_parallel ** 2).std())
        self.log("train/mse_ortho_std", (r_ortho ** 2).std())
        return loss

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # When the model is rebuilt with B=None (B is excluded from hparams),
        # register_buffer('B', None) does not appear in state_dict(), so
        # PyTorch raises "unexpected key: B" when loading a checkpoint that has
        # B saved.  Re-registering with the actual tensor before load_state_dict
        # is applied fixes the mismatch.
        sd = checkpoint.get('state_dict', {})
        if 'B' in sd and self.B is None:
            self.register_buffer('B', sd['B'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }

    def forward(self, x):
        if self.B is not None:
            # Independent Fourier features per input dimension
            # x: (batch, n_feat=3), B: (n_freq=128, n_feat=3)
            proj = 2. * torch.pi * x.unsqueeze(1) * self.B.unsqueeze(0)  # (batch, n_freq, n_feat)
            x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)     # (batch, 2*n_freq, n_feat)
            x = x.reshape(x.shape[0], -1)                                # (batch, 2*n_freq*n_feat = 768)
        x = self.net(x)
        x = self.output_layer(x)
        return x

class SineLayer(nn.Module):
    """SIREN sine activation layer."""

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


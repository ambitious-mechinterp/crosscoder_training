from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.models.activations.topk import (
    BatchTopkActivation,
    GroupMaxActivation,
    TopkActivation,
    topk_activation,
)
from crosscode.trainers.base_acausal_trainer import BaseModelHookpointAcausalTrainer
from crosscode.trainers.topk_crosscoder.config import TopKTrainConfig
from crosscode.trainers.utils import get_l0_stats, wandb_histogram
from crosscode.utils import calculate_reconstruction_loss_summed_norm_MSEs, not_none, l0_norm, l2_norm


class TopKStyleAcausalCrosscoderTrainer(
    BaseModelHookpointAcausalTrainer[TopKTrainConfig, BatchTopkActivation | GroupMaxActivation | TopkActivation]
):
    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: ModelHookpointAcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(batch_BMPD, train_res.recon_acts_BMPD)
        aux_loss = self.aux_loss(batch_BMPD, train_res)
        loss = reconstruction_loss + self.cfg.lambda_aux * aux_loss

        if log:
            # Calculate MSE
            mse = F.mse_loss(batch_BMPD, train_res.recon_acts_BMPD)
            
            # Calculate cosine similarity
            # Flatten the tensors to compute cosine similarity across all dimensions
            batch_flat = batch_BMPD.reshape(batch_BMPD.shape[0], -1)
            recon_flat = train_res.recon_acts_BMPD.reshape(train_res.recon_acts_BMPD.shape[0], -1)
            cosine_sim = F.cosine_similarity(batch_flat, recon_flat, dim=1).mean()

            # Calculate L0 norm per example
            l0_per_example_B = l0_norm(train_res.latents_BL, dim=-1)
            # Create L0 histogram for WandB
            l0_histogram = wandb_histogram(l0_per_example_B)

            log_dict: dict[str, Any] = {
                "train/loss": loss.item(),
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/aux_loss": aux_loss.item(),
                "train/aux_loss_weighted": self.cfg.lambda_aux * aux_loss.item(),
                "train/n_dead_latents": torch.sum(self.firing_tracker.tokens_since_fired_L > self.cfg.dead_latents_threshold_n_examples).item(),
                "train/reconstruction_mse": mse.item(),
                "train/reconstruction_cosine_sim": cosine_sim.item(),
                "media/l0_distribution": l0_histogram,
                **self._get_fvu_dict(batch_BMPD, train_res.recon_acts_BMPD),
                **get_l0_stats(train_res.latents_BL),
            }

            # <<< --- Calculate L2 norms of activation vectors AFTER scaling --- >>>
            # batch_BMPD is already scaled by the dataloader
            scaled_norms_BMP = l2_norm(batch_BMPD, dim=-1) # Result shape: (Batch, Model, Hookpoint)

            # Log histogram per model and hookpoint
            # Assuming self.n_models and self.hookpoints are available from BaseModelHookpointAcausalTrainer
            for m_idx in range(self.n_models):
                for p_idx, hp_name in enumerate(self.hookpoints):
                    # Extract norms for this specific model and hookpoint across the batch
                    norms_for_hist_B = scaled_norms_BMP[:, m_idx, p_idx]

                    # Create histogram using your utility
                    hist = wandb_histogram(norms_for_hist_B)

                    # Add to log dict under a descriptive key
                    log_key = f"media/scaled_norm_dist/model{m_idx}_hookpoint{hp_name}"
                    log_dict[log_key] = hist

            return reconstruction_loss, log_dict

        return reconstruction_loss, None

    def aux_loss(
        self, batch_BMPD: torch.Tensor, train_res: ModelHookpointAcausalCrosscoder.ForwardResult
    ) -> torch.Tensor:
        """train to reconstruct the error with the topk dead latents"""
        return aux_loss(
            pre_activations_BL=train_res.pre_activations_BL,
            dead_features_mask_L=self.firing_tracker.tokens_since_fired_L > self.cfg.dead_latents_threshold_n_examples,
            k_aux=not_none(self.cfg.k_aux),
            decode_BXD=self.model.decode_BMPD,
            error_BXD=batch_BMPD - train_res.recon_acts_BMPD,
        )


def aux_loss(
    pre_activations_BL: torch.Tensor,
    dead_features_mask_L: torch.Tensor,
    k_aux: int,
    decode_BXD: Callable[[torch.Tensor], torch.Tensor],
    error_BXD: torch.Tensor,
) -> torch.Tensor:
    if (topk_aux_output := topk_dead_latents(pre_activations_BL, dead_features_mask_L, k_aux)) is None:
        return torch.tensor(0.0, device=pre_activations_BL.device)

    aux_latents_BL, n_latents_used = topk_aux_output

    # If there's less than `k_aux` dead features, it's harder to reconstruct the error. so scale down the loss.
    aux_loss_scale = n_latents_used / k_aux

    # try to reconstruct the error with the topk dead latents
    error_recon_mse = calculate_reconstruction_loss_summed_norm_MSEs(decode_BXD(aux_latents_BL), error_BXD)
    return error_recon_mse * aux_loss_scale


def topk_dead_latents(
    pre_activations_BL: torch.Tensor,
    dead_features_mask_L: torch.Tensor,
    k_aux: int,
) -> tuple[torch.Tensor, int] | None:
    n_dead = int(dead_features_mask_L.sum())
    if n_dead == 0:
        return None

    dead_latents_BL = pre_activations_BL * dead_features_mask_L

    # we only need to actually do the topk operation if there are more dead features than k_aux_base
    aux_latents_BL = topk_activation(dead_latents_BL, k_aux) if n_dead > k_aux else dead_latents_BL
    n_latents_used = min(n_dead, k_aux)

    return aux_latents_BL, n_latents_used

from typing import Any

import einops
import torch
import torch.nn.functional as F

from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.models.activations.relu import ReLUActivation
from crosscode.trainers.base_acausal_trainer import BaseModelHookpointAcausalTrainer
from crosscode.trainers.l1_crosscoder.config import L1TrainConfig
from crosscode.trainers.utils import get_l0_stats
from crosscode.utils import calculate_reconstruction_loss_summed_norm_MSEs, l1_norm, l2_norm


class L1CrosscoderTrainer(BaseModelHookpointAcausalTrainer[L1TrainConfig, ReLUActivation]):
    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: ModelHookpointAcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(batch_BMPD, train_res.recon_acts_BMPD)

        sparsity_loss = sparsity_loss_l1_of_l2s(self.model.W_dec_LMPD, train_res.latents_BL)

        loss = reconstruction_loss + self._l1_coef_scheduler() * sparsity_loss

        if log:
            # Calculate MSE
            mse = F.mse_loss(batch_BMPD, train_res.recon_acts_BMPD)
            
            # Calculate cosine similarity
            batch_flat = batch_BMPD.reshape(batch_BMPD.shape[0], -1)
            recon_flat = train_res.recon_acts_BMPD.reshape(train_res.recon_acts_BMPD.shape[0], -1)
            cosine_sim = F.cosine_similarity(batch_flat, recon_flat, dim=1).mean()
            
            # Count dead latents
            n_dead_latents = torch.sum(self.firing_tracker.tokens_since_fired_L > self.cfg.dead_latents_threshold_n_examples).item()

            log_dict: dict[str, Any] = {
                "train/loss": loss.item(),
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/sparsity_loss_weighted": self._l1_coef_scheduler() * sparsity_loss.item(),
                "train/n_dead_latents": n_dead_latents,
                "train/reconstruction_mse": mse.item(),
                "train/reconstruction_cosine_sim": cosine_sim.item(),
                **self._get_fvu_dict(batch_BMPD, train_res.recon_acts_BMPD),
                **get_l0_stats(train_res.latents_BL),
            }

            return loss, log_dict

        return loss, None

    def _step_logs(self) -> dict[str, Any]:
        return {
            **super()._step_logs(),
            "train/l1_coef": self._l1_coef_scheduler(),
        }

    def _l1_coef_scheduler(self) -> float:
        if self.step < self.cfg.lambda_s_n_steps:
            return self.cfg.final_lambda_s * self.step / self.cfg.lambda_s_n_steps
        else:
            return self.cfg.final_lambda_s


def sparsity_loss_l1_of_l2s(
    W_dec_LMPD: torch.Tensor,
    latents_BL: torch.Tensor,
) -> torch.Tensor:
    assert (latents_BL >= 0).all()
    # think about it like: each latent has a separate projection onto each (model, hookpoint)
    # so we have a separate l2 norm for each (latent, model, hookpoint)
    norms_LX = einops.reduce(W_dec_LMPD, "... d_model -> ...", l2_norm)
    l1_of_norms_L = einops.reduce(norms_LX, "l ... -> l", l1_norm)

    # now we weight the latents by the sum of their output l2 norms
    weighted_latents_BL = latents_BL * l1_of_norms_L
    l1_of_weighted_latents_B = einops.reduce(weighted_latents_BL, "batch latent -> batch", l1_norm)

    return l1_of_weighted_latents_B.mean()

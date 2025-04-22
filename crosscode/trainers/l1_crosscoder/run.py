import fire  # type: ignore
import torch
import os

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models import AnthropicTransposeInit, ReLUActivation
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.trainers.base_trainer import run_exp
from crosscode.trainers.l1_crosscoder.config import L1ExperimentConfig
from crosscode.trainers.l1_crosscoder.trainer import L1CrosscoderTrainer
from crosscode.trainers.utils import build_wandb_run
from crosscode.utils import get_device


def build_l1_crosscoder_trainer(cfg: L1ExperimentConfig) -> L1CrosscoderTrainer:
    # This works with both CUDA_VISIBLE_DEVICES and explicit device selection
    device = get_device(cuda_device=cfg.cuda_device if hasattr(cfg, 'cuda_device') else None)
    
    # Print diagnostic info
    if torch.cuda.is_available():
        cuda_devices_env = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
        num_visible_gpus = torch.cuda.device_count()
        current_device = device.index if device.type == 'cuda' else 'cpu'
        
        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_devices_env}")
        logger.info(f"Number of visible GPUs: {num_visible_gpus}")
        logger.info(f"Selected device: {device} (index: {current_device})")
        
    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        inferenced_type=cfg.data.activations_harvester.inference_dtype,
    )

    dataloader = build_model_hookpoint_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    crosscoder = ModelHookpointAcausalCrosscoder(
        n_models=len(llms),
        n_hookpoints=len(cfg.hookpoints),
        d_model=llms[0].cfg.d_model,
        n_latents=cfg.crosscoder.n_latents,
        activation_fn=ReLUActivation(),
        use_encoder_bias=cfg.crosscoder.use_encoder_bias,
        use_decoder_bias=cfg.crosscoder.use_decoder_bias,
        init_strategy=AnthropicTransposeInit(dec_init_norm=cfg.crosscoder.dec_init_norm),
    )

    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

    return L1CrosscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        model=crosscoder,
        wandb_run=wandb_run,
        device=device,
        save_dir=cfg.save_dir
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_l1_crosscoder_trainer, L1ExperimentConfig))

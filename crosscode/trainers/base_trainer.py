import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from crosscode.data.activations_dataloader import ActivationsDataloader, ModelHookpointActivationsBatch
from crosscode.log import logger
from crosscode.models.base_crosscoder import BaseCrosscoder
from crosscode.trainers.config_common import BaseExperimentConfig, BaseTrainConfig
from crosscode.trainers.firing_tracker import FiringTracker
from crosscode.trainers.utils import build_lr_scheduler, build_optimizer, dict_join, wandb_histogram, build_wandb_run
from crosscode.utils import get_device

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TModel = TypeVar("TModel", bound=BaseCrosscoder[Any])
TBatch = TypeVar("TBatch")


class BufferedModelHookpointActivationsDataloader(ActivationsDataloader[ModelHookpointActivationsBatch]):
    """A dataloader that buffers multiple batches and shuffles them before yielding.
    
    This provides better shuffling by mixing data across multiple batches while maintaining
    the original batch size and tensor structure.
    """
    
    def __init__(
        self,
        base_dataloader: ActivationsDataloader[ModelHookpointActivationsBatch],
        buffer_size: int = 20,
    ):
        """Initialize the buffered dataloader.
        
        Args:
            base_dataloader: The base dataloader to buffer batches from
            buffer_size: Number of batches to collect before shuffling
        """
        self.base_dataloader = base_dataloader
        self.buffer_size = buffer_size
        self._base_iterator = None

    @property
    def n_models(self) -> int:
        """Number of models in the dataloader."""
        return self.base_dataloader.n_models

    @property
    def hookpoints(self) -> list[str]:
        """List of hookpoints in the dataloader."""
        return self.base_dataloader.hookpoints

    @property
    def n_hookpoints(self) -> int:
        """Number of hookpoints in the dataloader."""
        return self.base_dataloader.n_hookpoints
        
    def get_activations_iterator(self) -> Iterator[ModelHookpointActivationsBatch]:
        """Get an iterator that yields shuffled batches from the buffer."""
        base_iterator = self.base_dataloader.get_activations_iterator()
        
        while True:
            # Collect buffer_size batches
            buffer_batches = []
            for _ in range(self.buffer_size):
                try:
                    batch = next(base_iterator)
                    buffer_batches.append(batch.activations_BMPD)
                except StopIteration:
                    if not buffer_batches:  # If buffer is empty, we're done
                        return
                    break
            
            if not buffer_batches:  # If we couldn't collect any batches, we're done
                return
                
            try:
                # Stack all batches along the batch dimension
                combined_tensor = torch.cat(buffer_batches, dim=0)  # Shape: (buffer_size*B, M, P, D)
                
                # Get the total number of samples in the buffer
                total_samples = combined_tensor.shape[0]
                
                # Create a random permutation of indices
                indices = torch.randperm(total_samples)
                
                # Original batch size from the base dataloader
                original_batch_size = buffer_batches[0].shape[0]
                
                # Yield shuffled batches of the original batch size
                for i in range(0, total_samples, original_batch_size):
                    batch_indices = indices[i:i + original_batch_size]
                    if len(batch_indices) < original_batch_size:
                        # Skip the last batch if it's smaller than the original batch size
                        continue
                    shuffled_batch = combined_tensor[batch_indices]
                    yield ModelHookpointActivationsBatch(shuffled_batch)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("CUDA out of memory while buffering batches. Reducing buffer size...")
                    self.buffer_size = max(1, self.buffer_size // 2)
                    continue
                raise

    def get_scaling_factors(self) -> torch.Tensor:
        """Get the scaling factors from the base dataloader."""
        return self.base_dataloader.get_scaling_factors()


class BaseTrainer(Generic[TConfig, TModel, TBatch], ABC):
    LOG_HISTOGRAMS_EVERY_N_LOGS = 10

    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: ActivationsDataloader[TBatch],
        model: TModel,
        wandb_run: Run,
        device: torch.device,
        save_dir: Path | str,
        buffer_size: int | None = 20,
    ):
        self.cfg = cfg
        # Wrap the dataloader with buffering if buffer_size is specified
        if buffer_size is not None:
            self.activations_dataloader = BufferedModelHookpointActivationsDataloader(
                activations_dataloader,
                buffer_size=buffer_size
            )
        else:
            self.activations_dataloader = activations_dataloader

        self.model = model
        self.wandb_run = wandb_run
        self.device = device

        self.optimizer = build_optimizer(cfg.optimizer, model.parameters())

        self.lr_scheduler = build_lr_scheduler(cfg.optimizer, cfg.num_steps) if cfg.optimizer.type == "adam" else None

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.firing_tracker = FiringTracker(activation_size=model.n_latents, device=self.device)

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def train(self) -> None:
        # scaling_factors_MP = self.activations_dataloader.get_norm_scaling_factors_MP().to(self.device)
        epoch_dataloader = self.activations_dataloader.get_activations_iterator()

        for i in tqdm(
            range(self.cfg.num_steps),
            desc="Train Steps",
            smoothing=0.15,  # this loop is bursty because of activation harvesting
        ):
            """
            if i == 0:
                print('Started training, verify dec enc are transposes')
                enc_1 = self.model.W_enc_MPDL[0,0]
                dec_1 = self.model.W_dec_LMPD[:,0,0,:]
                assert torch.allclose(dec_1.T, enc_1), f'Dec enc not close, see enc{enc_1[:10,:10]} and dec {dec_1[:10,:10]}'
                print('All close!')
            """
            self._lr_step()
            self.optimizer.zero_grad()

            log_dicts: list[dict[str, float]] = []
            log = self.step % self.cfg.log_every_n_steps == 0

            for _ in range(self.cfg.gradient_accumulation_steps_per_batch):
                loss, log_dict, tokens_trained = self.run_batch(next(epoch_dataloader), log)
                if self.epoch == 0:
                    self.unique_tokens_trained += tokens_trained

                loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()
                if log_dict is not None:
                    log_dicts.append(log_dict)

            self._after_forward_passes()

            if log_dicts:
                batch_log_dict_avgs = {
                    **{k: sum(v) / len(v) for k, v in dict_join(log_dicts).items()},
                    **self._step_logs(),
                }
                self.wandb_run.log(batch_log_dict_avgs, step=self.step)

            self._maybe_save_model()

            #clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.step += 1

        self.wandb_run.finish()

    def _after_forward_passes(self): ...

    @abstractmethod
    def run_batch(self, batch: TBatch, log: bool) -> tuple[torch.Tensor, dict[str, float] | None, int]: ...

    @abstractmethod
    def _maybe_save_model(self) -> None: ...

    def _lr_step(self) -> None:
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        if self.lr_scheduler is not None:
            self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

    def _step_logs(self) -> dict[str, Any]:
        log_dict: dict[str, Any] = {
            "train/epoch": self.epoch,
            "train/unique_tokens_trained": self.unique_tokens_trained,
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        if self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0:
            tokens_since_fired_hist = wandb_histogram(self.firing_tracker.tokens_since_fired_L)
            log_dict.update({"media/tokens_since_fired": tokens_since_fired_hist})
            if self.model.b_enc_L is not None:
                log_dict["b_enc"] = wandb_histogram(self.model.b_enc_L)

        return log_dict


def save_config(config: BaseExperimentConfig) -> None:
    config.save_dir.mkdir(parents=True, exist_ok=True)
    with open(config.save_dir / "experiment_config.yaml", "w") as f:
        yaml.dump(config.model_dump(), f)
    logger.info(f"Saved config to {config.save_dir / 'experiment_config.yaml'}")


TCfg = TypeVar("TCfg", bound=BaseExperimentConfig)


def run_exp(build_trainer: Callable[[TCfg], Any], cfg_cls: type[TCfg]) -> Callable[[Path], None]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def inner(config_path: Path) -> None:
        config_path = Path(config_path)
        assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
        assert Path(config_path).exists(), f"Config file {config_path} does not exist."
        logger.info("Loading config...")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        logger.info(f"Loaded config (raw):\n{config_dict}")
        config = cfg_cls(**config_dict)
        logger.info(f"Loaded config (parsed):\n{config.model_dump_json(indent=2)}")
        config.experiment_name += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        logger.info(f"over-wrote experiment_name: {config.experiment_name}")
        logger.info(f"saving in save_dir: {config.save_dir}")
        save_config(config)
        logger.info("Building trainer")
        device = get_device(cuda_device=config.cuda_device if hasattr(config, 'cuda_device') else None)
        wandb_run = build_wandb_run(config)
        trainer = build_trainer(config)
        logger.info("Training")
        trainer.train()

    return inner

from math import prod
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar
import contextlib

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


class BufferedModelHookpointActivationsDataloader(
    ActivationsDataloader[ModelHookpointActivationsBatch]
):
    """Fixed‑memory wrapper around another activations dataloader.

    Parameters
    ----------
    base_dataloader : ActivationsDataloader
        The source dataloader producing activations.
    buffer_size : int, default 200
        Number of *mini‑batches* held in the circular buffer.
    device : torch.device, optional
        Device to which emitted batches are copied.  The buffer itself always
        resides on the CPU to keep GPU memory free.
    refill_ratio : float in (0, 1), default 0.25
        Fraction of the buffer that must be consumed before the unread tail is
        compacted and the free space refilled.
    """

    def __init__(
        self,
        base_dataloader: ActivationsDataloader[ModelHookpointActivationsBatch],
        buffer_size: int = 200,
        device: torch.device | None = None,
        refill_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if not (0.0 < refill_ratio < 1.0):
            raise ValueError("refill_ratio must be in (0, 1)")

        self._base = base_dataloader
        self._buffer_batches = buffer_size
        self._device = device or torch.device("cpu")
        self._refill_ratio = refill_ratio

        self._memory_logged = False  # print memory footprint once only

    # ------------------------------------------------------------------
    # Passthrough metadata ---------------------------------------------
    # ------------------------------------------------------------------
    @property
    def n_models(self) -> int:  # noqa: D401
        return self._base.n_models

    @property
    def hookpoints(self) -> list[str]:  # noqa: D401
        return self._base.hookpoints

    @property
    def n_hookpoints(self) -> int:  # noqa: D401
        return self._base.n_hookpoints

    # ------------------------------------------------------------------
    # Iterator ---------------------------------------------------------
    # ------------------------------------------------------------------
    def get_activations_iterator(self) -> Iterator[ModelHookpointActivationsBatch]:
        base_iter = self._base.get_activations_iterator()

        try:
            first_batch = next(base_iter)
        except StopIteration:
            # The wrapped dataloader is empty: yield nothing.
            return

        # Shapes: B = batch, M = model, P = hook_Point, D = dim
        B, M, P, D = first_batch.activations_BMPD.shape  # type: ignore[assignment]
        capacity_B = self._buffer_batches * B  # total samples held at once

        # ------------------------------------------------------------------
        # Allocate buffer tensors (CPU‑resident) ---------------------------
        # ------------------------------------------------------------------
        buffer_BMPD = torch.empty(
            capacity_B, M, P, D,
            dtype=first_batch.activations_BMPD.dtype,
            device="cpu",
        )
        consumed_mask = torch.zeros(capacity_B, dtype=torch.bool, device="cpu")

        # ------------------------------------------------------------------
        # Helper: refill buffer in place -----------------------------------
        # ------------------------------------------------------------------
        def _fill(start_ptr: int) -> int:
            """Fill buffer from ``start_ptr`` onward.  Returns new *end* pointer."""
            ptr = start_ptr
            while ptr + B <= capacity_B:
                try:
                    batch = next(base_iter)
                except StopIteration:
                    break
                buffer_BMPD[ptr : ptr + B] = batch.activations_BMPD.to(
                    "cpu", non_blocking=True
                )
                consumed_mask[ptr : ptr + B] = False
                ptr += B
            return ptr

        # ------------------------------------------------------------------
        # Log memory footprint once ---------------------------------------
        # ------------------------------------------------------------------
        if not self._memory_logged:
            total_bytes = first_batch.activations_BMPD.element_size() * capacity_B * prod((M, P, D))
            logger.info(
                "Buffered activations reserve ≈%.2f MB of CPU RAM (%d samples × %s)",
                total_bytes / 1_048_576,
                capacity_B,
                first_batch.activations_BMPD.dtype,
            )
            self._memory_logged = True

        # ------------------------------------------------------------------
        # Prime buffer with initial data ----------------------------------
        # ------------------------------------------------------------------
        buffer_BMPD[0:B] = first_batch.activations_BMPD.to("cpu", non_blocking=True)
        valid_end = _fill(B)

        rng = torch.Generator(device="cpu")

        # ------------------------------------------------------------------
        # Main iteration wrapped in try/finally for cleanup ----------------
        # ------------------------------------------------------------------
        try:
            while True:
                unread_idx = (~consumed_mask[:valid_end]).nonzero(as_tuple=False).squeeze(1)
                if unread_idx.numel() < B:
                    # Buffer nearly empty – attempt full refresh.
                    valid_end = _fill(0)
                    unread_idx = (~consumed_mask[:valid_end]).nonzero(as_tuple=False).squeeze(1)
                    if unread_idx.numel() < B:
                        # Source exhausted – we're done.
                        return

                # Shuffle unread indices.
                unread_idx_perm = unread_idx[torch.randperm(unread_idx.numel(), generator=rng)]

                # Emit as many full batches as available in this permutation.
                batches_this_cycle = unread_idx_perm.numel() // B
                for i in range(batches_this_cycle):
                    sel = unread_idx_perm[i * B : (i + 1) * B]
                    consumed_mask[sel] = True
                    try:
                        yield ModelHookpointActivationsBatch(
                            buffer_BMPD[sel].to(self._device, non_blocking=True)
                        )
                    except RuntimeError as exc:
                        # Most likely an OOM on device transfer.
                        logger.exception("Device transfer failed – freeing CUDA cache and re‑raising.")
                        if self._device.type == "cuda":
                            torch.cuda.empty_cache()
                        raise exc

                    # Trigger in‑place refill if threshold reached.
                    consumed_batches = consumed_mask[:valid_end].sum().item() // B
                    if consumed_batches >= int(self._buffer_batches * self._refill_ratio):
                        unread_idx_before_refill = (~consumed_mask[:valid_end]).nonzero(as_tuple=False).squeeze(1)
                        n_unread = unread_idx_before_refill.numel()

                        if n_unread:
                            buffer_BMPD[0:n_unread] = buffer_BMPD[unread_idx_before_refill]
                            consumed_mask[0:n_unread] = False

                        valid_end = _fill(n_unread)
                        consumed_mask[n_unread:valid_end] = False
                        break  # rebuild permutation after refill
        finally:
            # ------------------------------------------------------------------
            # Defensive cleanup on *any* exit path ------------------------------
            # ------------------------------------------------------------------
            buffer_BMPD = None  # type: ignore[assignment]
            consumed_mask = None  # type: ignore[assignment]
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Scaling factors passthrough --------------------------------------
    # ------------------------------------------------------------------
    def get_scaling_factors(self) -> torch.Tensor:  # noqa: D401
        return self._base.get_scaling_factors()

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
        buffer_size: int | None = 400,
        buffer_refill_ratio: float | None = 0.25,
    ):
        self.cfg = cfg
        self.device = device

        # Wrap the dataloader with buffering if buffer_size is specified
        if buffer_size is not None and not isinstance(activations_dataloader, BufferedModelHookpointActivationsDataloader):
            if buffer_refill_ratio is None:
                buffer_refill_ratio = 0.25 # Default refill ratio

            print(f'Using buffered dataloader with buffer size {buffer_size} and refill ratio {buffer_refill_ratio}')
            self.activations_dataloader = BufferedModelHookpointActivationsDataloader(
                activations_dataloader,
                device=self.device,
                buffer_size=buffer_size,
                refill_ratio=buffer_refill_ratio,
            )
        else:
            self.activations_dataloader = activations_dataloader

        self.model = model
        self.wandb_run = wandb_run

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

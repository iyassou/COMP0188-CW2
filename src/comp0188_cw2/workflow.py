import dataclasses
import numpy as np
import torch
import torcheval.metrics
import tqdm
import wandb

from pathlib import Path
from typing import (
    Any,
    Callable,
)

@dataclasses.dataclass
class WandBConfig:
    model: str
    seed: int
    batch_size: int
    epochs: int
    training_criterion: str
    validation_criterion: str
    _optimiser: torch.optim.Optimizer
    _scheduler: torch.optim.lr_scheduler.LRScheduler
    project: str = "comp0188-cw2"
    group: str = None
    registry: str = "ucabis4-ucl-org/wandb-registry-model"

    @property
    def optimiser(self) -> dict:
        return {
            "name": self._optimiser.__class__.__name__,
            **{
                k: v for k,v in self._optimiser.param_groups[0].items()
                if k != "params"
            }
        }

    @property
    def scheduler(self) -> dict:
        if self._scheduler is None:
            return {
                "name": "constant",
                "lr": self._optimiser.param_groups[0]["lr"],
            }
        return {
            "name": self._scheduler.__class__.__name__,
            **self._scheduler.state_dict()
        }

    def wandb_init_kwargs(self) -> dict:
        """Returns the configuration dictionary passed to `wandb.init`"""
        config = dataclasses.asdict(self)
        config["optimiser"] = self.optimiser
        config["scheduler"] = self.scheduler
        config = {k: v for k, v in config.items() if v is not None}
        kwargs = {"project": self.project}
        if self.group is not None:
            kwargs["group"] = self.group
        exclude_keys = (
            "_optimiser",
            "_scheduler",
            "registry",
            "project",
            "group",
        )
        kwargs["config"] = wandb.helper.parse_config(config, exclude=exclude_keys)
        return kwargs

@dataclasses.dataclass
class WandBMetric:
    names: tuple[str, ...]
    factory: Callable[..., torcheval.metrics.Metric]
    factory_kwargs: dict[str, Any]
    metric: torcheval.metrics.Metric = dataclasses.field(init=False)
    multiplier: int = 1
    argmax: bool = False

    def __post_init__(self):
        self.metric = self.factory(**self.factory_kwargs)

    def update(self, input: torch.Tensor, target: torch.Tensor):
        if self.argmax:
            target = torch.argmax(target, dim=1)
        self.metric.update(input, target)

    def reset(self):
        self.metric.reset()

    def asdict(self) -> dict[str, float]:
        metric: torch.Tensor = self.multiplier * self.metric.compute()
        metric = metric.tolist()
        if not isinstance(metric, list):
            # torch.Tensor.tolist returns a number for scalars
            metric = [metric]
        return dict(zip(self.names, metric))

def load_wandb_api_key(file: Path=None):
    import os
    key = "WANDB_API_KEY"
    try:
        from google.colab import userdata # pyright: ignore[reportMissingImports]
        os.environ[key] = userdata.get(key)
    except ImportError:
        if file is None:
            raise ValueError("must specify path to wandb.json")
        import json
        with open(file, 'rb') as wfp:
            os.environ[key] = json.load(wfp)[key.lower()]
    except FileNotFoundError as e:
        e.add_note("Missing Colab secret and wandb.json file, could not log in")
        raise

def train_for_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: torch.nn.modules.loss._Loss,
    wandb_metrics: dict[slice, tuple[WandBMetric, ...]]) -> float:
    """Trains the model for a single epoch on the supplied DataLoader and
    returns the mean loss.
    
    Notes
    -----
    `scheduler` and `wandb_metrics` can be set to None."""
    if wandb_metrics is None:
        wandb_metrics = {}
    model.train()
    losses = []
    for X, y in tqdm.tqdm(dataloader, desc="Training"):
        optimiser.zero_grad()
        prediction = model(X)
        loss = criterion(prediction, y)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
        for cut, wbms in wandb_metrics.items():
            for wbm in wbms:
                wbm.update(prediction[:, cut], y[:, cut])
    if scheduler is not None:
        scheduler.step()
    return np.mean(losses)

def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    wandb_metrics: dict[slice, tuple[WandBMetric, ...]]) -> float:
    """Evaluates the model's performance on the supplied DataLoader, updates
    the given metrics, and returns the mean loss.
    
    Notes
    -----
    `wandb_metrics` can be set to None."""
    if wandb_metrics is None:
        wandb_metrics = {}
    model.eval()
    losses = []
    with torch.no_grad():
        for X, y in tqdm.tqdm(dataloader, desc="Validation"):
            prediction = model(X)
            loss = criterion(prediction, y)
            losses.append(loss.item())
            for cut, wbms in wandb_metrics.items():
                for wbm in wbms:
                    wbm.update(prediction[:, cut], y[:, cut])
    return np.mean(losses)

def training_loop(
    wandb_config: WandBConfig,
    model: torch.nn.Module,
    checkpoint_directory: Path,

    training_dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader,

    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,

    training_criterion: torch.nn.modules.loss._Loss,
    validation_criterion: torch.nn.modules.loss._Loss,
    
    training_metrics: dict[slice, tuple[WandBMetric, ...]],
    validation_metrics: dict[slice, tuple[WandBMetric, ...]]):
    """Big try-finally that terminates the WandB run regardless."""
    splits: tuple[str, str] = "train", "val"
    split_wandb_metrics = (
        tuple(x for y in training_metrics.values() for x in y),
        tuple(x for y in validation_metrics.values() for x in y),
    )
    run = wandb.init(**wandb_config.wandb_init_kwargs())
    checkpoint_directory /= run.name
    checkpoint_directory.mkdir(exist_ok=True, parents=True)
    try:
        torch.manual_seed(wandb_config.seed)
        epochs = wandb_config.epochs
        for epoch in range(1, epochs + 1):
            # Train for a single epoch.
            training_loss = train_for_one_epoch(
                model=model,
                dataloader=training_dataloader,
                optimiser=optimiser,
                scheduler=scheduler,
                criterion=training_criterion,
                wandb_metrics=training_metrics
            )
            # Validate model.
            validation_loss = evaluate(
                model=model,
                dataloader=validation_dataloader,
                criterion=validation_criterion,
                wandb_metrics=validation_metrics
            )
            # Print performance.
            print(
                f"Epoch: {epoch} / {epochs} | Training Loss: {training_loss:.4f}"
                f" | Validation Loss: {validation_loss:.4f}"
            )
            # Log data to WandB.
            wandb_log = dict()
            for split, loss, wandb_metrics in zip(splits, (training_loss, validation_loss), split_wandb_metrics):
                wandb_log[f"{split}/loss"] = loss
                wandb_log.update(
                    {
                        f"{split}/{k}": v
                        for wbm in wandb_metrics
                        for k, v in wbm.asdict().items()
                    }
                )
            run.log(wandb_log, step=epoch)
            # Reset metrics.
            for wandb_metrics in split_wandb_metrics:
                for wbm in wandb_metrics:
                    wbm.reset()
            # Upload checkpoint.
            checkpoint_filepath: Path = checkpoint_directory / f"{run.name}_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                    'train/loss': training_loss,
                },
                checkpoint_filepath
            )
            artifact = wandb.Artifact(name=wandb_config.model, type="checkpoint")
            artifact.add_file(checkpoint_filepath)
            logged_artifact = run.log_artifact(artifact)
            run.link_artifact(
                artifact=logged_artifact,
                target_path=f"{wandb_config.registry}/{wandb_config.project}"
            )
    finally:
        run.finish()

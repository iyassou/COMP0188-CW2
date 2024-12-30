import comp0188_cw2.workflow as W

import json
import pytest
import torch
import uuid

@pytest.fixture
def sample_model() -> torch.nn.Module:
    return torch.nn.Linear(1, 1)

@pytest.fixture
def sample_wandbconfig_data(sample_model) -> tuple[callable, dict]:
    MODEL = "test-model"
    SEED = 13
    BATCH_SIZE = 128
    EPOCHS = 2
    TRAINING_CRITERION = "wing and a prayer"
    VALIDATION_CRITERION = "if it swims like a duck"

    LEARNING_RATE = 5
    MOMENTUM = 0
    DAMPENING = 0
    WEIGHT_DECAY = 0
    NESTEROV = False
    MAXIMIZE = False
    FOREACH = None
    DIFFERENTIABLE = False
    FUSED = None
    OPTIMISER = torch.optim.SGD(
        sample_model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        dampening=DAMPENING,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV,
        maximize=MAXIMIZE,
        foreach=FOREACH,
        differentiable=DIFFERENTIABLE,
        fused=FUSED,
    )

    def make_config(
        scheduler,
        group,
        extra_config) -> W.WandBConfig:
        return W.WandBConfig(
            model=MODEL,
            seed=SEED,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            training_criterion=TRAINING_CRITERION,
            validation_criterion=VALIDATION_CRITERION,
            _optimiser=OPTIMISER,
            _scheduler=scheduler,
            group=group,
            extra_config=extra_config
        )
    
    EXPECTED_WANDB_INIT_KWARGS = {
        "project": W.WandBConfig.project,
        "config": {
            "model": MODEL,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "training_criterion": TRAINING_CRITERION,
            "validation_criterion": VALIDATION_CRITERION,
            "optimiser": {
                "name": "SGD",
                "lr": LEARNING_RATE,
                "momentum": MOMENTUM,
                "dampening": DAMPENING,
                "weight_decay": WEIGHT_DECAY,
                "nesterov": NESTEROV,
                "maximize": MAXIMIZE,
                "foreach": FOREACH,
                "differentiable": DIFFERENTIABLE,
                "fused": FUSED,
            },
        }
    }

    return make_config, EXPECTED_WANDB_INIT_KWARGS

@pytest.mark.parametrize(
    "scheduler, group, extra_config, warns",
    [
        (
            None,
            None,
            {},
            False,
        ),
        (
            torch.optim.lr_scheduler.StepLR,
            None,
            {},
            False,
        ),
        (
            torch.optim.lr_scheduler.StepLR,
            "mustard",
            {},
            False,
        ),
        (
            None,
            None,
            {str(uuid.uuid4()): "get into it yuh"},
            False,
        ),
        (
            None,
            None,
            {"optimiser": "pop out with a truck"},
            True,
        )
    ]
)
def test_wandbconfig_wandb_init_kwargs(
    scheduler,
    group,
    extra_config,
    warns,
    sample_wandbconfig_data):
    make_config, DEFAULT_WANDB_INIT_KWARGS = sample_wandbconfig_data
    config: W.WandBConfig = make_config(scheduler=None, group=group, extra_config=extra_config)
    if scheduler is None:
        scheduler_dict = {
            "name": "constant",
            "lr": config.optimiser["lr"],
        }
    else:
        scheduler = scheduler(config._optimiser, step_size=30, gamma=0.1)
        config._scheduler = scheduler
        scheduler_dict = {
            "name": scheduler.__class__.__name__,
            **scheduler.state_dict()
        }
    if group is None:
        group_dict = {}
    else:
        group_dict = {"group": group}

    if warns:
        with pytest.warns(UserWarning):
            actual = config.wandb_init_kwargs()
    else:
        actual = config.wandb_init_kwargs()
    expected = DEFAULT_WANDB_INIT_KWARGS.copy()
    
    if scheduler is not None:
        expected["config"]["optimiser"]["initial_lr"] = expected[
            "config"]["optimiser"]["lr"]
    expected.update(group_dict)
    expected["config"].update({"scheduler": scheduler_dict})
    expected["config"].update(extra_config)

    assert expected == actual

@pytest.mark.parametrize(
    "file, exception",
    [
        (None, ValueError),
        (str(uuid.uuid4()), None),
        (str(uuid.uuid4()), FileNotFoundError)
    ]
)
def test_load_wandb_api_key(file, exception, tmpdir):
    if file is not None:
        data = {"wandb_api_key": file}
        file = tmpdir / "wandb_api_key.json"
        if exception is FileNotFoundError:
            assert not file.exists()
        else:
            with open(file, "w") as handle:
                json.dump(data, handle)
    if exception is not None:
        with pytest.raises(exception):
            W.load_wandb_api_key(file)
    else:
        W.load_wandb_api_key(file)

# COMP0188-CW2

Python package for _COMP0188 – Deep Representations and Learning_ coursework 2.

![Sample trajectory](trajectory.gif)

## Structure

```
.
├── LICENSE
├── pyproject.toml
├── README.md
├── src/
│   └── comp0188_cw2/
│       ├── __init__.py
│       ├── dataset.py =========> Custom PyTorch Dataset
│       ├── datatypes.py -------> Dataclasses and types
│       ├── loss.py ============> Baseline model loss, β-VAE loss
│       ├── metrics.py ---------> Cosine similarity and mean absolute error implementations
│       ├── models.py ==========> Baseline model, VAE
│       ├── utils.py -----------> Cosine annealer implementation
│       ├── visualisation.py ===> uh-huh
│       └── workflow.py --------> Model training/evaluation with integrated W&B support
├── tests/
│   └── ...
└── trajectory.gif
```

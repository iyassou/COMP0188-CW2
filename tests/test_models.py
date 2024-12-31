import comp0188_cw2.models as M

import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float32
SAMPLE_INPUT = (
    torch.zeros((1, 2, 224, 224), dtype=DTYPE, device=DEVICE),
    torch.zeros((1, 15), dtype=DTYPE, device=DEVICE)
)

def test_vanilla_baseline_model_initialisation():
    model = M.VanillaBaselineModel(dtype=DTYPE, device=DEVICE)
    expected = """
VanillaBaselineModel(
  (joint_cnn_encoder): Sequential(
    (0): Conv2d(2, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU()
    (8): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (9): Flatten(start_dim=1, end_dim=-1)
    (10): LazyLinear(in_features=0, out_features=256, bias=True)
    (11): Linear(in_features=256, out_features=128, bias=True)
  )
  (dynamics_encoder): Sequential(
    (0): Linear(in_features=15, out_features=256, bias=True)
    (1): Linear(in_features=256, out_features=256, bias=True)
    (2): Linear(in_features=256, out_features=128, bias=True)
  )
  (fusion_layer): Sequential(
    (0): Linear(in_features=128, out_features=64, bias=True)
    (1): Linear(in_features=64, out_features=32, bias=True)
    (2): Linear(in_features=32, out_features=6, bias=True)
  )
)
    """.strip()
    assert repr(model) == expected
    _ = model(SAMPLE_INPUT) # NOTE: initialising lazy modules
    # NOTE: Number of parameters is a relatively good indicator of whether
    #       I've ported the baseline model correctly.
    #       Expected number of parameters retrieved from Josh's implementation.
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_parameters == 6_574_958

def test_variational_auto_encoder_initialisation():
    config = M.VariationalAutoEncoderConfiguration(latent_space_dimension=1)
    model = M.VariationalAutoEncoder(config=config, dtype=DTYPE, device=DEVICE)
    expected = f"""
VariationalAutoEncoder(
  (encoder): Sequential(
    (0): Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.01)
    (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (3): LeakyReLU(negative_slope=0.01)
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (5): LeakyReLU(negative_slope=0.01)
    (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (7): LeakyReLU(negative_slope=0.01)
    (8): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (9): LeakyReLU(negative_slope=0.01)
    (10): Flatten(start_dim=1, end_dim=-1)
  )
  (fc_mu): Linear(in_features=25088, out_features={config.latent_space_dimension}, bias=True)
  (fc_logvar): Linear(in_features=25088, out_features={config.latent_space_dimension}, bias=True)
  (decoder): Sequential(
    (0): Linear(in_features={config.latent_space_dimension}, out_features=25088, bias=True)
    (1): Unflatten(dim=1, unflattened_size=(512, 7, 7))
    (2): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (3): LeakyReLU(negative_slope=0.01)
    (4): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (5): LeakyReLU(negative_slope=0.01)
    (6): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (7): LeakyReLU(negative_slope=0.01)
    (8): ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (9): LeakyReLU(negative_slope=0.01)
    (10): ConvTranspose2d(32, 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (11): Sigmoid()
  )
)""".strip()
    assert repr(model) == expected

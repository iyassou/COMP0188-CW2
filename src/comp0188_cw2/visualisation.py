from comp0188_cw2.datatypes import (
    Chunk,
    GripperAction,
    Observation,
)
from comp0188_cw2.models import VariationalAutoEncoder

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import umap
import warnings

from pathlib import Path
from typing import Optional

def _textual_data_from_observation(obs: Observation, decimal: int) -> str:
    return "\n\n".join((
        rf"Actions $\Delta$: {obs.actions[:3].round(decimal)}",
        f"Gripper Action: {obs.gripper_action.name}",
        f"Gripper XYZ: {obs.ee_cartesian_pos_ob[:3].round(decimal)}",
        f"Gripper Quaternion: {obs.ee_cartesian_pos_ob[3:].round(decimal)}",
        rf"Gripper $\Delta$XYZ: {obs.ee_cartesian_vel_ob[:3].round(decimal)}",
        rf"Gripper $\Delta$(Roll, Pitch, Yaw): {obs.ee_cartesian_vel_ob[3:].round(decimal)}",
        f"Gripper Joints: {obs.joint_pos_ob.round(decimal)}",
    ))

def display_observation(obs: Observation, axes: matplotlib.axes.Axes):
    """Basic rendering of a single observation."""
    axes = axes.reshape(-1)
    for ax in axes:
        ax.set_axis_off()
    axes[0].set_title("front_cam_ob")
    axes[0].imshow(obs.front_cam_ob, cmap="gray")
    axes[1].set_title("mount_cam_ob")
    axes[1].imshow(obs.mount_cam_ob, cmap="gray")
    textual = _textual_data_from_observation(obs, decimal=2)
    axes[2].text(0.5, 0.5, textual, fontsize=20, ha="center", va="center")

def display_chunk(
        chunk: Chunk, fig: matplotlib.figure.Figure, axes: matplotlib.axes.Axes, file: Path,
        frames: Optional[int]=None):
    """Basic rendering of a chunk i.e. collection of sequential observations."""
    axes = axes.reshape(-1)
    for ax in axes:
        ax.set_axis_off()
    decimal = 2
    axes[0].set_title("front_cam_ob")
    axes[1].set_title("mount_cam_ob")

    obs = chunk[0]
    front_cam_display = axes[0].imshow(obs.front_cam_ob, cmap="gray", animated=True)
    mount_cam_display = axes[1].imshow(obs.mount_cam_ob, cmap="gray", animated=True)
    text_display = axes[2].text(
        0.5, 0.5, _textual_data_from_observation(obs, decimal),
        fontsize=16, ha="center", va="center",
    )
    fig.tight_layout()

    def update(frame: int):
        obs: Observation = chunk[frame]
        front_cam_display.set_data(obs.front_cam_ob)
        mount_cam_display.set_data(obs.mount_cam_ob)
        text_display.set_text(_textual_data_from_observation(obs, decimal))
        return front_cam_display, mount_cam_display, text_display
    
    ani = matplotlib.animation.FuncAnimation(
        fig, update, frames=frames or len(chunk), interval=75, blit=False, # text gets funky when blitting
    )
    ani.save(gif)

def latent_space_umap(
        vae: VariationalAutoEncoder,
        dataloader: torch.utils.data.DataLoader,
        ax: matplotlib.axes.Axes,
        umap_kwargs: Optional[dict]=None):
    """Visualises the VAE's latent space on a given dataloader using UMAP."""
    vae.eval()
    latents = []
    gripper_actions = []
    with torch.no_grad():
        for (images, _), actions in dataloader:
            mu, _ = vae.encode(images)
            latents.append(mu.cpu())
            gripper_actions.extend(
                map(
                    lambda x: GripperAction(x.item()).name,
                    torch.argmax(actions[:, 3:], dim=1)
                )
            )
    latents = torch.cat(latents, dim=0).numpy()
    reducer = umap.UMAP(**(umap_kwargs or {}))
    with warnings.catch_warnings(action="ignore"):
        embedding = reducer.fit_transform(latents)
    df = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'Gripper Action': gripper_actions})
    ax.grid(True)
    ax.set_aspect("equal")
    sns.scatterplot(data=df, x='x', y='y', hue='Gripper Action', ax=ax)

if __name__ == '__main__':
    from .datatypes import Chunk
    from pathlib import Path
    root = Path(__file__).parent.parent.parent.parent / "Cw2_upload"
    chunk = Chunk(
        root / "data" / "debug" / "train" / "train_0.h5"
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    gif = root / "figures" / "trajectory.gif"
    display_chunk(chunk, fig, axes, gif, frames=74)

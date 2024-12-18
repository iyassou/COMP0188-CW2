from .datatypes import (
    Observation,
    Chunk,
)

import matplotlib.animation
import matplotlib.pyplot as plt

from pathlib import Path

def _textual_data_from_observation(obs: Observation, decimal: int) -> str:
    return "\n\n".join((
        rf"Actions $\Delta$: {obs.actions[:3].round(decimal)}",
        f"Gripper Action: {obs.gripper_action.name}",
        f"Gripper XYZ: {obs.ee_cartesian_pos_ob[:3].round(decimal)}",
        f"Gripper Quaternion: {obs.ee_cartesian_pos_ob[3:].round(decimal)}",
        rf"Gripper $\Delta$XYZ: {obs.ee_cartesian_vel_ob[:3].round(decimal)}",
        rf"Gripper $\Delta$(Roll, Pitch, Yaw): {obs.ee_cartesian_vel_ob[3:].round(decimal)}",
        f"Gripper Joints: {obs.joint_pos_ob.round(decimal)}",
        f"Terminal? {'Yes' if obs.terminals else 'No'}",
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

def display_chunk(chunk: Trajectory, fig: matplotlib.figure.Figure, axes: matplotlib.axes.Axes, file: Path):
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
    
    _ = matplotlib.animation.FuncAnimation(
        fig, update, frames=len(chunk), interval=200, blit=False, # text gets funky when blitting
    )

    plt.show()

if __name__ == '__main__':
    from .datatypes import Trajectory
    from pathlib import Path
    chunk = Chunk(
        Path(__file__).parent.parent / "data" / "debug" / "train" / "train_0.h5" # frames [0, 74)
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    gif = Path(__file__).parent.parent / "test.gif"
    display_chunk(chunk, fig, axes, gif)

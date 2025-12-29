"""Render expert agent rollout to MP4."""
import argparse

import cv2
import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
from dm_control import suite
from shimmy import DmControlCompatibilityV0


def draw_reward_text(frame, reward, avg_reward):
    """Draw reward info on top-right of frame."""
    frame = frame.copy()
    text1 = f"Current reward: {reward:.3f}"
    text2 = f"Average reward: {avg_reward:.3f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    (w1, h1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
    (w2, h2), _ = cv2.getTextSize(text2, font, font_scale, thickness)

    padding = 4
    x1 = frame.shape[1] - max(w1, w2) - padding - 2
    y1 = padding + h1
    y2 = y1 + h2 + padding

    cv2.rectangle(frame, (x1 - 2, y1 - h1 - 2), (x1 + w1 + 2, y1 + 2), bg_color, -1)
    cv2.rectangle(frame, (x1 - 2, y2 - h2 - 2), (x1 + w2 + 2, y2 + 2), bg_color, -1)
    cv2.putText(frame, text1, (x1, y1), font, font_scale, color, thickness)
    cv2.putText(frame, text2, (x1, y2), font, font_scale, color, thickness)

    return frame


class ObsNormalizer:
    """Observation normalizer using running mean/std from checkpoint."""

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def normalize(self, x):
        normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -10, 10)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def make_env(domain, task, render_size=256):
    dm_env = suite.load(domain_name=domain, task_name=task)
    env = DmControlCompatibilityV0(
        dm_env,
        render_mode="rgb_array",
        render_kwargs=dict(height=render_size, width=render_size, camera_id=0),
    )
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.DtypeObservation(env, np.float32)
    env = gym.wrappers.ClipAction(env)
    return env


def load_expert(checkpoint_path, obs_dim, act_dim, hidden_dim=512, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Load actor
    actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
    actor_state = {
        k.replace("actor_mean.", ""): v
        for k, v in ckpt.items()
        if k.startswith("actor_mean.")
    }
    actor.net.load_state_dict(actor_state)
    actor.eval()

    # Load observation normalizer
    obs_normalizer = ObsNormalizer(
        mean=ckpt["obs_rms.mean"].cpu().numpy(),
        var=ckpt["obs_rms.var"].cpu().numpy(),
    )

    return actor, obs_normalizer


def main():
    parser = argparse.ArgumentParser(description="Render expert agent to MP4")
    parser.add_argument("--task", type=str, default="hopper-hop",
                        help="Task in format 'domain-task' (e.g., hopper-hop, walker-run, cheetah-run)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second")
    parser.add_argument("--size", type=int, default=256,
                        help="Render size in pixels")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: out/expert-{task}.mp4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    args = parser.parse_args()

    # Parse task
    parts = args.task.split("-")
    domain, task = parts[0], "-".join(parts[1:])

    checkpoint_path = f"scripts/data_collection/checkpoints/{args.task}-expert/checkpoint.pt"
    config_path = f"scripts/data_collection/checkpoints/{args.task}-expert/config.pt"

    # Load config and expert
    config = torch.load(config_path, map_location="cpu")
    hidden_dim = config.get("hidden_dim", 512)

    print(f"Task: {domain}-{task}")
    print(f"Checkpoint: {checkpoint_path}")

    # Create environment
    env = make_env(domain, task, render_size=args.size)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

    # Load expert
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor, obs_normalizer = load_expert(checkpoint_path, obs_dim, act_dim, hidden_dim, device)

    # Rollout
    max_frames = int(args.duration * args.fps)
    images = []
    total_reward = 0.0
    step_count = 0
    episode_count = 0

    obs, _ = env.reset(seed=args.seed)
    images.append(draw_reward_text(env.render(), 0.0, 0.0))

    while len(images) <= max_frames:
        obs_norm = obs_normalizer.normalize(obs)
        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action = actor(obs_tensor).squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1

        frame = draw_reward_text(env.render(), reward, total_reward / step_count)
        images.append(frame)

        if terminated or truncated:
            episode_count += 1
            print(f"Episode {episode_count} finished with reward: {total_reward:.2f}")
            total_reward = 0.0
            step_count = 0
            obs, _ = env.reset(seed=args.seed + episode_count)

    env.close()

    # Save video
    output_path = args.output or f"out/expert-{args.task}.mp4"
    print(f"Saving {len(images)} frames to {output_path}...")
    iio.imwrite(output_path, np.array(images), fps=args.fps)
    print("Done!")


if __name__ == "__main__":
    main()

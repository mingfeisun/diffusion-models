import argparse
import pathlib
from typing import Tuple

import imageio.v2 as imageio  # type: ignore[import-not-found]
import numpy as np


def make_procgen_env(env_name: str, start_level: int, num_levels: int, distribution_mode: str):
    from procgen import ProcgenEnv  # type: ignore[import-not-found]

    env = ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        render_mode="rgb_array",
    )
    return env, env_name, "procgen"


def _extract_frame(obs):
    if isinstance(obs, dict):
        frame = obs.get("rgb")
    else:
        frame = obs
    if frame is None:
        return None
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim == 4:
        frame = frame[0]
    return frame


def _safe_reset(env, seed: int | None):
    if seed is not None:
        np.random.seed(seed)
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _safe_step(env, action):
    out = env.step(action)
    obs, reward, done, info = out
    reward_value = float(np.asarray(reward).reshape(-1)[0])
    done_value = bool(np.asarray(done).reshape(-1)[0])
    return obs, reward_value, done_value, info


def _sample_action(env, deterministic_action: int | None):
    if deterministic_action is None:
        sampled = env.action_space.sample()
    else:
        sampled = int(deterministic_action)

    action_array = np.asarray(sampled, dtype=np.int32)
    if action_array.ndim == 0:
        action_array = action_array.reshape(1)
    return action_array


def _write_frames(output_path: pathlib.Path, frames, fps: int) -> pathlib.Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = output_path.with_suffix(".gif")
    imageio.mimsave(str(target_path), frames, fps=fps)
    return target_path


def rollout_and_record(
    env_name: str,
    output_path: pathlib.Path,
    num_steps: int,
    fps: int,
    start_level: int,
    num_levels: int,
    distribution_mode: str,
    seed: int | None,
    deterministic_action: int | None,
) -> Tuple[int, float]:
    env, env_id, backend_name = make_procgen_env(
        env_name=env_name,
        start_level=start_level,
        num_levels=num_levels,
        distribution_mode=distribution_mode,
    )

    obs = _safe_reset(env, seed=seed)

    frames = []
    first_frame = _extract_frame(obs)
    if first_frame is not None:
        frames.append(first_frame)

    total_reward = 0.0
    steps = 0
    done = False

    while steps < num_steps and not done:
        action = _sample_action(env, deterministic_action)
        obs, reward, done, _ = _safe_step(env, action)
        total_reward += reward
        steps += 1

        frame = _extract_frame(obs)
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise RuntimeError("No frames were rendered. Check render_mode / env setup.")

    saved_path = _write_frames(output_path, frames, fps)

    env.close()
    print(f"Saved rollout video: {saved_path}")
    print(f"Env id: {env_id} (backend: {backend_name})")
    print(f"Steps: {steps} | Episode reward: {total_reward:.3f}")

    return steps, total_reward


def parse_args():
    parser = argparse.ArgumentParser(description="Record a rollout from a Procgen env.")
    parser.add_argument("--env", type=str, default="coinrun", help="Procgen game name, e.g. coinrun")
    parser.add_argument("--out", type=str, default="rollouts/procgen_rollout.gif", help="Output video path")
    parser.add_argument("--steps", type=int, default=500, help="Maximum rollout steps")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--start-level", type=int, default=0, help="Procgen start_level")
    parser.add_argument("--num-levels", type=int, default=0, help="Procgen num_levels (0 = unlimited)")
    parser.add_argument(
        "--distribution-mode",
        type=str,
        default="easy",
        choices=["easy", "hard", "exploration", "memory", "extreme"],
        help="Procgen distribution mode",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument(
        "--action",
        type=int,
        default=None,
        help="Optional fixed action to take each step (default: random action)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rollout_and_record(
        env_name=args.env,
        output_path=pathlib.Path(args.out),
        num_steps=args.steps,
        fps=args.fps,
        start_level=args.start_level,
        num_levels=args.num_levels,
        distribution_mode=args.distribution_mode,
        seed=args.seed,
        deterministic_action=args.action,
    )
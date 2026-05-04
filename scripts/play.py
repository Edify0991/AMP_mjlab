"""Script to play RL agent with RSL-RL."""

import os
import inspect
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  export_onnx: bool = True
  """Export loaded trained policy to ONNX under current run directory/export."""
  record_policy_io: bool = False
  """Record per-step policy observations and actions to an npz file."""
  policy_io_output: str | None = None
  """Output path for policy IO recording. Defaults under the checkpoint run directory."""
  policy_io_max_steps: int | None = None
  """Maximum number of policy calls to record. None records until play exits."""
  zero_commands: bool = False
  """Set velocity commands to zero during play for deterministic policy IO recording."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  def zero_velocity_commands(env_cfg: Any) -> None:
    """Clamp velocity command ranges to zero for deterministic play rollouts."""
    twist_cmd = env_cfg.commands.get("twist") if env_cfg.commands is not None else None
    if twist_cmd is None:
      print("[WARN]: --zero-commands was set, but command 'twist' was not found.")
      return
    if not isinstance(twist_cmd, UniformVelocityCommandCfg):
      print(
        "[WARN]: --zero-commands currently supports UniformVelocityCommandCfg; "
        f"got {type(twist_cmd).__name__}."
      )
      return

    zero = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_x = zero
    twist_cmd.ranges.lin_vel_y = zero
    twist_cmd.ranges.ang_vel_z = zero
    if twist_cmd.ranges.heading is not None:
      twist_cmd.ranges.heading = zero
    twist_cmd.rel_standing_envs = 1.0
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.init_velocity_prob = 0.0
    print("[INFO]: Velocity command ranges are fixed to zero for play.")

  def onnx_export_kwargs_single_file() -> dict:
    """Build kwargs that request single-file ONNX export across torch versions."""
    try:
      params = inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):
      return {}

    if "external_data" in params:
      return {"external_data": False}
    if "use_external_data_format" in params:
      return {"use_external_data_format": False}
    return {}

  def inline_external_onnx_data(onnx_path: Path) -> None:
    """Merge external tensor data back into a single ONNX file if needed."""
    data_path = Path(str(onnx_path) + ".data")
    if not data_path.exists():
      return

    try:
      import onnx

      model = onnx.load(str(onnx_path), load_external_data=True)
      onnx.save_model(model, str(onnx_path), save_as_external_data=False)
      if data_path.exists():
        data_path.unlink()
      print(f"[INFO]: Inlined external ONNX data into single file: {onnx_path}")
    except Exception as exc:
      print(f"[WARN]: Failed to inline ONNX external data for {onnx_path}: {exc}")

  class _OnnxPolicyWrapper(torch.nn.Module):
    """Expose act_inference as forward and optionally include obs normalizer."""

    def __init__(self, actor_critic: torch.nn.Module, obs_normalizer: Any = None):
      super().__init__()
      self.actor_critic = actor_critic
      self.obs_normalizer = obs_normalizer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
      if self.obs_normalizer is not None:
        obs = self.obs_normalizer(obs)
      return self.actor_critic.act_inference(obs)

  class PolicyIORecorder:
    """Wrap a policy and record the exact play-loop input/output tensors."""

    def __init__(
      self,
      policy: Any,
      output_path: Path,
      runner: Any | None = None,
      clip_actions: float | None = None,
      max_steps: int | None = None,
    ):
      self.policy = policy
      self.output_path = output_path
      self.runner = runner
      self.clip_actions = clip_actions
      self.max_steps = max_steps
      self.actor_obs_raw: list[np.ndarray] = []
      self.actor_obs_normalized: list[np.ndarray] = []
      self.actions: list[np.ndarray] = []
      self.actions_clipped: list[np.ndarray] = []
      self.num_calls = 0

    @staticmethod
    def _actor_obs(obs: Any) -> torch.Tensor:
      if hasattr(obs, "keys"):
        actor_key = "actor" if "actor" in obs.keys() else "policy"
        return obs[actor_key]
      return obs

    def _normalized_actor_obs(self, actor_obs: torch.Tensor) -> torch.Tensor | None:
      if self.runner is None:
        return None
      empirical_normalization = bool(
        getattr(self.runner, "empirical_normalization", False)
        or getattr(self.runner, "cfg", {}).get("empirical_normalization", False)
      )
      obs_normalizer = getattr(self.runner, "obs_normalizer", None)
      if not empirical_normalization or obs_normalizer is None:
        return None
      return obs_normalizer(actor_obs)

    def __call__(self, obs: Any) -> torch.Tensor:
      actor_obs = self._actor_obs(obs)
      normalized_actor_obs = self._normalized_actor_obs(actor_obs)
      actions = self.policy(obs)

      should_record = self.max_steps is None or self.num_calls < self.max_steps
      if should_record:
        self.actor_obs_raw.append(actor_obs.detach().cpu().numpy())
        if normalized_actor_obs is not None:
          self.actor_obs_normalized.append(normalized_actor_obs.detach().cpu().numpy())
        self.actions.append(actions.detach().cpu().numpy())
        if self.clip_actions is not None:
          actions_clipped = torch.clamp(actions, -self.clip_actions, self.clip_actions)
          self.actions_clipped.append(actions_clipped.detach().cpu().numpy())
      self.num_calls += 1
      return actions

    def save(self) -> None:
      if not self.actions:
        print("[WARN]: Policy IO recording was enabled, but no policy calls were recorded.")
        return

      self.output_path.parent.mkdir(parents=True, exist_ok=True)
      data: dict[str, np.ndarray] = {
        "step": np.arange(len(self.actions), dtype=np.int64),
        "actor_obs_raw": np.stack(self.actor_obs_raw, axis=0),
        "actions": np.stack(self.actions, axis=0),
      }
      if self.actor_obs_normalized:
        data["actor_obs_normalized"] = np.stack(self.actor_obs_normalized, axis=0)
      if self.actions_clipped:
        data["actions_clipped"] = np.stack(self.actions_clipped, axis=0)
      np.savez_compressed(self.output_path, **data)
      print(f"[INFO]: Saved policy IO recording to: {self.output_path}")

  def export_runner_policy_to_onnx(runner: Any, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer runner-provided exporters to keep behavior consistent with training.
    if hasattr(runner, "export_policy_to_onnx"):
      runner.export_policy_to_onnx(str(output_path.parent), output_path.name)
      inline_external_onnx_data(output_path)
      return
    if hasattr(runner, "_export_policy_to_onnx"):
      runner._export_policy_to_onnx(str(output_path.parent), output_path.name)
      inline_external_onnx_data(output_path)
      return

    # Fallback exporter for runners without explicit ONNX export helper.
    policy = runner.alg.policy
    obs_normalizer = None
    if getattr(runner, "empirical_normalization", False) and hasattr(
      runner, "obs_normalizer"
    ):
      obs_normalizer = runner.obs_normalizer
      obs_normalizer.to("cpu")
      obs_normalizer.eval()

    wrapper = _OnnxPolicyWrapper(policy, obs_normalizer)
    wrapper.to("cpu")
    wrapper.eval()
    num_obs = policy.actor[0].in_features
    dummy_input = torch.zeros(1, num_obs)
    torch.onnx.export(
      wrapper,
      dummy_input,
      str(output_path),
      export_params=True,
      opset_version=18,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
      **onnx_export_kwargs_single_file(),
    )
    inline_external_onnx_data(output_path)

    runner_device = getattr(runner, "device", None)
    if runner_device is not None:
      policy.to(runner_device)
      if obs_normalizer is not None:
        obs_normalizer.to(runner_device)

  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  if cfg.zero_commands:
    zero_velocity_commands(env_cfg)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner: Any | None = None
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    if cfg.export_onnx:
      safe_task_name = task_id.replace("/", "_").replace(":", "_")
      checkpoint_stem = resume_path.stem if resume_path is not None else "policy"
      export_root = log_dir if log_dir is not None else Path("logs")
      onnx_path = (export_root / "export" / f"{safe_task_name}_{checkpoint_stem}.onnx").resolve()
      try:
        export_runner_policy_to_onnx(runner, onnx_path)
        print(f"[INFO]: Exported ONNX policy to: {onnx_path}")
      except Exception as exc:
        print(f"[WARN]: Failed to export ONNX policy: {exc}")
  if DUMMY_MODE and cfg.export_onnx:
    print("[WARN]: ONNX export is only available for trained agents.")

  policy_io_recorder: PolicyIORecorder | None = None
  if cfg.record_policy_io:
    if cfg.policy_io_output is not None:
      policy_io_path = Path(cfg.policy_io_output).expanduser()
    else:
      safe_task_name = task_id.replace("/", "_").replace(":", "_")
      checkpoint_stem = resume_path.stem if resume_path is not None else cfg.agent
      output_root = log_dir if log_dir is not None else Path("logs") / "play_policy_io"
      policy_io_path = output_root / "policy_io" / f"{safe_task_name}_{checkpoint_stem}.npz"
    if policy_io_path.suffix != ".npz":
      policy_io_path = policy_io_path.with_suffix(".npz")
    policy_io_path = policy_io_path.resolve()
    policy_io_recorder = PolicyIORecorder(
      policy,
      policy_io_path,
      runner=runner,
      clip_actions=agent_cfg.clip_actions,
      max_steps=cfg.policy_io_max_steps,
    )
    policy = policy_io_recorder
    limit_msg = (
      "unlimited" if cfg.policy_io_max_steps is None else str(cfg.policy_io_max_steps)
    )
    print(f"[INFO]: Recording policy IO to: {policy_io_path} (max_steps={limit_msg})")

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  try:
    if resolved_viewer == "native":
      NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
      ViserPlayViewer(env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    if policy_io_recorder is not None:
      policy_io_recorder.save()
    env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()

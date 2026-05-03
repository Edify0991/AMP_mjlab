"""Jingchu01 AMP Locomotion environment configurations."""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import (
  JINGCHU01_ACTION_SCALE,
  JINGCHU01_AMP_BODY_NAMES,
  JINGCHU01_ANCHOR_BODY_NAME,
  JINGCHU01_JOINT_NAMES,
  get_jingchu01_robot_cfg,
)
from src.assets.robots.jingchu01.jingchu01_constants import (
  JINGCHU01_FEET_GEOM_NAMES,
  JINGCHU01_FEET_SITE_NAMES,
)
from src.tasks.amp_loco import mdp
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg


def jingchu01_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Jingchu01 rough terrain velocity configuration."""
  cfg = make_amp_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 128
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 64

  cfg.scene.entities = {"robot": get_jingchu01_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = JINGCHU01_ANCHOR_BODY_NAME

  feet_pattern = r"^(right_ankle_roll|left_ankle_roll)$"
  foot_geom_names = JINGCHU01_FEET_GEOM_NAMES
  body_names = JINGCHU01_AMP_BODY_NAMES
  anchor_name = JINGCHU01_ANCHOR_BODY_NAME
  root_name = JINGCHU01_ANCHOR_BODY_NAME

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=feet_pattern,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=JINGCHU01_ANCHOR_BODY_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=JINGCHU01_ANCHOR_BODY_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = JINGCHU01_ACTION_SCALE

  cfg.viewer.body_name = JINGCHU01_ANCHOR_BODY_NAME

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["actor"].terms["base_ang_vel"].params[
    "sensor_name"
  ] = "robot/angular-velocity"
  cfg.observations["critic"].terms["base_ang_vel"].params[
    "sensor_name"
  ] = "robot/angular-velocity"
  cfg.observations["critic"].terms["base_lin_vel"].params[
    "sensor_name"
  ] = "robot/linear-velocity"

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = (JINGCHU01_ANCHOR_BODY_NAME,)
  cfg.events["reset_from_motion"].params["asset_cfg"].joint_names = JINGCHU01_JOINT_NAMES

  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 250

  _motion_base = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "jingchu01", "amp"
  )
  _motion_dir = os.path.abspath(os.path.join(_motion_base, "WalkandRun"))
  _recovery_dir = os.path.abspath(os.path.join(_motion_base, "Recovery"))

  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = _recovery_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["foot_slip"].params["asset_cfg"] = SceneEntityCfg(
    "robot", site_names=JINGCHU01_FEET_SITE_NAMES
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)

  cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

  cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = body_names

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0

  return cfg


def jingchu01_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Jingchu01 flat terrain velocity configuration."""
  cfg = jingchu01_amp_rough_env_cfg(play=play)

  cfg.sim.njmax = 1500
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 3.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-3.14 / 2, 3.14 / 2)

  return cfg

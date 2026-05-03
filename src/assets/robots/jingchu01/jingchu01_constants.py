"""Jingchu01 robot constants and mjlab entity config."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and motion metadata.
##

JINGCHU01_XML = Path(
  "/home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/jingchu01.xml"
)
assert JINGCHU01_XML.exists()

JINGCHU01_JOINT_NAMES = (
  "right_hip_roll",
  "right_hip_yaw",
  "right_hip_pitch",
  "right_knee_pitch",
  "right_ankle_pitch",
  "right_ankle_roll",
  "left_hip_roll",
  "left_hip_yaw",
  "left_hip_pitch",
  "left_knee_pitch",
  "left_ankle_pitch",
  "left_ankle_roll",
  "waist_roll",
  "waist_yaw",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow_pitch",
  "right_elbow_yaw",
  "right_wrist_pitch",
  "right_wrist_roll",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow_pitch",
  "left_elbow_yaw",
  "left_wrist_pitch",
  "left_wrist_roll",
)

JINGCHU01_ANCHOR_BODY_NAME = "Body"

JINGCHU01_AMP_BODY_NAMES = (
  "Body",
  "right_hip_roll",
  "right_hip_yaw",
  "right_hip_pitch",
  "right_knee_pitch",
  "right_ankle_pitch",
  "right_ankle_roll",
  "left_hip_roll",
  "left_hip_yaw",
  "left_hip_pitch",
  "left_knee_pitch",
  "left_ankle_pitch",
  "left_ankle_roll",
  "waist_roll",
  "waist_yaw",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow_pitch",
  "right_elbow_yaw",
  "right_wrist_pitch",
  "right_wrist_roll",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow_pitch",
  "left_elbow_yaw",
  "left_wrist_pitch",
  "left_wrist_roll",
)

JINGCHU01_FEET_BODY_NAMES = ("right_ankle_roll", "left_ankle_roll")
JINGCHU01_FEET_SITE_NAMES = ("right_foot_site", "left_foot_site")
JINGCHU01_COLLISION_GEOM_NAMES = (r".*_collision_[0-9]+$",)
JINGCHU01_FEET_GEOM_NAMES = (r"^(right|left)_ankle_roll_collision_[0-9]+$",)


def get_spec() -> mujoco.MjSpec:
  """Load the Jingchu01 MJCF as a MuJoCo spec."""
  spec = mujoco.MjSpec.from_file(str(JINGCHU01_XML))
  spec.meshdir = spec.meshdir.rstrip("/")
  assets: dict[str, bytes] = {}
  update_assets(assets, JINGCHU01_XML.parent / spec.meshdir, spec.meshdir)
  spec.assets = assets

  root_body = next(body for body in spec.bodies if body.name == JINGCHU01_ANCHOR_BODY_NAME)
  freejoints = [joint for joint in root_body.joints if joint.type == mujoco.mjtJoint.mjJNT_FREE]
  if freejoints:
    if not freejoints[0].name:
      freejoints[0].name = "root_freejoint"
  else:
    freejoint = root_body.add_freejoint()
    freejoint.name = "root_freejoint"

  return spec


##
# Actuator config.
##

# Values mirror the lower-body parameters in beyondmimic's jingchu01.py.
NATURAL_FREQ = 6.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 1.1

ARMATURE_A10020_P224 = 0.2773762228
ARMATURE_A10020_P112 = 0.07001770124
ARMATURE_A8112_P118 = 0.0485578476
ARMATURE_A6408_P225 = 0.03960461065
ARMATURE_A4310_P236 = 0.02422284137
ARMATURE_KNEE = 300.0 / (NATURAL_FREQ**2)

STIFFNESS_A10020_P224 = ARMATURE_A10020_P224 * NATURAL_FREQ**2
STIFFNESS_A10020_P112 = ARMATURE_A10020_P112 * NATURAL_FREQ**2
STIFFNESS_A8112_P118 = ARMATURE_A8112_P118 * NATURAL_FREQ**2
STIFFNESS_A6408_P225 = ARMATURE_A6408_P225 * NATURAL_FREQ**2
STIFFNESS_A4310_P236 = ARMATURE_A4310_P236 * NATURAL_FREQ**2
STIFFNESS_KNEE = 300.0

DAMPING_A10020_P224 = 2.0 * DAMPING_RATIO * ARMATURE_A10020_P224 * NATURAL_FREQ
DAMPING_A10020_P112 = 2.0 * DAMPING_RATIO * ARMATURE_A10020_P112 * NATURAL_FREQ
DAMPING_A8112_P118 = 2.0 * DAMPING_RATIO * ARMATURE_A8112_P118 * NATURAL_FREQ
DAMPING_A6408_P225 = 2.0 * DAMPING_RATIO * ARMATURE_A6408_P225 * NATURAL_FREQ
DAMPING_A4310_P236 = 2.0 * DAMPING_RATIO * ARMATURE_A4310_P236 * NATURAL_FREQ
DAMPING_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_KNEE * NATURAL_FREQ

UPPER_BODY_EFFORT_LIMIT = 25.0
WRIST_EFFORT_LIMIT = UPPER_BODY_EFFORT_LIMIT
UPPER_BODY_STIFFNESS = STIFFNESS_A6408_P225
UPPER_BODY_DAMPING = DAMPING_A6408_P225
UPPER_BODY_ARMATURE = ARMATURE_A6408_P225

HIP_ROLL = r".*_hip_roll$"
HIP_YAW = r".*_hip_yaw$"
HIP_PITCH = r".*_hip_pitch$"
KNEE_PITCH = r".*_knee_pitch$"
ANKLE_PITCH = r".*_ankle_pitch$"
ANKLE_ROLL = r".*_ankle_roll$"
WAIST_ROLL = r"waist_roll$"
WAIST_YAW = r"waist_yaw$"
SHOULDER_PITCH = r".*_shoulder_pitch$"
SHOULDER_ROLL = r".*_shoulder_roll$"
SHOULDER_YAW = r".*_shoulder_yaw$"
ELBOW_PITCH = r".*_elbow_pitch$"
ELBOW_YAW = r".*_elbow_yaw$"
WRIST_PITCH = r".*_wrist_pitch$"
WRIST_ROLL = r".*_wrist_roll$"
SHOULDER_ELBOW = r".*_(shoulder|elbow)_(pitch|roll|yaw)$"
WRIST = r".*_wrist_(pitch|roll)$"

JINGCHU01_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(HIP_ROLL,),
  stiffness=STIFFNESS_A10020_P224,
  damping=DAMPING_A10020_P224,
  effort_limit=330.0,
  armature=ARMATURE_A10020_P224,
)

JINGCHU01_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(HIP_YAW,),
  stiffness=STIFFNESS_A10020_P112,
  damping=DAMPING_A10020_P112,
  effort_limit=150.0,
  armature=ARMATURE_A10020_P112,
)

JINGCHU01_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(HIP_PITCH,),
  stiffness=STIFFNESS_A10020_P224,
  damping=DAMPING_A10020_P224,
  effort_limit=330.0,
  armature=ARMATURE_A10020_P224,
)

JINGCHU01_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(KNEE_PITCH,),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=306.0,
  armature=ARMATURE_KNEE,
)

JINGCHU01_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(ANKLE_PITCH,),
  stiffness=2.0 * STIFFNESS_A8112_P118,
  damping=2.0 * DAMPING_A8112_P118,
  effort_limit=90.0,
  armature=2.0 * ARMATURE_A8112_P118,
)

JINGCHU01_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(ANKLE_ROLL,),
  stiffness=2.0 * STIFFNESS_A8112_P118,
  damping=2.0 * DAMPING_A8112_P118,
  effort_limit=90.0,
  armature=2.0 * ARMATURE_A8112_P118,
)

JINGCHU01_ACTUATOR_WAIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(WAIST_ROLL,),
  stiffness=STIFFNESS_A10020_P224,
  damping=DAMPING_A10020_P224,
  effort_limit=330.0,
  armature=ARMATURE_A10020_P224,
)

JINGCHU01_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(WAIST_YAW,),
  stiffness=STIFFNESS_A10020_P112,
  damping=DAMPING_A10020_P112,
  effort_limit=150.0,
  armature=ARMATURE_A10020_P112,
)

JINGCHU01_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=(SHOULDER_PITCH, SHOULDER_ROLL),
  stiffness=STIFFNESS_A8112_P118,
  damping=DAMPING_A8112_P118,
  effort_limit=90.0,
  armature=ARMATURE_A8112_P118,
)

JINGCHU01_ACTUATOR_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(SHOULDER_YAW, ELBOW_PITCH),
  stiffness=STIFFNESS_A6408_P225,
  damping=DAMPING_A6408_P225,
  effort_limit=60.0,
  armature=ARMATURE_A6408_P225,
)

JINGCHU01_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(ELBOW_YAW, WRIST_PITCH, WRIST_ROLL),
  stiffness=STIFFNESS_A4310_P236,
  damping=DAMPING_A4310_P236,
  effort_limit=36.0,
  armature=ARMATURE_A4310_P236,
)

##
# Initial state and collision config.
##

STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.92),
  joint_pos={
    HIP_PITCH: -0.24,
    KNEE_PITCH: 0.48,
    ANKLE_PITCH: -0.24,
    ELBOW_PITCH: 0.3,
    SHOULDER_ROLL: 0.2,
    SHOULDER_PITCH: 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=JINGCHU01_COLLISION_GEOM_NAMES,
  condim={
    JINGCHU01_FEET_GEOM_NAMES[0]: 3,
    JINGCHU01_COLLISION_GEOM_NAMES[0]: 1,
  },
  priority={JINGCHU01_FEET_GEOM_NAMES[0]: 1},
  friction={JINGCHU01_FEET_GEOM_NAMES[0]: (0.6,)},
)


# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=JINGCHU01_FEET_GEOM_NAMES,
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


JINGCHU01_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    JINGCHU01_ACTUATOR_HIP_ROLL,
    JINGCHU01_ACTUATOR_HIP_YAW,
    JINGCHU01_ACTUATOR_HIP_PITCH,
    JINGCHU01_ACTUATOR_KNEE,
    JINGCHU01_ACTUATOR_ANKLE_PITCH,
    JINGCHU01_ACTUATOR_ANKLE_ROLL,
    JINGCHU01_ACTUATOR_WAIST_ROLL,
    JINGCHU01_ACTUATOR_WAIST_YAW,
    JINGCHU01_ACTUATOR_SHOULDER,
    JINGCHU01_ACTUATOR_ELBOW,
    JINGCHU01_ACTUATOR_WRIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_jingchu01_robot_cfg() -> EntityCfg:
  """Get a fresh Jingchu01 robot configuration."""
  return EntityCfg(
    init_state=STANDING_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=JINGCHU01_ARTICULATION,
  )


JINGCHU01_ACTION_SCALE: dict[str, float] = {}
for actuator in JINGCHU01_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  assert actuator.effort_limit is not None
  for name in actuator.target_names_expr:
    JINGCHU01_ACTION_SCALE[name] = 0.25 * actuator.effort_limit / actuator.stiffness

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_jingchu01_robot_cfg())

  viewer.launch(robot.spec.compile())

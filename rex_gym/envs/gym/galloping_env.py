"""This file implements the gym environment of Rex.

"""
import collections
import math
from gym import spaces
import numpy as np

from .. import rex_gym_env

# Radiant
from ...model.rex import Rex

INIT_SHOULDER_POS = 0.0
# -math.pi / 5
INIT_LEG_POS = -0.658319
# math.pi / 3
INIT_FOOT_POS = 1.0472
NUM_LEGS = 4
NUM_MOTORS = 3 * NUM_LEGS

RexPose = collections.namedtuple(
    "RexPose", "shoulder_angle_1, leg_angle_1, foot_angle_1, "
               "shoulder_angle_2, leg_angle_2, foot_angle_2, shoulder_angle_3, leg_angle_3, foot_angle_3,"
               "shoulder_angle_4, leg_angle_4, foot_angle_4")


class RexReactiveEnv(rex_gym_env.RexGymEnv):
    """The gym environment for Rex.

  It simulates the locomotion of Rex, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far Rex walks in 1000 steps and penalizes the energy
  expenditure.

  """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 166}

    def __init__(self,
                 urdf_version=None,
                 energy_weight=0.005,
                 control_time_step=0.006,
                 action_repeat=6,
                 control_latency=0.02,
                 pd_latency=0.003,
                 on_rack=False,
                 motor_kp=1.0,
                 motor_kd=0.015,
                 remove_default_joint_damping=True,
                 render=False,
                 num_steps_to_log=1000,
                 accurate_motor_model_enabled=True,
                 use_angle_in_observation=True,
                 hard_reset=False,
                 env_randomizer=None,
                 log_path=None):
        """Initialize Rex trotting gym environment.

    Args:
      urdf_version: [DEFAULT_URDF_VERSION] are allowable
        versions. If None, DEFAULT_URDF_VERSION is used. Refer to
        rex_gym_env for more details.
      energy_weight: The weight of the energy term in the reward function. Refer
        to rex_gym_env for more details.
      control_time_step: The time step between two successive control signals.
      action_repeat: The number of simulation steps that an action is repeated.
      control_latency: The latency between get_observation() and the actual
        observation. See rex.py for more details.
      pd_latency: The latency used to get motor angles/velocities used to
        compute PD controllers. See rex.py for more details.
      on_rack: Whether to place Rex on rack. This is only used to debug
        the walking gait. In this mode, Rex's base is hung midair so
        that its walking gait is clearer to visualize.
      motor_kp: The P gain of the motor.
      motor_kd: The D gain of the motor.
      remove_default_joint_damping: Whether to remove the default joint damping.
      render: Whether to render the simulation.
      num_steps_to_log: The max number of control steps in one episode. If the
        number of steps is over num_steps_to_log, the environment will still
        be running, but only first num_steps_to_log will be recorded in logging.
      accurate_motor_model_enabled: Whether to use the accurate motor model from
        system identification. Refer to rex_gym_env for more details.
      use_angle_in_observation: Whether to include motor angles in observation.
      hard_reset: Whether hard reset (swipe out everything and reload) the
        simulation. If it is false, Rex is set to the default pose
        and moved to the origin.
      env_randomizer: An instance (or a list) of EnvRanzomier(s) that can
        randomize the environment during when env.reset() is called and add
        perturbations when env.step() is called.
      log_path: The path to write out logs. For the details of logging, refer to
        rex_logging.proto.
    """
        self._use_angle_in_observation = use_angle_in_observation

        super(RexReactiveEnv,
              self).__init__(urdf_version=urdf_version,
                             energy_weight=energy_weight,
                             accurate_motor_model_enabled=accurate_motor_model_enabled,
                             motor_overheat_protection=True,
                             motor_kp=motor_kp,
                             motor_kd=motor_kd,
                             remove_default_joint_damping=remove_default_joint_damping,
                             control_latency=control_latency,
                             pd_latency=pd_latency,
                             on_rack=on_rack,
                             render=render,
                             hard_reset=hard_reset,
                             num_steps_to_log=num_steps_to_log,
                             env_randomizer=env_randomizer,
                             log_path=log_path,
                             control_time_step=control_time_step,
                             action_repeat=action_repeat)

        action_dim = 4
        action_low = np.array([-0.6] * action_dim)
        action_high = -action_low
        self.action_space = spaces.Box(action_low, action_high)
        self._cam_dist = 1.0
        self._cam_yaw = 30
        self._cam_pitch = -30
        self.init_leg = INIT_LEG_POS
        self.init_foot = INIT_FOOT_POS

    def reset(self):
        super(RexReactiveEnv, self).reset(initial_motor_angles=Rex.INIT_POSES['stand_high'],
                                          reset_duration=0.5)
        return self._get_observation()

    def _convert_from_leg_model(self, leg_pose):
        motor_pose = np.zeros(NUM_MOTORS)
        for i in range(NUM_LEGS):
            motor_pose[int(3 * i)] = 0
            if i == 0 or i == 1:
                leg_action = self.init_leg + leg_pose[0]
                motor_pose[int(3 * i + 1)] = max(min(leg_action, self.init_leg + 0.60), self.init_leg - 0.60)
                foot_pose = self.init_foot + leg_pose[1]
                motor_pose[int(3 * i + 2)] = max(min(foot_pose, self.init_foot + 0.60), self.init_foot - 0.60)
            else:
                leg_action = self.init_leg + leg_pose[2]
                motor_pose[int(3 * i + 1)] = max(min(leg_action, self.init_leg + 0.60), self.init_leg - 0.60)
                foot_pose = self.init_foot + leg_pose[3]
                motor_pose[int(3 * i + 2)] = max(min(foot_pose, self.init_foot + 0.60), self.init_foot - 0.60)

        return motor_pose

    # REX_LEG_MODELS = [
    #   JUMP  ------>
    #   motor_pose[int(3 * i)] = 0
    #   leg_action = self.init_leg + leg_pose[int(3 * i)]
    #   motor_pose[int(3 * i + 1)] = max(min(leg_action, self.init_leg + 0.45), self.init_leg - 0.78)
    #   foot_pose = self.init_foot + leg_pose[int(3 * i)]
    #   motor_pose[int(3 * i + 2)] = max(min(foot_pose, self.init_foot + 0.63), self.init_foot - 0.63)
    #   -----------------------------------------------------------------------------------------------
    # ]

    def _transform_action_to_motor_command(self, action):
        # Add the reference trajectory.
        return self._convert_from_leg_model(action)

    def is_fallen(self):
        """Decides whether Rex is in a fallen state.

    If the roll or the pitch of the base is greater than 0.3 radians, the
    rex is considered fallen.

    Returns:
      Boolean value that indicates whether Rex has fallen.
    """
        roll, pitch, _ = self.rex.GetTrueBaseRollPitchYaw()
        return math.fabs(roll) > 0.3 or math.fabs(pitch) > 0.5

    def _get_true_observation(self):
        """Get the true observations of this environment.

    It includes the roll, the pitch, the roll dot and the pitch dot of the base.
    If _use_angle_in_observation is true, eight motor angles are added into the
    observation.

    Returns:
      The observation list, which is a numpy array of floating-point values.
    """
        roll, pitch, _ = self.rex.GetTrueBaseRollPitchYaw()
        roll_rate, pitch_rate, _ = self.rex.GetTrueBaseRollPitchYawRate()
        observation = [roll, pitch, roll_rate, pitch_rate]
        if self._use_angle_in_observation:
            observation.extend(self.rex.GetMotorAngles().tolist())
        self._true_observation = np.array(observation)
        return self._true_observation

    def _get_observation(self):
        roll, pitch, _ = self.rex.GetBaseRollPitchYaw()
        roll_rate, pitch_rate, _ = self.rex.GetBaseRollPitchYawRate()
        observation = [roll, pitch, roll_rate, pitch_rate]
        if self._use_angle_in_observation:
            observation.extend(self.rex.GetMotorAngles().tolist())
        self._observation = np.array(observation)
        return self._observation

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See _get_true_observation() for the
      details of each element of an observation.
    """
        upper_bound_roll = 2 * math.pi
        upper_bound_pitch = 2 * math.pi
        upper_bound_roll_dot = 2 * math.pi / self._time_step
        upper_bound_pitch_dot = 2 * math.pi / self._time_step
        upper_bound_motor_angle = 2 * math.pi
        upper_bound = [
            upper_bound_roll, upper_bound_pitch, upper_bound_roll_dot, upper_bound_pitch_dot
        ]

        if self._use_angle_in_observation:
            upper_bound.extend([upper_bound_motor_angle] * NUM_MOTORS)
        return np.array(upper_bound)

    def _get_observation_lower_bound(self):
        lower_bound = -self._get_observation_upper_bound()
        return lower_bound

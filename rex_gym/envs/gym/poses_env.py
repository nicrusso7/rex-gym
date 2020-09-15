"""This file implements the gym environment of rex alternating legs.

"""
import collections
import math
import random

from gym import spaces
import numpy as np
from .. import rex_gym_env
from rex_gym.model.kinematics import Kinematics
from ...model.rex import Rex

STEP_PERIOD = 1.0 / 15.0  # 15 steps per second.
STEP_AMPLITUDE = 0.25

NUM_LEGS = 4
NUM_MOTORS = 3 * NUM_LEGS


class RexPosesEnv(rex_gym_env.RexGymEnv):
    """The gym environment for the rex.

  It simulates the locomotion of a rex, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the rex walks in 1000 steps and penalizes the energy
  expenditure.

  """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 66}
    load_ui = True
    manual_control = False

    def __init__(self,
                 debug=False,
                 urdf_version=None,
                 control_time_step=0.006,
                 action_repeat=6,
                 control_latency=0,
                 pd_latency=0,
                 on_rack=False,
                 motor_kp=1.0,
                 motor_kd=0.02,
                 remove_default_joint_damping=False,
                 render=False,
                 num_steps_to_log=1000,
                 env_randomizer=None,
                 log_path=None,
                 base_y=None,
                 base_z=None,
                 base_roll=None,
                 base_pitch=None,
                 base_yaw=None,
                 signal_type='ik',
                 terrain_type="plane",
                 terrain_id=None,
                 mark='base'):
        """Initialize the rex alternating legs gym environment.

    Args:
      urdf_version: [DEFAULT_URDF_VERSION, DERPY_V0_URDF_VERSION] are allowable
        versions. If None, DEFAULT_URDF_VERSION is used. Refer to
        rex_gym_env for more details.
      control_time_step: The time step between two successive control signals.
      action_repeat: The number of simulation steps that an action is repeated.
      control_latency: The latency between get_observation() and the actual
        observation. See minituar.py for more details.
      pd_latency: The latency used to get motor angles/velocities used to
        compute PD controllers. See rex.py for more details.
      on_rack: Whether to place the rex on rack. This is only used to debug
        the walk gait. In this mode, the rex's base is hung midair so
        that its walk gait is clearer to visualize.
      motor_kp: The P gain of the motor.
      motor_kd: The D gain of the motor.
      remove_default_joint_damping: Whether to remove the default joint damping.
      render: Whether to render the simulation.
      num_steps_to_log: The max number of control steps in one episode. If the
        number of steps is over num_steps_to_log, the environment will still
        be running, but only first num_steps_to_log will be recorded in logging.
      env_randomizer: An instance (or a list) of EnvRanzomier(s) that can
        randomize the environment during when env.reset() is called and add
        perturbations when env.step() is called.
      log_path: The path to write out logs. For the details of logging, refer to
        rex_logging.proto.
    """
        super(RexPosesEnv,
              self).__init__(urdf_version=urdf_version,
                             accurate_motor_model_enabled=True,
                             motor_overheat_protection=True,
                             hard_reset=False,
                             motor_kp=motor_kp,
                             motor_kd=motor_kd,
                             remove_default_joint_damping=remove_default_joint_damping,
                             control_latency=control_latency,
                             pd_latency=pd_latency,
                             on_rack=on_rack,
                             render=render,
                             num_steps_to_log=num_steps_to_log,
                             env_randomizer=env_randomizer,
                             log_path=log_path,
                             control_time_step=control_time_step,
                             action_repeat=action_repeat,
                             base_y=base_y,
                             base_z=base_z,
                             base_roll=base_roll,
                             base_pitch=base_pitch,
                             base_yaw=base_yaw,
                             debug=debug,
                             signal_type=signal_type,
                             terrain_id=terrain_id,
                             terrain_type=terrain_type,
                             mark=mark)
        self.mark = mark
        action_dim = 1
        action_high = np.array([0.1] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self._cam_dist = 1.0
        self._cam_yaw = 30
        self._cam_pitch = -30
        self.stand = False
        if self._on_rack:
            self._cam_pitch = 0
        self._init_base_x = self._ranges["base_x"][2]

    def setup_ui_params(self):
        self.base_x = self._pybullet_client.addUserDebugParameter("base_x",
                                                                  self._ranges["base_x"][0],
                                                                  self._ranges["base_x"][1],
                                                                  self._ranges["base_x"][2])
        self.base_y = self._pybullet_client.addUserDebugParameter("base_y",
                                                                  self._ranges["base_y"][0],
                                                                  self._ranges["base_y"][1],
                                                                  self._ranges["base_y"][2])
        self.base_z = self._pybullet_client.addUserDebugParameter("base_z",
                                                                  self._ranges["base_z"][0],
                                                                  self._ranges["base_z"][1],
                                                                  self._ranges["base_z"][2])
        self.roll = self._pybullet_client.addUserDebugParameter("roll",
                                                                self._ranges["roll"][0],
                                                                self._ranges["roll"][1],
                                                                self._ranges["roll"][2])
        self.pitch = self._pybullet_client.addUserDebugParameter("pitch",
                                                                 self._ranges["pitch"][0],
                                                                 self._ranges["pitch"][1],
                                                                 self._ranges["pitch"][2])
        self.yaw = self._pybullet_client.addUserDebugParameter("yaw",
                                                               self._ranges["yaw"][0],
                                                               self._ranges["yaw"][1],
                                                               self._ranges["yaw"][2])

    def reset(self):
        super(RexPosesEnv, self).reset()
        if self._is_render:
            if self.load_ui:
                self.setup_ui_params()
                self.load_ui = False
                self.manual_control = True
        else:
            if self._base_y is not None or self._base_z is not None or self._base_roll is not None \
                    or self._base_pitch is not None or self._base_yaw is not None:
                self.fill_next_pose_and_target()
            else:

                self.next_pose = self._queue.popleft()
                # requeue element
                self._queue.append(self.next_pose)
                self.target_value = random.uniform(self._ranges[self.next_pose][0], self._ranges[self.next_pose][1])
        self.values = self._ranges.copy()
        return self._get_observation()

    def fill_next_pose_and_target(self):
        if self._base_y != 0.0:
            self.next_pose = "base_y"
            self.target_value = self._base_y
        elif self._base_z != 0.0:
            self.next_pose = "base_z"
            self.target_value = self._base_z
        elif self._base_roll != 0.0:
            self.next_pose = "roll"
            self.target_value = self._base_roll
        elif self._base_pitch != 0.0:
            self.next_pose = "pitch"
            self.target_value = self._base_pitch
        else:
            self.next_pose = "yaw"
            self.target_value = self._base_yaw

    @staticmethod
    def _evaluate_stage_coefficient(current_t, action, end_t=0.0):
        # ramp function
        p = 0.8 + action[0]
        if end_t <= current_t <= p + end_t:
            return current_t
        else:
            return 1.0

    def _signal(self, t, action):
        if not self.manual_control:
            stage_coeff = self._evaluate_stage_coefficient(t, action)
            staged_value = self.target_value * stage_coeff
            self.values[self.next_pose] = (self.values[self.next_pose][0],
                                           self.values[self.next_pose][1],
                                           staged_value)
            self.position = np.array([
                self.values["base_x"][2],
                self.values["base_y"][2],
                self.values["base_z"][2]
            ])
            self.orientation = np.array([
                self.values["roll"][2],
                self.values["pitch"][2],
                self.values["yaw"][2]
            ])
        else:
            self.position, self.orientation = self._read_inputs()
        kinematics = Kinematics()
        fr_angles, fl_angles, rr_angles, rl_angles, _ = kinematics.solve(self.orientation, self.position)
        signal = [
            fl_angles[0], fl_angles[1], fl_angles[2],
            fr_angles[0], fr_angles[1], fr_angles[2],
            rl_angles[0], rl_angles[1], rl_angles[2],
            rr_angles[0], rr_angles[1], rr_angles[2]
        ]
        return signal

    def _read_inputs(self):
        position = np.array(
            [
                self._pybullet_client.readUserDebugParameter(self.base_x),
                self._pybullet_client.readUserDebugParameter(self.base_y),
                self._pybullet_client.readUserDebugParameter(self.base_z)
            ]
        )
        orientation = np.array(
            [
                self._pybullet_client.readUserDebugParameter(self.roll),
                self._pybullet_client.readUserDebugParameter(self.pitch),
                self._pybullet_client.readUserDebugParameter(self.yaw)
            ]
        )
        return position, orientation

    @staticmethod
    def _convert_from_leg_model(leg_pose):
        motor_pose = np.zeros(NUM_MOTORS)
        for i in range(NUM_LEGS):
            motor_pose[3 * i] = leg_pose[3 * i]
            motor_pose[3 * i + 1] = leg_pose[3 * i + 1]
            motor_pose[3 * i + 2] = leg_pose[3 * i + 2]
        return motor_pose

    def _transform_action_to_motor_command(self, action):
        action = self._signal(self.rex.GetTimeSinceReset(), action)
        action = self._convert_from_leg_model(action)
        action = super(RexPosesEnv, self)._transform_action_to_motor_command(action)
        return action

    def is_fallen(self):
        """Decide whether the rex has fallen.
    Returns:
      Boolean value that indicates whether the rex has fallen.
    """
        roll, _, _ = self.rex.GetTrueBaseRollPitchYaw()
        return False

    def _reward(self):
        # positive reward as long as rex stands
        return 1.0

    def _get_true_observation(self):
        """Get the true observations of this environment.

    It includes the roll, the error between current pitch and desired pitch,
    roll dot and pitch dot of the base.

    Returns:
      The observation list.
    """
        observation = []
        roll, pitch, _ = self.rex.GetTrueBaseRollPitchYaw()
        roll_rate, pitch_rate, _ = self.rex.GetTrueBaseRollPitchYawRate()
        observation.extend([roll, pitch, roll_rate, pitch_rate])
        self._true_observation = np.array(observation)
        return self._true_observation

    def _get_observation(self):
        observation = []
        roll, pitch, _ = self.rex.GetBaseRollPitchYaw()
        roll_rate, pitch_rate, _ = self.rex.GetBaseRollPitchYawRate()
        observation.extend([roll, pitch, roll_rate, pitch_rate])
        self._observation = np.array(observation)
        return self._observation

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
        upper_bound = np.zeros(self._get_observation_dimension())
        upper_bound[0:2] = 2 * math.pi  # Roll, pitch, yaw of the base.
        upper_bound[2:4] = 2 * math.pi / self._time_step  # Roll, pitch, yaw rate.
        return upper_bound

    def _get_observation_lower_bound(self):
        lower_bound = -self._get_observation_upper_bound()
        return lower_bound

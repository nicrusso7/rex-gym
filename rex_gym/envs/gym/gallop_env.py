"""This file implements the gym environment of Rex.

"""
import collections
import math
import random

from gym import spaces
import numpy as np
from rex_gym.model import mark_constants

from rex_gym.model.gait_planner import GaitPlanner

from .. import rex_gym_env

# Radiant
from ...model import rex_constants
from ...model.kinematics import Kinematics

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
    load_ui = True
    is_terminating = False

    def __init__(self,
                 debug=False,
                 urdf_version=None,
                 energy_weight=0.005,
                 control_time_step=0.006,
                 action_repeat=6,
                 control_latency=0.0,
                 pd_latency=0.0,
                 on_rack=False,
                 motor_kp=1.0,
                 motor_kd=0.02,
                 render=False,
                 num_steps_to_log=2000,
                 use_angle_in_observation=True,
                 env_randomizer=None,
                 log_path=None,
                 target_position=None,
                 signal_type="ik",
                 terrain_type="plane",
                 terrain_id=None,
                 mark='base'):
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
        the walk gait. In this mode, Rex's base is hung midair so
        that its walk gait is clearer to visualize.
      motor_kp: The P gain of the motor.
      motor_kd: The D gain of the motor.
      num_steps_to_log: The max number of control steps in one episode. If the
        number of steps is over num_steps_to_log, the environment will still
        be running, but only first num_steps_to_log will be recorded in logging.
      use_angle_in_observation: Whether to include motor angles in observation.
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
                             accurate_motor_model_enabled=True,
                             motor_overheat_protection=True,
                             hard_reset=False,
                             motor_kp=motor_kp,
                             motor_kd=motor_kd,
                             remove_default_joint_damping=False,
                             control_latency=control_latency,
                             pd_latency=pd_latency,
                             on_rack=on_rack,
                             render=render,
                             num_steps_to_log=num_steps_to_log,
                             env_randomizer=env_randomizer,
                             log_path=log_path,
                             control_time_step=control_time_step,
                             action_repeat=action_repeat,
                             target_position=target_position,
                             signal_type=signal_type,
                             debug=debug,
                             terrain_id=terrain_id,
                             terrain_type=terrain_type,
                             mark=mark)
        # (eventually) allow different feedback ranges/action spaces for different signals
        action_max = {
            'ik': 0.4,
            'ol': 0.3
        }
        action_dim_map = {
            'ik': 2,
            'ol': 4,
        }
        action_dim = action_dim_map[self._signal_type]
        action_low = np.array([action_max[self._signal_type]] * action_dim)
        action_high = -action_low
        self.action_space = spaces.Box(action_low, action_high)
        self._cam_dist = 1.0
        self._cam_yaw = 0.0
        self._cam_pitch = -20
        self._target_position = target_position
        self._signal_type = signal_type
        self._gait_planner = GaitPlanner("gallop")
        self._kinematics = Kinematics()
        self.goal_reached = False
        self._stay_still = False
        self.is_terminating = False

    def reset(self):
        self.init_pose = rex_constants.INIT_POSES["stand"]
        if self._signal_type == 'ol':
            self.init_pose = rex_constants.INIT_POSES["stand_ol"]
        super(RexReactiveEnv, self).reset(initial_motor_angles=self.init_pose, reset_duration=0.5)
        self.goal_reached = False
        self._stay_still = False
        self.is_terminating = False
        if not self._target_position or self._random_pos_target:
            self._target_position = random.uniform(1, 3)
            self._random_pos_target = True
        if self._is_render and self._signal_type == 'ik':
            if self.load_ui:
                self.setup_ui()
                self.load_ui = False
        if self._is_debug:
            print(f"Target Position x={self._target_position}, Random assignment: {self._random_pos_target}")
        return self._get_observation()

    def setup_ui(self):
        self.base_x_ui = self._pybullet_client.addUserDebugParameter("base_x",
                                                                     self._ranges["base_x"][0],
                                                                     self._ranges["base_x"][1],
                                                                     0.01)
        self.base_y_ui = self._pybullet_client.addUserDebugParameter("base_y",
                                                                     self._ranges["base_y"][0],
                                                                     self._ranges["base_y"][1],
                                                                     self._ranges["base_y"][2])
        self.base_z_ui = self._pybullet_client.addUserDebugParameter("base_z",
                                                                     self._ranges["base_z"][0],
                                                                     self._ranges["base_z"][1],
                                                                     -0.007)
        self.roll_ui = self._pybullet_client.addUserDebugParameter("roll",
                                                                   self._ranges["roll"][0],
                                                                   self._ranges["roll"][1],
                                                                   self._ranges["roll"][2])
        self.pitch_ui = self._pybullet_client.addUserDebugParameter("pitch",
                                                                    self._ranges["pitch"][0],
                                                                    self._ranges["pitch"][1],
                                                                    self._ranges["pitch"][2])
        self.yaw_ui = self._pybullet_client.addUserDebugParameter("yaw",
                                                                  self._ranges["yaw"][0],
                                                                  self._ranges["yaw"][1],
                                                                  self._ranges["yaw"][2])
        self.step_length_ui = self._pybullet_client.addUserDebugParameter("step_length", -0.7, 1.5,  1.3)
        self.step_rotation_ui = self._pybullet_client.addUserDebugParameter("step_rotation", -1.5, 1.5, 0.)
        self.step_angle_ui = self._pybullet_client.addUserDebugParameter("step_angle", -180., 180., 0.)
        self.step_period_ui = self._pybullet_client.addUserDebugParameter("step_period", 0.1, .9, 0.3)

    def _read_inputs(self, base_pos_coeff, gait_stage_coeff):
        position = np.array(
            [
                self._pybullet_client.readUserDebugParameter(self.base_x_ui),
                self._pybullet_client.readUserDebugParameter(self.base_y_ui) * base_pos_coeff,
                self._pybullet_client.readUserDebugParameter(self.base_z_ui)
            ]
        )
        orientation = np.array(
            [
                self._pybullet_client.readUserDebugParameter(self.roll_ui) * base_pos_coeff,
                self._pybullet_client.readUserDebugParameter(self.pitch_ui) * base_pos_coeff,
                self._pybullet_client.readUserDebugParameter(self.yaw_ui) * base_pos_coeff
            ]
        )
        step_length = self._pybullet_client.readUserDebugParameter(self.step_length_ui) * gait_stage_coeff
        step_rotation = self._pybullet_client.readUserDebugParameter(self.step_rotation_ui)
        step_angle = self._pybullet_client.readUserDebugParameter(self.step_angle_ui)
        step_period = self._pybullet_client.readUserDebugParameter(self.step_period_ui)
        return position, orientation, step_length, step_rotation, step_angle, step_period

    def _check_target_position(self, t):
        if self._target_position:
            current_x = abs(self.rex.GetBasePosition()[0])
            # give 0.15 stop space
            if current_x >= abs(self._target_position):
                self.goal_reached = True
                if not self.is_terminating:
                    self.end_time = t
                    self.is_terminating = True

    @staticmethod
    def _evaluate_stage_coefficient(current_t, end_t=0.0, width=0.001):
        # sigmoid function
        beta = p = width
        if p - beta + end_t <= current_t <= p - (beta / 2) + end_t:
            return (2 / beta ** 2) * (current_t - p + beta) ** 2
        elif p - (beta/2) + end_t <= current_t <= p + end_t:
            return 1 - (2 / beta ** 2) * (current_t - p) ** 2
        else:
            return 1

    @staticmethod
    def _evaluate_brakes_stage_coeff(current_t, action, end_t=0.0, end_value=0.0):
        # ramp function
        p = 1. + action[0]
        if end_t <= current_t <= p + end_t:
            return 1 - (current_t - end_t)
        else:
            return end_value

    @staticmethod
    def _evaluate_gait_stage_coeff(current_t, action, end_t=0.0):
        # ramp function
        p = 1. + action[1]
        if end_t <= current_t <= p + end_t:
            return current_t
        else:
            return 1.0

    def _signal(self, t, action):
        if self._signal_type == 'ik':
            return self._IK_signal(t, action)
        if self._signal_type == 'ol':
            return self._open_loop_signal(t, action)

    def _IK_signal(self, t, action):
        base_pos_coeff = self._evaluate_stage_coefficient(t, width=1.5)
        gait_stage_coeff = self._evaluate_gait_stage_coeff(t, action)
        if self._is_render:
            position, orientation, step_length, step_rotation, step_angle, step_period = \
                self._read_inputs(base_pos_coeff, gait_stage_coeff)
        else:
            position = np.array([0.01,
                                 self._base_y * base_pos_coeff,
                                 -0.007])
            orientation = np.array([self._base_roll * base_pos_coeff,
                                    self._base_pitch * base_pos_coeff,
                                    self._base_yaw * base_pos_coeff])
            step_length = (self.step_length if self.step_length is not None else 1.3) * gait_stage_coeff
            step_rotation = (self.step_rotation if self.step_rotation is not None else 0.0)
            step_angle = self.step_angle if self.step_angle is not None else 0.0
            step_period = (self.step_period if self.step_period is not None else 0.3)
        if self.goal_reached:
            brakes_coeff = self._evaluate_brakes_stage_coeff(t, action, self.end_time)
            step_length *= brakes_coeff
        frames = self._gait_planner.loop(step_length, step_angle, step_rotation, step_period, 1.0)
        fr_angles, fl_angles, rr_angles, rl_angles, _ = self._kinematics.solve(orientation, position, frames)
        signal = [
            fl_angles[0], fl_angles[1], fl_angles[2],
            fr_angles[0], fr_angles[1], fr_angles[2],
            rl_angles[0], rl_angles[1], rl_angles[2],
            rr_angles[0], rr_angles[1], rr_angles[2]
        ]
        return signal

    def _open_loop_signal(self, t, leg_pose):
        if self.goal_reached:
            coeff = self._evaluate_brakes_stage_coeff(t, [.0], end_t=self.end_time, end_value=0.0)
            leg_pose *= coeff
            if coeff is 0.0:
                self._stay_still = True
        motor_pose = np.zeros(NUM_MOTORS)
        for i in range(NUM_LEGS):
            motor_pose[int(3 * i)] = self.init_pose[3 * i]
            init_leg = self.init_pose[3 * i + 1]
            init_foot = self.init_pose[3 * i + 2]
            if i == 0 or i == 1:
                motor_pose[int(3 * i + 1)] = init_leg + leg_pose[0]
                motor_pose[int(3 * i + 2)] = init_foot + leg_pose[1]
            else:
                motor_pose[int(3 * i + 1)] = init_leg + leg_pose[2]
                motor_pose[int(3 * i + 2)] = init_foot + leg_pose[3]
        return motor_pose

    def _transform_action_to_motor_command(self, action):
        if self._stay_still:
            return self.rex.initial_pose
        t = self.rex.GetTimeSinceReset()
        self._check_target_position(t)
        action = self._signal(t, action)
        action = super(RexReactiveEnv, self)._transform_action_to_motor_command(action)
        return action

    def _out_of_trajectory(self):
        current_base_position = self.rex.GetBasePosition()
        return current_base_position[1] > 0.3

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
            upper_bound.extend([upper_bound_motor_angle] * mark_constants.MARK_DETAILS['motors_num'][self.mark])
        return np.array(upper_bound)

    def _get_observation_lower_bound(self):
        lower_bound = -self._get_observation_upper_bound()
        return lower_bound

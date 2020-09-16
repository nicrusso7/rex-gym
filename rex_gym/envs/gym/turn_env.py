"""This file implements the gym environment of rex turning on the spot."""
import math
import random

from gym import spaces
import numpy as np
from rex_gym.model.gait_planner import GaitPlanner

from rex_gym.util import pybullet_data

from .. import rex_gym_env
from ...model import rex_constants
from ...model.kinematics import Kinematics
from ...model.rex import Rex

NUM_LEGS = 4
STEP_PERIOD = 1.0 / 10.0  # 10 steps per second.


class RexTurnEnv(rex_gym_env.RexGymEnv):
    """The gym environment for the rex.
    It simulates the locomotion of a rex, a quadruped robot. The state space
    include the angles, velocities and torques for all the motors and the action
    space is the desired motor angle for each motor. The reward function is based
    on how far the rex walks in 1000 steps and penalizes the energy expenditure."""
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 66}
    load_ui = True
    is_terminating = False

    def __init__(self,
                 debug=False,
                 urdf_version=None,
                 control_time_step=0.005,
                 action_repeat=5,
                 control_latency=0,
                 pd_latency=0,
                 on_rack=False,
                 motor_kp=1.0,
                 motor_kd=0.02,
                 render=False,
                 num_steps_to_log=1000,
                 env_randomizer=None,
                 log_path=None,
                 target_orient=None,
                 init_orient=None,
                 signal_type="ik",
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
            observation. See rex.py for more details.
          pd_latency: The latency used to get motor angles/velocities used to
            compute PD controllers. See rex.py for more details.
          on_rack: Whether to place the rex on rack. This is only used to debug
            the walk gait. In this mode, the rex's base is hung midair so
            that its walk gait is clearer to visualize.
          motor_kp: The P gain of the motor.
          motor_kd: The D gain of the motor.
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
        super(RexTurnEnv, self).__init__(
            debug=debug,
            urdf_version=urdf_version,
            accurate_motor_model_enabled=True,
            motor_overheat_protection=True,
            hard_reset=False,
            motor_kp=motor_kp,
            motor_kd=motor_kd,
            control_latency=control_latency,
            pd_latency=pd_latency,
            on_rack=on_rack,
            render=render,
            num_steps_to_log=num_steps_to_log,
            env_randomizer=env_randomizer,
            log_path=log_path,
            control_time_step=control_time_step,
            action_repeat=action_repeat,
            target_orient=target_orient,
            signal_type=signal_type,
            init_orient=init_orient,
            terrain_id=terrain_id,
            terrain_type=terrain_type,
            mark=mark)
        action_max = {
            'ik': 0.01,
            'ol': 0.01
        }
        action_dim_map = {
            'ik': 2,
            'ol': 2
        }
        action_dim = action_dim_map[self._signal_type]
        action_high = np.array([action_max[self._signal_type]] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self._cam_dist = 1.1
        self._cam_yaw = 30
        self._cam_pitch = -30
        self._signal_type = signal_type
        # we need an alternate gait, so walk will works
        self._gait_planner = GaitPlanner("walk")
        self._kinematics = Kinematics()
        self._target_orient = target_orient
        self._init_orient = init_orient
        self._random_orient_target = False
        self._random_orient_start = False
        self._cube = None
        self.goal_reached = False
        self._stay_still = False
        self.is_terminating = False
        if self._on_rack:
            self._cam_pitch = 0

    def reset(self):
        self.goal_reached = False
        self.is_terminating = False
        self._stay_still = False
        self.init_pose = rex_constants.INIT_POSES["stand"]
        if self._signal_type == 'ol':
            self.init_pose = rex_constants.INIT_POSES["stand_ol"]
        super(RexTurnEnv, self).reset(initial_motor_angles=self.init_pose, reset_duration=0.5)
        if not self._target_orient or self._random_orient_target:
            self._target_orient = random.uniform(0.2, 6)
            self._random_orient_target = True
        if self._on_rack:
            # on rack debug simulation
            self._init_orient = 2.1
            position = self.rex.init_on_rack_position
        else:
            position = self.rex.init_position
            if self._init_orient is None or self._random_orient_start:
                self._init_orient = random.uniform(0.2, 6)
                self._random_orient_start = True
        self.clockwise = self._solve_direction()
        if self._is_debug:
            print(f"Start Orientation: {self._init_orient}, Target Orientation: {self._target_orient}")
            print("Turning right") if self.clockwise else print("Turning left")
        if self._is_render and self._signal_type == 'ik':
            if self.load_ui:
                self.setup_ui()
                self.load_ui = False
        self._load_cube(self._target_orient)
        q = self.pybullet_client.getQuaternionFromEuler([0, 0, self._init_orient])
        self.pybullet_client.resetBasePositionAndOrientation(self.rex.quadruped, position, q)
        return self._get_observation()

    def setup_ui(self):
        self.base_x_ui = self._pybullet_client.addUserDebugParameter("base_x",
                                                                     self._ranges["base_x"][0],
                                                                     self._ranges["base_x"][1],
                                                                     0.009)
        self.base_y_ui = self._pybullet_client.addUserDebugParameter("base_y",
                                                                     self._ranges["base_y"][0],
                                                                     self._ranges["base_y"][1],
                                                                     self._ranges["base_y"][2])
        self.base_z_ui = self._pybullet_client.addUserDebugParameter("base_z",
                                                                     self._ranges["base_z"][0],
                                                                     self._ranges["base_z"][1],
                                                                     self._ranges["base_z"][2])
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
        self.step_length_ui = self._pybullet_client.addUserDebugParameter("step_length", -0.7, 0.7, 0.02)
        self.step_rotation_ui = self._pybullet_client.addUserDebugParameter("step_rotation", -1.5, 1.5, 0.5)
        self.step_angle_ui = self._pybullet_client.addUserDebugParameter("step_angle", -180., 180., 0.)
        self.step_period_ui = self._pybullet_client.addUserDebugParameter("step_period", 0.2, 0.9, 0.75)

    def _read_inputs(self, base_pos_coeff, gait_stage_coeff):
        position = np.array(
            [
                self._pybullet_client.readUserDebugParameter(self.base_x_ui),
                self._pybullet_client.readUserDebugParameter(self.base_y_ui) * base_pos_coeff,
                self._pybullet_client.readUserDebugParameter(self.base_z_ui) * base_pos_coeff
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
        step_rotation = self._pybullet_client.readUserDebugParameter(self.step_rotation_ui) * gait_stage_coeff
        step_angle = self._pybullet_client.readUserDebugParameter(self.step_angle_ui)
        step_period = self._pybullet_client.readUserDebugParameter(self.step_period_ui)
        return position, orientation, step_length, step_rotation, step_angle, step_period

    def _signal(self, t, action):
        if self._signal_type == 'ik':
            return self._IK_signal(t, action)
        if self._signal_type == 'ol':
            return self._open_loop_signal(t, action)

    @staticmethod
    def _evaluate_base_stage_coeff(current_t, end_t=0.0, width=0.001):
        # sigmoid function
        beta = p = width
        if p - beta + end_t <= current_t <= p - (beta / 2) + end_t:
            return (2 / beta ** 2) * (current_t - p + beta) ** 2
        elif p - (beta/2) + end_t <= current_t <= p + end_t:
            return 1 - (2 / beta ** 2) * (current_t - p) ** 2
        else:
            return 1

    @staticmethod
    def _evaluate_gait_stage_coeff(current_t, end_t=0.0):
        # ramp function
        p = .8
        if end_t <= current_t <= p + end_t:
            return current_t
        else:
            return 1.0

    def _IK_signal(self, t, action):
        base_pos_coeff = self._evaluate_base_stage_coeff(t, width=1.5)
        gait_stage_coeff = self._evaluate_gait_stage_coeff(t)
        if self._is_render and self._is_debug:
            position, orientation, step_length, step_rotation, step_angle, step_period = \
                self._read_inputs(base_pos_coeff, gait_stage_coeff)
        else:
            step_dir_value = -0.5 * gait_stage_coeff
            if self.clockwise:
                step_dir_value *= -1
            position = np.array([0.009,
                                 self._base_y * base_pos_coeff,
                                 self._base_z * base_pos_coeff])
            orientation = np.array([self._base_roll * base_pos_coeff,
                                    self._base_pitch * base_pos_coeff,
                                    self._base_yaw * base_pos_coeff])
            step_length = (self.step_length if self.step_length is not None else 0.02)
            step_rotation = (self.step_rotation if self.step_rotation is not None else step_dir_value) + action[0]
            step_angle = self.step_angle if self.step_angle is not None else 0.0
            step_period = (self.step_period if self.step_period is not None else 0.75) + action[1]
        if self.goal_reached:
            self._stay_still = True
        frames = self._gait_planner.loop(step_length, step_angle, step_rotation, step_period, 1.0)
        fr_angles, fl_angles, rr_angles, rl_angles, _ = self._kinematics.solve(orientation, position, frames)
        signal = [
            fl_angles[0], fl_angles[1], fl_angles[2],
            fr_angles[0], fr_angles[1], fr_angles[2],
            rl_angles[0], rl_angles[1], rl_angles[2],
            rr_angles[0], rr_angles[1], rr_angles[2]
        ]
        return signal

    def _open_loop_signal(self, t, action):
        if self.goal_reached:
            self._stay_still = True
        initial_pose = rex_constants.INIT_POSES['stand_ol']
        period = STEP_PERIOD
        extension = 0.1
        swing = 0.03 + action[0]
        swipe = 0.05 + action[1]
        ith_leg = int(t / period) % 2
        pose = {
            'left_0': np.array([swipe, extension, -swing,
                                -swipe, extension, swing,
                                swipe, -extension, swing,
                                -swipe, -extension, -swing]),
            'left_1': np.array([-swipe, 0, swing,
                                swipe, 0, -swing,
                                -swipe, 0, -swing,
                                swipe, 0, swing]),
            'right_0': np.array([swipe, extension, swing,
                                 -swipe, extension, -swing,
                                 swipe, -extension, -swing,
                                 -swipe, -extension, swing]),
            'right_1': np.array([-swipe, 0, -swing,
                                 swipe, 0, swing,
                                 -swipe, 0, swing,
                                 swipe, 0, -swing])
        }
        clockwise = self._solve_direction()
        if clockwise:
            # turn right
            first_leg = pose['right_0']
            second_leg = pose['right_1']
        else:
            # turn left
            first_leg = pose['left_0']
            second_leg = pose['left_1']
        if ith_leg:
            signal = initial_pose + second_leg
        else:
            signal = initial_pose + first_leg
        return signal

    def _solve_direction(self):
        diff = abs(self._init_orient - self._target_orient)
        clockwise = False
        if self._init_orient < self._target_orient:
            if diff > 3.14:
                clockwise = True
        else:
            if diff < 3.14:
                clockwise = True
        return clockwise

    def _check_target_position(self, t):
        current_z = self.pybullet_client.getEulerFromQuaternion(self.rex.GetBaseOrientation())[2]
        if current_z < 0:
            current_z += 6.28
        if abs(self._target_orient - current_z) <= 0.01:
            self.goal_reached = True
            if not self.is_terminating:
                self.end_time = t
                self.is_terminating = True

    def _terminate_with_delay(self, current_t):
        if current_t - self.end_time >= 1.:
            self.env_goal_reached = True

    def _transform_action_to_motor_command(self, action):
        if self._stay_still:
            self._terminate_with_delay(self.rex.GetTimeSinceReset())
            return self.init_pose
        t = self.rex.GetTimeSinceReset()
        self._check_target_position(t)
        action = self._signal(t, action)
        action = super(RexTurnEnv, self)._transform_action_to_motor_command(action)
        return action

    def is_fallen(self):
        """Decide whether the rex has fallen.

        If the up directions between the base and the world is large (the dot
        product is smaller than 0.85), the rex is considered fallen.

        Returns:
          Boolean value that indicates whether the rex has fallen.
        """
        orientation = self.rex.GetBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        return np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85

    def _reward(self):
        current_base_position = self.rex.GetBasePosition()
        # tolerance: 0.035
        position_penality = 0.035 - abs(current_base_position[0]) - abs(current_base_position[1])
        reward = position_penality
        return reward

    def _load_cube(self, angle):
        if len(self._companion_obj) > 0:
            self.pybullet_client.removeBody(self._companion_obj['cube'])
        urdf_root = pybullet_data.getDataPath()
        self._cube = self._pybullet_client.loadURDF(f"{urdf_root}/cube_small.urdf")
        self._companion_obj['cube'] = self._cube
        orientation = [0, 0, 0, 1]
        x, y = math.cos(angle + 3.14), math.sin(angle + 3.14)
        position = [x, y, 1]
        self.pybullet_client.resetBasePositionAndOrientation(self._cube, position, orientation)

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

"""This file implements the gym environment of rex 'go to'.

"""
import math
import os
import random
import site

from gym import spaces
import tensorflow as tf
import numpy as np
from rex_gym.agents.ppo import simple_ppo_agent

from rex_gym.util import pybullet_data

from .. import rex_gym_env
from ...agents.scripts import utility
from ...util import action_mapper

COMPANION_OBJECTS = {}


class RexGoEnv(rex_gym_env.RexGymEnv):
    """The gym environment for the rex task 'go to'.
    It simulates the locomotion of a rex, a quadruped robot. The state space
    include the angles, velocities and torques for all the motors and the action
    space is the desired motor angle for each motor. The reward function is based
    on rex position and its goal position."""
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 66}

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
                 start_position=None,
                 target_position=None,
                 init_orient=None,
                 signal_type='ol'):
        """Initialize the rex go to gym environment.

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
        super(RexGoEnv, self).__init__(
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
            init_orient=init_orient,
            signal_type=signal_type,
            debug=debug)

        action_dim = 12
        action_high = np.array([0.0] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self._cam_dist = 3.0
        self._cam_yaw = 90
        self._cam_pitch = -60
        self.last_step = 0

        self._target_position = target_position
        self._start_position = start_position
        self._init_orient = init_orient
        self._random_pos_target = False
        self._random_pos_start = False
        self._random_orient_start = False
        self.gym_dir_path = str(site.getsitepackages()[0])
        if self._on_rack:
            self._cam_pitch = 0

    def reset(self):
        self._phases = {'target_alignment': False, 'walk_towards': False, 'target_reached': False}
        super(RexGoEnv, self).reset()
        if self._target_position is None or self._random_pos_target:
            # set random target position
            x, y = random.uniform(-2, 2), random.uniform(-2, 2)
            self._target_position = [x, y, 0.21]
            self._random_pos_target = True

        if self._on_rack:
            # on rack debug simulation
            self._init_orient = 2.1
            self._start_position = self.rex.init_on_rack_position
        else:
            if self._random_pos_start:
                # set random start position
                x, y = random.uniform(-5, 5), random.uniform(-5, 5)
                self._start_position = [x, y, 0.21]
                self._random_pos_start = True
            else:
                self._start_position = self.rex.init_position
            if not self._init_orient or self._random_orient_start:
                self._init_orient = random.uniform(0.1, 6)
                self._random_orient_start = True
        self._load_cube(self._target_position)
        self.load_turn_policy()
        self.load_walk_policy()
        print(f"Start Orientation: {self._init_orient}")
        print(f"Start Position: {self._start_position}, Target Position: {self._target_position}")
        q = self.pybullet_client.getQuaternionFromEuler([0, 0, self._init_orient])
        self.pybullet_client.resetBasePositionAndOrientation(self.rex.quadruped, self._start_position, q)
        return self._get_observation()

    def load_turn_policy(self):
        # solve target orientation
        angle = math.atan((self._target_position[1]) / (self._target_position[0]))
        if angle < 0:
            angle += 6.28
        target_orient = angle
        # load turn Policy in a new Graph
        turn_graph = tf.Graph()
        with turn_graph.as_default():
            gym_dir_path = str(site.getsitepackages()[0])
            policy_dir = os.path.join(gym_dir_path, action_mapper.ENV_ID_TO_POLICY['turn_ol'][0])
            config = utility.load_config(policy_dir)
            policy_layers = config.policy_layers
            value_layers = config.value_layers
            self.turn_env = config.env(render=False, init_orient=self._init_orient, target_orient=target_orient)
            network = config.network
            checkpoint = os.path.join(policy_dir, action_mapper.ENV_ID_TO_POLICY['turn_ol'][1])
            self.turn_agent = simple_ppo_agent.SimplePPOPolicy(tf.compat.v1.Session(graph=turn_graph),
                                                               self.turn_env,
                                                               network,
                                                               policy_layers=policy_layers,
                                                               value_layers=value_layers,
                                                               checkpoint=checkpoint)

    def load_walk_policy(self):
        # load walk Policy in a new Graph
        walk_graph = tf.Graph()
        with walk_graph.as_default():
            gym_dir_path = str(site.getsitepackages()[0])
            policy_dir = os.path.join(gym_dir_path, action_mapper.ENV_ID_TO_POLICY['walk_ol'][0])
            config = utility.load_config(policy_dir)
            policy_layers = config.policy_layers
            value_layers = config.value_layers
            self.walk_env = config.env(render=False, target_position=self._target_position[0])
            network = config.network
            checkpoint = os.path.join(policy_dir, action_mapper.ENV_ID_TO_POLICY['walk_ol'][1])
            self.walk_agent = simple_ppo_agent.SimplePPOPolicy(tf.compat.v1.Session(graph=walk_graph),
                                                               self.walk_env,
                                                               network,
                                                               policy_layers=policy_layers,
                                                               value_layers=value_layers,
                                                               checkpoint=checkpoint)

    def _load_cube(self, target):
        if len(COMPANION_OBJECTS) > 0:
            self.pybullet_client.removeBody(COMPANION_OBJECTS['cube'])
        urdf_root = pybullet_data.getDataPath()
        self._cube = self._pybullet_client.loadURDF(f"{urdf_root}/cube_small.urdf")
        COMPANION_OBJECTS['cube'] = self._cube
        orientation = [0, 0, 0, 1]
        position = [target[0], target[1], 1]
        self.pybullet_client.resetBasePositionAndOrientation(self._cube, position, orientation)

    def _signal(self, observation):
        if not self._phases['target_alignment']:
            action = self.turn_agent.get_action([observation])[0]
            _, _, done, info = self.turn_env.step(action)
            signal = info['action']
            if done:
                self._phases['target_alignment'] = True
        elif not self._phases['walk_towards']:
            action = self.walk_agent.get_action([observation])[0]
            _, _, done, info = self.walk_env.step(action)
            signal = info['action']
            if done:
                self._phases['walk_towards'] = True
        else:
            signal = self.rex.initial_pose
            self._phases['target_reached'] = True
        return signal

    def _transform_action_to_motor_command(self, action):
        # if self._phases['target_reached'] is True:
        #     self.env_goal_reached = True
        # observation = self._get_observation()
        # action = self._signal(observation)
        # return action
        return self.rex.initial_pose

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

        position_reward = \
            abs(self._target_position[0] - current_base_position[0]) + \
            abs(self._target_position[1] - current_base_position[1]) + \
            abs(self._target_position[2] - current_base_position[2])

        is_in_pos = False

        if abs(position_reward) < 0.1:
            position_reward = 100 - position_reward
            is_in_pos = True
        else:
            position_reward = -position_reward

        if is_in_pos:
            self.goal_reached = True
            self.goal_t = self.rex.GetTimeSinceReset()

        reward = position_reward
        return reward

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

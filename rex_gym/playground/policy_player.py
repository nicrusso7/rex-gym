r"""Running a pre-trained ppo agent on rex environments"""
import os
import site
import time

import tensorflow as tf
from rex_gym.agents.scripts import utility
from rex_gym.agents.ppo import simple_ppo_agent
from rex_gym.util import action_mapper


class PolicyPlayer:
    def __init__(self, env_id: str, args: dict):
        self.gym_dir_path = str(site.getsitepackages()[0])
        self.env_id = env_id
        self.args = args

    def play(self):
        policy_dir = os.path.join(self.gym_dir_path, action_mapper.ENV_ID_TO_POLICY[self.env_id][0])
        config = utility.load_config(policy_dir)
        policy_layers = config.policy_layers
        value_layers = config.value_layers
        env = config.env(render=True, **self.args)
        network = config.network
        checkpoint = os.path.join(policy_dir, action_mapper.ENV_ID_TO_POLICY[self.env_id][1])
        with tf.Session() as sess:
            agent = simple_ppo_agent.SimplePPOPolicy(sess,
                                                     env,
                                                     network,
                                                     policy_layers=policy_layers,
                                                     value_layers=value_layers,
                                                     checkpoint=checkpoint)
            sum_reward = 0
            observation = env.reset()
            while True:
                action = agent.get_action([observation])
                observation, reward, done, _ = env.step(action[0])
                time.sleep(0.002)
                sum_reward += reward
                if done:
                    break

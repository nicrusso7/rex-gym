r"""Running a pre-trained ppo agent on rex environments"""
import os
import site
import time

import tensorflow as tf
from rex_gym.agents.scripts import utility
from rex_gym.agents.ppo import simple_ppo_agent

gym_dir_path = os.path.join(str(site.getsitepackages()[0]), 'rex_gym')

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
POLICIES = {
    'rex_galloping': ('policies/galloping/balanced', 'model.ckpt-20000000'),
    'rex_walk': ('policies/walking/alternating_legs', 'model.ckpt-16000000'),
    'rex_turn': ('policies/turn', 'model.ckpt-16000000'),
    'rex_standup': ('policies/standup', 'model.ckpt-10000000')
}


def main(_):
    policy_dir = os.path.join(gym_dir_path, POLICIES[FLAGS.env][0])
    config = utility.load_config(policy_dir)
    policy_layers = config.policy_layers
    value_layers = config.value_layers
    env = config.env(render=True)
    network = config.network

    with tf.Session() as sess:
        agent = simple_ppo_agent.SimplePPOPolicy(sess,
                                                 env,
                                                 network,
                                                 policy_layers=policy_layers,
                                                 value_layers=value_layers,
                                                 checkpoint=os.path.join(policy_dir, POLICIES[FLAGS.env][1]))

        sum_reward = 0
        observation = env.reset()
        while True:
            action = agent.get_action([observation])
            observation, reward, done, _ = env.step(action[0])
            time.sleep(0.002)
            sum_reward += reward
            if done:
                break
        tf.logging.info("reward: %s", sum_reward)


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('env', None, 'Environment name.')
    tf.app.run()

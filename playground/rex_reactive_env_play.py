r"""Running a pre-trained ppo agent on rex_reactive_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
from agents.scripts import utility
from agents.ppo import simple_ppo_agent

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
LOG_DIR = "policies/galloping/bounded-rex_reactive"
CHECKPOINT = "model.ckpt-14000000"

"""
Replace the policy config.yml "env" field with this:

env: !!python/object/apply:functools.partial
  args:
  - &id001 !!python/name:envs.rex_reactive_env.RexReactiveEnv ''
  state: !!python/tuple
  - *id001
  - !!python/tuple []
  - accurate_motor_model_enabled: true
    control_latency: 0.02
    energy_weight: 0.005
    env_randomizer: null
    motor_kd: 0.015
    num_steps_to_log: 1000
    pd_latency: 0.003
    remove_default_joint_damping: true
    render: true
  - null

  """


def main(argv):
    del argv  # Unused.
    config = utility.load_config(LOG_DIR)
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
                                                 checkpoint=os.path.join(LOG_DIR, CHECKPOINT))

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
    tf.app.run(main)

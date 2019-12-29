r"""Running a pre-trained ppo agent on rex_trotting_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import site
import time
import tensorflow as tf
from rex_gym.agents.scripts import utility
from rex_gym.agents.ppo import simple_ppo_agent

gym_dir_path = os.path.join(str(site.getsitepackages()[0]), 'rex_gym')

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
LOG_DIR = os.path.join(gym_dir_path, 'policies/walking/alternating_legs')
CHECKPOINT = "model.ckpt-4000000"

"""
  env: !!python/object/apply:functools.partial
    args:
    - &id001 !!python/name:rex_gym.envs.gym.rex_alternating_legs_env.RexAlternatingLegsEnv ''
    state: !!python/tuple
    - *id001
    - !!python/tuple []
    - env_randomizer: null
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

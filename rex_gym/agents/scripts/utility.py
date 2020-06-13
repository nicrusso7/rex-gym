# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for using reinforcement learning algorithms."""
import logging
import os
import re
import warnings

import ruamel.yaml as yaml
import tensorflow as tf

from rex_gym.agents.tools import wrappers
from rex_gym.agents.tools.attr_dict import AttrDict
from rex_gym.agents.tools.batch_env import BatchEnv
from rex_gym.agents.tools.count_weights import count_weights
from rex_gym.agents.tools.in_graph_batch_env import InGraphBatchEnv
from rex_gym.agents.tools.simulate import simulate

warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
warnings.simplefilter('ignore', yaml.error.ReusedAnchorWarning)


def define_simulation_graph(batch_env, algo_cls, config):
    """Define the algortihm and environment interaction.

  Args:
    batch_env: In-graph environments object.
    algo_cls: Constructor of a batch algorithm.
    config: Configuration object for the algorithm.

  Returns:
    Object providing graph elements via attributes.
  """
    step = tf.Variable(0, False, dtype=tf.int32, name='global_step')
    is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
    should_log = tf.compat.v1.placeholder(tf.bool, name='should_log')
    do_report = tf.compat.v1.placeholder(tf.bool, name='do_report')
    force_reset = tf.compat.v1.placeholder(tf.bool, name='force_reset')
    algo = algo_cls(batch_env, step, is_training, should_log, config)
    done, score, summary = simulate(batch_env, algo, should_log, force_reset)
    message = 'Graph contains {} trainable variables.'
    tf.compat.v1.logging.info(message.format(count_weights()))
    return AttrDict(locals())


def define_batch_env(constructor, num_agents, env_processes):
    """Create environments and apply all desired wrappers.

  Args:
    constructor: Constructor of an OpenAI gym environment.
    num_agents: Number of environments to combine in the batch.
    env_processes: Whether to step environment in external processes.

  Returns:
    In-graph environments object.
  """
    with tf.compat.v1.variable_scope('environments'):
        if env_processes:
            envs = [wrappers.ExternalProcess(constructor) for _ in range(num_agents)]
        else:
            envs = [constructor() for _ in range(num_agents)]
        batch_env = BatchEnv(envs, blocking=not env_processes)
        batch_env = InGraphBatchEnv(batch_env)
    return batch_env


def define_saver(exclude=None):
    """Create a saver for the variables we want to checkpoint.

  Args:
    exclude: List of regexes to match variable names to exclude.

  Returns:
    Saver object.
  """
    variables = []
    exclude = exclude or []
    exclude = [re.compile(regex) for regex in exclude]
    for variable in tf.compat.v1.global_variables():
        if any(regex.match(variable.name) for regex in exclude):
            continue
        variables.append(variable)
    saver = tf.compat.v1.train.Saver(variables, keep_checkpoint_every_n_hours=5)
    return saver


def define_network(constructor, config, action_size):
    """Constructor for the recurrent cell for the algorithm.

  Args:
    constructor: Callable returning the network as RNNCell.
    config: Object providing configurations via attributes.
    action_size: Integer indicating the amount of action dimensions.

  Returns:
    Created recurrent cell object.
  """
    mean_weights_initializer = (tf.contrib.layers.variance_scaling_initializer(
        factor=config.init_mean_factor))
    logstd_initializer = tf.random_normal_initializer(config.init_logstd, 1e-10)
    network = constructor(config.policy_layers,
                          config.value_layers,
                          action_size,
                          mean_weights_initializer=mean_weights_initializer,
                          logstd_initializer=logstd_initializer)
    return network


def initialize_variables(sess, saver, logdir, checkpoint=None, resume=None):
    """Initialize or restore variables from a checkpoint if available.

  Args:
    sess: Session to initialize variables in.
    saver: Saver to restore variables.
    logdir: Directory to search for checkpoints.
    checkpoint: Specify what checkpoint name to use; defaults to most recent.
    resume: Whether to expect recovering a checkpoint or starting a new run.

  Raises:
    ValueError: If resume expected but no log directory specified.
    RuntimeError: If no resume expected but a checkpoint was found.
  """
    sess.run(tf.group(tf.compat.v1.local_variables_initializer(), tf.compat.v1.global_variables_initializer()))
    if resume and not (logdir or checkpoint):
        raise ValueError('Need to specify logdir to resume a checkpoint.')
    if logdir:
        state = tf.train.get_checkpoint_state(logdir)
        if checkpoint:
            checkpoint = os.path.join(logdir, checkpoint)
        if not checkpoint and state and state.model_checkpoint_path:
            checkpoint = state.model_checkpoint_path
        if checkpoint and resume is False:
            message = 'Found unexpected checkpoint when starting a new run.'
            raise RuntimeError(message)
        if checkpoint:
            saver.restore(sess, checkpoint)


def save_config(config, logdir=None):
    """Save a new configuration by name.

  If a logging directory is specified, is will be created and the configuration
  will be stored there. Otherwise, a log message will be printed.

  Args:
    config: Configuration object.
    logdir: Location for writing summaries and checkpoints if specified.

  Returns:
    Configuration object.
  """
    if logdir:
        with config.unlocked:
            config.logdir = logdir
        message = 'Start a new run and write summaries and checkpoints to {}.'
        tf.compat.v1.logging.info(message.format(config.logdir))
        tf.io.gfile.makedirs(config.logdir)
        config_path = os.path.join(config.logdir, 'config.yaml')
        with tf.io.gfile.GFile(config_path, 'w') as file_:
            yaml.dump(config, file_, default_flow_style=False)
    else:
        message = ('Start a new run without storing summaries and checkpoints since no '
                   'logging directory was specified.')
        tf.logging.info(message)
    return config


def load_config(logdir):
    """Load a configuration from the log directory.

  Args:
    logdir: The logging directory containing the configuration file.

  Raises:
    IOError: The logging directory does not contain a configuration file.

  Returns:
    Configuration object.
  """
    config_path = logdir and os.path.join(logdir, 'config.yaml')
    if not config_path or not tf.io.gfile.exists(config_path):
        message = ('Cannot resume an existing run since the logging directory does not '
                   'contain a configuration file.')
        raise IOError(message)
    with tf.io.gfile.GFile(config_path, 'r') as file_:
        config = yaml.load(file_)
    message = 'Resume run and write summaries and checkpoints to {}.'
    tf.compat.v1.logging.info(message.format(config.logdir))
    return config


def set_up_logging():
    """Configure the TensorFlow logger."""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    logging.getLogger('tensorflow').propagate = False

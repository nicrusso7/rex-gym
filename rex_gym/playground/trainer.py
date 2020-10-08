r"""Script to render a training session."""
import datetime
import functools
import logging
import os
import platform

import gym
import tensorflow.compat.v1 as tf

from rex_gym.agents.tools import wrappers
from rex_gym.agents.scripts import configs, utility
from rex_gym.agents.tools.attr_dict import AttrDict
from rex_gym.agents.tools.loop import Loop
from rex_gym.util import flag_mapper


class Trainer:
    def __init__(self, env_id: str, args: dict, playground: bool, log_dir: str, agents_number, signal_type):
        self.args = args
        self.playground = playground
        self.log_dir = log_dir
        self.agents = agents_number
        self.signal_type = signal_type
        if self.signal_type:
            env_signal = self.signal_type
        else:
            env_signal = flag_mapper.DEFAULT_SIGNAL[env_id]
        self.env_id = f"{env_id}_{env_signal}"

    def _create_environment(self, config):
        """Constructor for an instance of the environment.

          Args:
            config: Object providing configurations via attributes.

          Returns:
            Wrapped OpenAI Gym environment.
        """
        if self.playground:
            self.args['render'] = True
            self.args['debug'] = True
        else:
            self.args['debug'] = False
        if self.signal_type:
            self.args['signal_type'] = self.signal_type
        env = gym.make(config.env, **self.args)
        if config.max_length:
            env = wrappers.LimitDuration(env, config.max_length)
        env = wrappers.RangeNormalize(env)
        env = wrappers.ClipAction(env)
        env = wrappers.ConvertTo32Bit(env)
        return env

    @staticmethod
    def _define_loop(graph, logdir, train_steps, eval_steps):
        """Create and configure a training loop with training and evaluation phases.

          Args:
            graph: Object providing graph elements via attributes.
            logdir: Log directory for storing checkpoints and summaries.
            train_steps: Number of training steps per epoch.
            eval_steps: Number of evaluation steps per epoch.

          Returns:
            Loop object.
        """
        loop = Loop(logdir, graph.step, graph.should_log, graph.do_report, graph.force_reset)
        loop.add_phase('train',
                       graph.done,
                       graph.score,
                       graph.summary,
                       train_steps,
                       report_every=None,
                       log_every=train_steps // 2,
                       checkpoint_every=None,
                       feed={graph.is_training: True})
        loop.add_phase('eval',
                       graph.done,
                       graph.score,
                       graph.summary,
                       eval_steps,
                       report_every=eval_steps,
                       log_every=eval_steps // 2,
                       checkpoint_every=10 * eval_steps,
                       feed={graph.is_training: False})
        return loop

    def _train(self, config, env_processes):
        """Training and evaluation entry point yielding scores.

          Resolves some configuration attributes, creates environments, graph, and
          training loop. By default, assigns all operations to the CPU.

          Args:
            config: Object providing configurations via attributes.
            env_processes: Whether to step environments in separate processes.

          Yields:
            Evaluation scores.
        """
        tf.reset_default_graph()
        with config.unlocked:
            config.network = functools.partial(utility.define_network, config.network, config)
            config.policy_optimizer = getattr(tf.train, config.policy_optimizer)
            config.value_optimizer = getattr(tf.train, config.value_optimizer)
        if config.update_every % config.num_agents:
            logging.warning('Number of agents should divide episodes per update.')
        with tf.device('/cpu:0'):
            agents = self.agents if self.agents is not None else config.num_agents
            num_agents = 1 if self.playground else agents
            batch_env = utility.define_batch_env(lambda: self._create_environment(config), num_agents, env_processes)
            graph = utility.define_simulation_graph(batch_env, config.algorithm, config)
            loop = self._define_loop(graph, config.logdir, config.update_every * config.max_length,
                                     config.eval_episodes * config.max_length)
            total_steps = int(config.steps / config.update_every * (config.update_every + config.eval_episodes))
        # Exclude episode related variables since the Python state of environments is
        # not checkpointed and thus new episodes start after resuming.
        saver = utility.define_saver(exclude=(r'.*_temporary/.*',))
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            utility.initialize_variables(sess, saver, config.logdir)
            for score in loop.run(sess, saver, total_steps):
                yield score
        batch_env.close()

    def start_training(self):
        """Create configuration and launch the trainer."""
        utility.set_up_logging()
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        full_logdir = os.path.expanduser(os.path.join(self.log_dir, '{}-{}'.format(timestamp, self.env_id)))
        config = AttrDict(getattr(configs, self.env_id)())
        config = utility.save_config(config, full_logdir)
        os_name = platform.system()
        enable_processes = False if os_name == 'Windows' else True
        for score in self._train(config, enable_processes):
            tf.logging.info('Score {}.'.format(score))

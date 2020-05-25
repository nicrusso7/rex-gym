import click

from rex_gym.util import action_mapper


@click.group()
def cli():
    pass


@cli.command()
@click.option('--env', '-e', required=True, help="The Environment name.",
              type=click.Choice([e for e in action_mapper.ENV_ID_TO_ENV_NAMES.keys()], case_sensitive=True))
@click.option('--arg', '-a', type=(str, int), help="The Environment's arg(s).", multiple=True)
def policy(env, arg):
    from rex_gym.playground.policy_player import PolicyPlayer
    args = {}
    for k, v in arg:
        args[k] = v
    PolicyPlayer(env, args).play()


@cli.command()
@click.option('--env', '-e', required=True, help="The Environment name.",
              type=click.Choice([e for e in action_mapper.ENV_ID_TO_ENV_NAMES.keys()], case_sensitive=True))
@click.option('--arg', '-a', type=(str, int), help="The Environment's arg(s).", multiple=True)
@click.option('--log-dir', '-l', required=True, help="The path where the log directory will be created.")
@click.option('--playground', '-p', type=bool, default=False, help="Playground training: 1 agent, render enabled.")
@click.option('--agents-number', '-n', type=int, default=None, help=" Set the number of parallel agents.")
def train(env, arg, log_dir, playground, agent_number):
    from rex_gym.playground.trainer import Trainer
    args = {}
    for k, v in arg:
        args[k] = v
    Trainer(env, args, playground, log_dir, agent_number).start_training()

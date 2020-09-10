import click

from rex_gym.util import flag_mapper


@click.group()
def cli():
    pass


@cli.command()
@click.option('--env', '-e', required=True, help="The Environment name.",
              type=click.Choice([e for e in flag_mapper.ENV_ID_TO_ENV_NAMES.keys()], case_sensitive=True))
@click.option('--arg', '-a', type=(str, float), help="The Environment's arg(s).", multiple=True)
@click.option('--flag', '-f', type=(str, bool), help="The Environment's flag(s).", multiple=True)
@click.option('--open-loop', '-ol', is_flag=True, help="Use the open loop controller")
@click.option('--inverse-kinematics', '-ik', is_flag=True, help="Use the inverse kinematics controller")
@click.option('--terrain', '-t', default="plane", help="Set the simulation terrain.",
              type=click.Choice([t for t in flag_mapper.TERRAIN_TYPE.keys()]))
def policy(env, arg, flag, open_loop, inverse_kinematics, terrain):
    from rex_gym.playground.policy_player import PolicyPlayer
    terrain_args = _parse_terrain(terrain)
    args = _parse_args(arg + flag)
    args.update(terrain_args)
    signal_type = _parse_signal_type(open_loop, inverse_kinematics)
    PolicyPlayer(env, args, signal_type).play()


@cli.command()
@click.option('--env', '-e', required=True, help="The Environment name.",
              type=click.Choice([e for e in flag_mapper.ENV_ID_TO_ENV_NAMES.keys()], case_sensitive=True))
@click.option('--arg', '-a', type=(str, float), help="The Environment's arg(s).", multiple=True)
@click.option('--flag', '-f', type=(str, bool), help="The Environment's flag(s).", multiple=True)
@click.option('--log-dir', '-log', '-l', required=True, help="The path where the log directory will be created.")
@click.option('--playground', '-p', type=bool, default=False, help="Playground training: 1 agent, render enabled.")
@click.option('--agents-number', '-n', type=int, default=None, help="Set the number of parallel agents.")
@click.option('--open-loop', '-ol', is_flag=True, help="Use the open loop controller")
@click.option('--inverse-kinematics', '-ik', is_flag=True, help="Use the inverse kinematics controller")
@click.option('--terrain', '-t', default="plane", help="Set the simulation terrain.",
              type=click.Choice([t for t in flag_mapper.TERRAIN_TYPE.keys()]))
def train(env, arg, flag, log_dir, playground, agents_number, open_loop, inverse_kinematics, terrain):
    from rex_gym.playground.trainer import Trainer
    terrain_args = _parse_terrain(terrain)
    args = _parse_args(arg + flag)
    args.update(terrain_args)
    signal_type = _parse_signal_type(open_loop, inverse_kinematics)
    Trainer(env, args, playground, log_dir, agents_number, signal_type).start_training()


def _parse_terrain(terrain_id):
    arg = {
        "terrain_type": flag_mapper.TERRAIN_TYPE[terrain_id],
        "terrain_id": terrain_id
    }
    return arg


def _parse_args(params):
    args = {}
    for k, v in params:
        args[k] = v
    return args


def _parse_signal_type(open_loop, inverse_kinematics):
    if open_loop:
        return 'ol'
    elif inverse_kinematics:
        return 'ik'
    return None

STATIC_ACTIONS_MAP = {
    'gallop': ('rex_gym/policies/galloping/balanced', 'model.ckpt-20000000'),
    'walk': ('rex_gym/policies/walking/alternating_legs', 'model.ckpt-16000000'),
    'standup': ('rex_gym/policies/standup', 'model.ckpt-10000000')
}

DYNAMIC_ACTIONS_MAP = {
    'turn': ('rex_gym/policies/turn', 'model.ckpt-16000000')
}

ACTIONS_TO_ENV_NAMES = {
    'gallop': 'RexReactiveEnv',
    'walk': 'RexWalkEnv',
    'turn': 'RexTurnEnv',
    'standup': 'RexStandupEnv'
}

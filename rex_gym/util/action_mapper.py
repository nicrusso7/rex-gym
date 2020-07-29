ENV_ID_TO_POLICY = {
    'gallop_ol': ('rex_gym/policies/gallop/ol', 'model.ckpt-4000000'),
    'gallop_ik': ('rex_gym/policies/gallop/ik', 'model.ckpt-1500000'),
    'walk_ik': ('rex_gym/policies/walk/ik', 'model.ckpt-2000000'),
    'walk_ol': ('rex_gym/policies/walk/ol', 'model.ckpt-4000000'),
    'standup_ol': ('rex_gym/policies/standup/ol', 'model.ckpt-2000000'),
    'turn_ik': ('rex_gym/policies/turn/ik', 'model.ckpt-2000000'),
    'turn_ol': ('rex_gym/policies/turn/ol', 'model.ckpt-2000000')
}

ENV_ID_TO_ENV_NAMES = {
    'gallop': 'RexReactiveEnv',
    'walk': 'RexWalkEnv',
    'turn': 'RexTurnEnv',
    'standup': 'RexStandupEnv',
    'go': 'RexGoEnv',
    'poses': 'RexPosesEnv'
}

DEFAULT_SIGNAL = {
    'gallop': 'ik',
    'walk': 'ik',
    'turn': 'ol',
    'standup': 'ol',
    'go': 'ik',
    'poses': 'ol'
}

ACTION_MAP = {
    'run': ('rex_gym/policies/galloping/balanced', '20000000')
}


def fromAction(action):
    return ACTION_MAP[action]

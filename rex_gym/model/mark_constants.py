MARK_LIST = ['base', 'arm']

BASE_MOTOR_NAMES = [
    "motor_front_left_shoulder", "motor_front_left_leg", "foot_motor_front_left",
    "motor_front_right_shoulder", "motor_front_right_leg", "foot_motor_front_right",
    "motor_rear_left_shoulder", "motor_rear_left_leg", "foot_motor_rear_left",
    "motor_rear_right_shoulder", "motor_rear_right_leg", "foot_motor_rear_right"
]
ARM_MOTOR_NAMES = [
    "motor_arm_m1", "motor_arm_m2", "motor_arm_m3",
    "motor_arm_m4", "motor_arm_m5", "motor_arm_m6"
]

MARK_DETAILS = {
    'motors_num': {
        'base': 12,
        'arm': 18
    },
    'motors_names': {
        'base': BASE_MOTOR_NAMES,
        'arm': BASE_MOTOR_NAMES + ARM_MOTOR_NAMES
    },
    'urdf_name': {
        'base': 'rex.urdf',
        'arm': 'rex_arm.urdf'
    }
}

import numpy as np


class Kinematics:
    def __init__(self):
        self._l = 0.23
        self._w = 0.075
        self._hip = 0.055
        self._leg = 0.10652
        self._foot = 0.145
        self.y_dist = 0.185
        self.x_dist = self._l
        self.height = 0.2
        # frame vectors
        self._hip_front_right_v = np.array([self._l / 2, -self._w / 2, 0])
        self._hip_front_left_v = np.array([self._l / 2, self._w / 2, 0])
        self._hip_rear_right_v = np.array([-self._l / 2, -self._w / 2, 0])
        self._hip_rear_left_v = np.array([-self._l / 2, self._w / 2, 0])
        self._foot_front_right_v = np.array([self.x_dist / 2, -self.y_dist / 2, -self.height])
        self._foot_front_left_v = np.array([self.x_dist / 2, self.y_dist / 2, -self.height])
        self._foot_rear_right_v = np.array([-self.x_dist / 2, -self.y_dist / 2, -self.height])
        self._foot_rear_left_v = np.array([-self.x_dist / 2, self.y_dist / 2, -self.height])
        self._frames = np.asmatrix([[self.x_dist / 2, -self.y_dist / 2, -self.height],
                                    [self.x_dist / 2, self.y_dist / 2, -self.height],
                                    [-self.x_dist / 2, -self.y_dist / 2, -self.height],
                                    [-self.x_dist / 2, self.y_dist / 2, -self.height]])

    @staticmethod
    def get_Rx(x):
        return np.asmatrix([[1, 0, 0, 0],
                            [0, np.cos(x), -np.sin(x), 0],
                            [0, np.sin(x), np.cos(x), 0],
                            [0, 0, 0, 1]])

    @staticmethod
    def get_Ry(y):
        return np.asmatrix([[np.cos(y), 0, np.sin(y), 0],
                            [0, 1, 0, 0],
                            [-np.sin(y), 0, np.cos(y), 0],
                            [0, 0, 0, 1]])

    @staticmethod
    def get_Rz(z):
        return np.asmatrix([[np.cos(z), -np.sin(z), 0, 0],
                            [np.sin(z), np.cos(z), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

    def get_Rxyz(self, x, y, z):
        if x != 0 or y != 0 or z != 0:
            R = self.get_Rx(x) * self.get_Ry(y) * self.get_Rz(z)
            return R
        else:
            return np.identity(4)

    def get_RT(self, orientation, position):
        roll = orientation[0]
        pitch = orientation[1]
        yaw = orientation[2]
        x0 = position[0]
        y0 = position[1]
        z0 = position[2]

        translation = np.asmatrix([[1, 0, 0, x0],
                                   [0, 1, 0, y0],
                                   [0, 0, 1, z0],
                                   [0, 0, 0, 1]])
        rotation = self.get_Rxyz(roll, pitch, yaw)
        return rotation * translation

    def transform(self, coord, rotation, translation):
        vector = np.array([[coord[0]],
                           [coord[1]],
                           [coord[2]],
                           [1]])

        transform_vector = self.get_RT(rotation, translation) * vector
        return np.array([transform_vector[0, 0], transform_vector[1, 0], transform_vector[2, 0]])

    @staticmethod
    def check_domain(domain):
        if domain > 1 or domain < -1:
            if domain > 1:
                domain = 0.99
            else:
                domain = -0.99
        return domain

    def _solve_IK(self, coord, hip, leg, foot, right_side):
        domain = (coord[1] ** 2 + (-coord[2]) ** 2 - hip ** 2 + (-coord[0]) ** 2 - leg ** 2 - foot ** 2) / (2 * foot * leg)
        domain = self.check_domain(domain)
        gamma = np.arctan2(-np.sqrt(1 - domain ** 2), domain)
        sqrt_value = coord[1] ** 2 + (-coord[2]) ** 2 - hip ** 2
        if sqrt_value < 0.0:
            sqrt_value = 0.0
        alpha = np.arctan2(-coord[0], np.sqrt(sqrt_value)) - np.arctan2(foot * np.sin(gamma), leg + foot * np.cos(gamma))
        hip_val = hip
        if right_side:
            hip_val = -hip
        theta = -np.arctan2(coord[2], coord[1]) - np.arctan2(np.sqrt(sqrt_value), hip_val)
        angles = np.array([theta, -alpha, -gamma])
        return angles

    def solve(self, orientation, position, frames=None):
        if frames is not None:
            self._frames = frames
        foot_front_right = np.asarray([self._frames[0, 0], self._frames[0, 1], self._frames[0, 2]])
        foot_front_left = np.asarray([self._frames[1, 0], self._frames[1, 1], self._frames[1, 2]])
        foot_rear_right = np.asarray([self._frames[2, 0], self._frames[2, 1], self._frames[2, 2]])
        foot_rear_left = np.asarray([self._frames[3, 0], self._frames[3, 1], self._frames[3, 2]])
        # rotation vertices
        hip_front_right_vertex = self.transform(self._hip_front_right_v, orientation, position)
        hip_front_left_vertex = self.transform(self._hip_front_left_v, orientation, position)
        hip_rear_right_vertex = self.transform(self._hip_rear_right_v, orientation, position)
        hip_rear_left_vertex = self.transform(self._hip_rear_left_v, orientation, position)
        # leg vectors
        front_right_coord = foot_front_right - hip_front_right_vertex
        front_left_coord = foot_front_left - hip_front_left_vertex
        rear_right_coord = foot_rear_right - hip_rear_right_vertex
        rear_left_coord = foot_rear_left - hip_rear_left_vertex
        # leg vectors transformation
        inv_orientation = -orientation
        inv_position = -position
        t_front_right_coord = self.transform(front_right_coord, inv_orientation, inv_position)
        t_front_left_coord = self.transform(front_left_coord, inv_orientation, inv_position)
        t_rear_right_coord = self.transform(rear_right_coord, inv_orientation, inv_position)
        t_rear_left_coord = self.transform(rear_left_coord, inv_orientation, inv_position)
        # solve IK
        front_right_angles = self._solve_IK(t_front_right_coord, self._hip, self._leg, self._foot, True)
        front_left_angles = self._solve_IK(t_front_left_coord, self._hip, self._leg, self._foot, False)
        rear_right_angles = self._solve_IK(t_rear_right_coord, self._hip, self._leg, self._foot, True)
        rear_left_angles = self._solve_IK(t_rear_left_coord, self._hip, self._leg, self._foot, False)

        t_front_right = hip_front_right_vertex + t_front_right_coord
        t_front_left = hip_front_left_vertex + t_front_left_coord
        t_rear_right = hip_rear_right_vertex + t_rear_right_coord
        t_rear_left = hip_rear_left_vertex + t_rear_left_coord
        t_frames = np.asmatrix([[t_front_right[0], t_front_right[1], t_front_right[2]],
                                [t_front_left[0], t_front_left[1], t_front_left[2]],
                                [t_rear_right[0], t_rear_right[1], t_rear_right[2]],
                                [t_rear_left[0], t_rear_left[1], t_rear_left[2]]])
        return front_right_angles, front_left_angles, rear_right_angles, rear_left_angles, t_frames

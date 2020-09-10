import time
import numpy as np

from rex_gym.model.kinematics import Kinematics


class GaitPlanner:
    def __init__(self, mode):
        self._frame = np.zeros([4, 3])
        self._phi = 0.
        self._phi_stance = 0.
        self._last_time = 0.
        self._alpha = 0.
        self._s = False
        if mode == "walk":
            self._offset = np.array([0., 0.5, 0.5, 0.])
            self.step_offset = 0.5
        else:
            self._offset = np.array([0., 0., 0.8, 0.8])
            self.step_offset = 0.5

    @staticmethod
    def solve_bin_factor(n, k):
        return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))

    def bezier_curve(self, t, k, point):
        n = 11
        return point * self.solve_bin_factor(n, k) * np.power(t, k) * np.power(1 - t, n - k)

    @staticmethod
    def calculate_stance(phi_st, v, angle):
        c = np.cos(np.deg2rad(angle))
        s = np.sin(np.deg2rad(angle))
        A = 0.001
        half_l = 0.05
        p_stance = half_l * (1 - 2 * phi_st)
        stance_x = c * p_stance * np.abs(v)
        stance_y = -s * p_stance * np.abs(v)
        stance_z = -A * np.cos(np.pi / (2 * half_l) * p_stance)
        return stance_x, stance_y, stance_z

    def calculate_bezier_swing(self, phi_sw, v, angle, direction):
        c = np.cos(np.deg2rad(angle))
        s = np.sin(np.deg2rad(angle))
        X = np.abs(v) * c * np.array([-0.04, -0.056, -0.06, -0.06, -0.06, 0.,
                                      0., 0., 0.06, 0.06, 0.056, 0.04]) * direction
        Y = np.abs(v) * s * (-X)
        Z = np.abs(v) * np.array([0., 0., 0.0405, 0.0405, 0.0405, 0.0405,
                                  0.0405, 0.0495, 0.0495, 0.0495, 0., 0.])
        swing_x = 0.
        swing_y = 0.
        swing_z = 0.
        # TODO Use all 12 points
        for i in range(10):
            swing_x = swing_x + self.bezier_curve(phi_sw, i, X[i])
            swing_y = swing_y + self.bezier_curve(phi_sw, i, Y[i])
            swing_z = swing_z + self.bezier_curve(phi_sw, i, Z[i])
        return swing_x, swing_y, swing_z

    def step_trajectory(self, phi, v, angle, w_rot, center_to_foot, direction):
        if phi >= 1:
            phi = phi - 1.
        r = np.sqrt(center_to_foot[0] ** 2 + center_to_foot[1] ** 2)
        foot_angle = np.arctan2(center_to_foot[1], center_to_foot[0])
        if w_rot >= 0.:
            circle_trajectory = 90. - np.rad2deg(foot_angle - self._alpha)
        else:
            circle_trajectory = 270. - np.rad2deg(foot_angle - self._alpha)

        if phi <= self.step_offset:
            # stance phase
            phi_stance = phi / self.step_offset
            stepX_long, stepY_long, stepZ_long = self.calculate_stance(phi_stance, v, angle)
            stepX_rot, stepY_rot, stepZ_rot = self.calculate_stance(phi_stance, w_rot, circle_trajectory)
        else:
            # swing phase
            phiSwing = (phi - self.step_offset) / (1 - self.step_offset)
            stepX_long, stepY_long, stepZ_long = self.calculate_bezier_swing(phiSwing, v, angle, direction)
            stepX_rot, stepY_rot, stepZ_rot = self.calculate_bezier_swing(phiSwing, w_rot, circle_trajectory, direction)
        if center_to_foot[1] > 0:
            if stepX_rot < 0:
                self._alpha = -np.arctan2(np.sqrt(stepX_rot ** 2 + stepY_rot ** 2), r)
            else:
                self._alpha = np.arctan2(np.sqrt(stepX_rot ** 2 + stepY_rot ** 2), r)
        else:
            if stepX_rot < 0:
                self._alpha = np.arctan2(np.sqrt(stepX_rot ** 2 + stepY_rot ** 2), r)
            else:
                self._alpha = -np.arctan2(np.sqrt(stepX_rot ** 2 + stepY_rot ** 2), r)
        coord = np.empty(3)
        coord[0] = stepX_long + stepX_rot
        coord[1] = stepY_long + stepY_rot
        coord[2] = stepZ_long + stepZ_rot
        return coord

    def loop(self, v, angle, w_rot, t, direction, frames=None):
        if frames is None:
            k_obj = Kinematics()
            x_dist = k_obj.x_dist
            y_dist = k_obj.y_dist
            height = k_obj.height
            frames = np.asmatrix([[x_dist / 2, -y_dist / 2, -height],
                                  [x_dist / 2, y_dist / 2, -height],
                                  [-x_dist / 2, -y_dist / 2, -height],
                                  [-x_dist / 2, y_dist / 2, -height]])
        if t <= 0.01:
            t = 0.01
        if self._phi >= 0.99:
            self._last_time = time.time()
        self._phi = (time.time() - self._last_time) / t
        step_coord = self.step_trajectory(self._phi + self._offset[0], v, angle, w_rot,
                                          np.squeeze(np.asarray(frames[0, :])), direction)  # FR
        self._frame[0, 0] = frames[0, 0] + step_coord[0]
        self._frame[0, 1] = frames[0, 1] + step_coord[1]
        self._frame[0, 2] = frames[0, 2] + step_coord[2]

        step_coord = self.step_trajectory(self._phi + self._offset[1], v, angle, w_rot,
                                          np.squeeze(np.asarray(frames[1, :])), direction)  # FL
        self._frame[1, 0] = frames[1, 0] + step_coord[0]
        self._frame[1, 1] = frames[1, 1] + step_coord[1]
        self._frame[1, 2] = frames[1, 2] + step_coord[2]

        step_coord = self.step_trajectory(self._phi + self._offset[2], v, angle, w_rot,
                                          np.squeeze(np.asarray(frames[2, :])), direction)  # BR
        self._frame[2, 0] = frames[2, 0] + step_coord[0]
        self._frame[2, 1] = frames[2, 1] + step_coord[1]
        self._frame[2, 2] = frames[2, 2] + step_coord[2]

        step_coord = self.step_trajectory(self._phi + self._offset[3], v, angle, w_rot,
                                          np.squeeze(np.asarray(frames[3, :])), direction)  # BL
        self._frame[3, 0] = frames[3, 0] + step_coord[0]
        self._frame[3, 1] = frames[3, 1] + step_coord[1]
        self._frame[3, 2] = frames[3, 2] + step_coord[2]
        return self._frame

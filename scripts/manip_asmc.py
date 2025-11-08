#!/usr/bin/env python3
import numpy as np

# CRITICAL FIX: This class should NOT be a ROS Node. It's a plain Python object.
class ManipulatorControllerLogic:
    def __init__(self, n_joints=3):
        self.n = n_joints
        self.max_effort = 10.0
        
        # Controller state variables
        self.des_vel = np.zeros(self.n)
        self.des_acc = np.zeros(self.n)
      
        # Gains
        self.Lam_m = np.array([5.0, 5.0, 2.0])
        self.Phi_m = np.array([10.0, 10.0, 0.2])
        self.M_bar = 0.45
        self.Kp0_m = np.array([0.1, 0.1, 0.1])
        self.alpha_m = np.array([1.0, 1.0, 1.0])
        self.v_m = 0.1

    def des_eff(self, curr_pos, des_pos, curr_vel, arm_currN, dt, active_joints=None):
        """Calculates the desired effort for the manipulator joints."""
        if active_joints is None:
            active_joints = [0, 1] # Default to controlling the first two joints

        # Input validation for dt
        dt = min(dt, 0.04)

        errPos = curr_pos - des_pos
        errVel = curr_vel - self.des_vel

        sv = errVel + np.multiply(self.Phi_m, errPos)
        
        # Update adaptive gain
        self.Kp0_m = (np.linalg.norm(sv) - np.multiply(self.alpha_m, self.Kp0_m)) * dt
        self.Kp0_m = np.maximum(self.Kp0_m, 0.0001 * np.ones(self.n))

        Rho = self.Kp0_m 

        delTau = np.zeros(self.n)
        for i in range(self.n):
            delTau[i] = Rho[i] * self.sigmoid(sv[i], self.v_m)

        des_a = (
            -np.multiply(self.Lam_m, sv)
            + self.M_bar * (self.des_acc - np.multiply(self.Phi_m, errVel))
            + arm_currN
            - delTau
        )

        efforts = np.zeros(self.n)
        for i in active_joints:
            efforts[i] = np.clip(des_a[i], -self.max_effort, self.max_effort)

        return efforts

    def sigmoid(self, s, v):
        if np.absolute(s) > v:
            return np.sign(s)
        else:
            return s / v
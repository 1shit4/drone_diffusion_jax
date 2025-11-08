#!/usr/bin/env python3
import numpy as np
import tf_transformations as transformations
from tf_transformations import *

class DroneControllerLogic:
    def __init__(self):
        # --- Controller Gains and Parameters (from original script) ---
        self.Lam = np.array([2.0, 2.0, 4.5])
        self.Phi = np.array([1.5, 1.5, 1.2])
        self.M_bar = 0.45
        self.Kp0 = np.array([1.0, 1.0, 1.0])
        self.alpha_0 = np.array([1, 1, 1])
        self.v = 0.1
        self.norm_thrust_const = 0.06
        self.max_th = 12.0
        self.max_throttle = 0.96
        self.gravity = np.array([0, 0, 9.8])
        self.desVel = np.zeros(3) # Assuming desired velocity is always zero
        self.desAcc = np.zeros(3) # Assuming desired acceleration is always zero
        
    def sigmoid(self, s, v):
        if np.absolute(s) > v:
            return s / np.absolute(s)
        else:
            return s / v


    def a_des(self, curPos, curVel, desPos, drone_currN, dt, armed):
        """Calculates the desired acceleration vector."""
        # Input validation for dt
        if dt > 0.01:
            dt = 0.01

        errPos = curPos - desPos
        errVel = curVel - self.desVel
        sv = errVel + np.multiply(self.Phi, errPos)
        
        if armed:
            self.Kp0 = (np.linalg.norm(sv) - np.multiply(self.alpha_0, self.Kp0)) * dt
            self.Kp0 = np.maximum(self.Kp0, 0.0001 * np.ones(3))
            
        Rho = self.Kp0
        delTau = np.zeros(3)
        for i in range(3):
            delTau[i] = Rho[i] * self.sigmoid(sv[i], self.v)
        
        des_a = (
            -np.multiply(self.Lam, sv)
            + self.M_bar * (self.desAcc + self.gravity - np.multiply(self.Phi, errVel))
            + drone_currN
            - delTau
        )

        # Clamp desired acceleration to a maximum magnitude
        if np.linalg.norm(des_a) > self.max_th:
            des_a = (self.max_th / np.linalg.norm(des_a)) * des_a
            
        return des_a
        
    def geo_con(self, q_curr, des_a):
        """Geometric controller: converts desired acceleration to quaternion and thrust."""
        pose = transformations.quaternion_matrix(q_curr)  # 4*4 matrix
        pose_temp1 = np.delete(pose, -1, axis=1)
        rot_curr = np.delete(pose_temp1, -1, axis=0)  # 3*3 current rotation matrix
        zb_curr = rot_curr[:, 2]
        thrust = self.norm_thrust_const * des_a.dot(zb_curr)
        collective_thrust = np.maximum(0.0, np.minimum(thrust, self.max_throttle))

        # Desired Euler orientation and Desired Rotation matrix
        rot_des = self.acc2quat(des_a, 0)  # desired yaw = 0
        rot_44 = np.vstack(
            (np.hstack((rot_des, np.array([[0, 0, 0]]).T)), np.array([[0, 0, 0, 1]]))
        )
        quat_des = quaternion_from_matrix(rot_44)

        return quat_des, collective_thrust

    def acc2quat(self, des_a, des_yaw):
        proj_xb_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        if np.linalg.norm(des_a) == 0.0:
            zb_des = np.array([0, 0, 1])
        else:
            zb_des = des_a / np.linalg.norm(des_a)
        # ensure cross products are safe (avoid zero division)
        cross_y = np.cross(zb_des, proj_xb_des)
        if np.linalg.norm(cross_y) == 0:
            # fallback orthogonal vector
            if abs(zb_des[2]) < 0.9:
                cross_y = np.cross(zb_des, np.array([0, 0, 1.0]))
            else:
                cross_y = np.cross(zb_des, np.array([0, 1.0, 0]))
        yb_des = cross_y / np.linalg.norm(cross_y)
        xb_des = np.cross(yb_des, zb_des)
        xb_des = xb_des / np.linalg.norm(xb_des)

        rotmat = np.transpose(np.array([xb_des, yb_des, zb_des]))
        return rotmat
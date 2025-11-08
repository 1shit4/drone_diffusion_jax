#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch # <-- Using PyTorch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

# ROS Message Imports
from sensor_msgs.msg import Imu, JointState
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory as Mdjt
from std_msgs.msg import Float32, Float64MultiArray

# Use SciPy for transforms, as it's a reliable pip package
from scipy.spatial.transform import Rotation as R

# Import the PyTorch version of the policy
from diffusion_pt.diffusion_inference import Policy


class Controller(Node):
    def __init__(self):
        super().__init__("setpoint_node")

        # --- System Definition for 2-DOF Drone+Manipulator ---
        self.OBS_DIM = 21
        self.ACTION_DIM = 5
        self.DRONE_ACTION_DIM = 3
        self.MANIPULATOR_ACTION_DIM = 2
        self.MANIPULATOR_JOINT_NAMES = ['arm_to_L1', 'L1_to_L2']
        
        # --- State Variables ---
        self.state, self.sp, self.imu = State(), PoseStamped(), Imu()
        self.sp.pose.position.x = 0.0; self.sp.pose.position.y = 0.0; self.sp.pose.position.z = 3.0
        self.local_pos, self.local_vel = PoseStamped(), TwistStamped()
        self.joint_states = DynamicJointState()
        self.desVel, self.desAcc = np.zeros(3), np.zeros(3)
        self.att_cmd = AttitudeTarget()
        self.pre_time = self.get_clock().now().nanoseconds * 1e-9
        self.armed = False
        self.last_tau = np.zeros(self.ACTION_DIM)
        self.currN = np.zeros(self.ACTION_DIM)
        
        # --- Controller Gains and Parameters (Faithful Replication) ---
        self.Lam = np.array([2.0, 2.0, 8.0])
        self.Phi = np.array([1.5, 1.5, 1.2]) # Your tuned value
        self.M_bar = 1.0 # Nominal mass from training
        self.Kp0, self.alpha_0, self.v = (np.array([0.1, 0.1, 0.1]), np.array([1, 1, 1]), 0.1,)
        
        # Physical parameters tuned for the heavier drone
        total_mass_in_sim = 2.456 # Replace with your calculated mass
        self.norm_thrust_const = 0.08 #0.056 * (total_mass_in_sim / 2.0)
        self.max_th = 12.0 # Increased from 12.0 for heavier system
        self.max_throttle = 1.0
        self.gravity = np.array([0, 0, 9.8])

        # --- Publishers ---
        self.att_pub = self.create_publisher(AttitudeTarget, "/mavros/setpoint_raw/attitude", 10)
        self.manip_correction_pub = self.create_publisher(Float64MultiArray, '/diffusion/manipulator_correction', 10)
        self.joint_ref_pub = self.create_publisher(Float64MultiArray, '/joint_reference', 10)

        # --- Subscriptions (Using the robust timer-based model) ---
        self.create_subscription(State, "/mavros/state", self.stateCb, 10)
        self.create_subscription(Imu, "/mavros/imu/data", self.accCB, qos_profile=qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.posCb, qos_profile=qos_profile_sensor_data)
        self.create_subscription(TwistStamped, "/mavros/local_position/velocity_local", self.velCb, qos_profile=qos_profile_sensor_data)
        self.create_subscription(Mdjt, "/command/trajectory", self.multiDoFCb, 10)
        self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_cb, qos_profile=qos_profile_sensor_data)
        
        # --- Service Clients ---
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # --- Policy Loading (PyTorch) ---
        self.get_logger().info("Loading DRONE+MANIPULATOR diffusion policy (PyTorch)...")
        self.policy = Policy(config_path="config/experiment_pt.yaml", use_ema=True, device='cpu')
        self.get_logger().info("Policy loaded. Warming up...")
        warmup_input = np.ones((self.OBS_DIM,), dtype=np.float32)
        with torch.no_grad():
            _ = self.policy(warmup_input)
        self.get_logger().info("Policy is ready.")
        
        # --- Timers for ROS2 Execution Model ---
        self.control_loop_timer = self.create_timer(1/30.0, self.run_control_loop); self.control_loop_timer.cancel()
        self.startup_timer = self.create_timer(1.0, self.run_startup_sequence)
        self.main_loop_counter = 0

    # --- Callbacks ---
    def stateCb(self, msg: State): self.state = msg
    def accCB(self, msg: Imu): self.imu = msg
    def posCb(self, msg: PoseStamped): self.local_pos = msg
    def velCb(self, msg: TwistStamped): self.local_vel = msg
    def joint_state_cb(self, msg: DynamicJointState): self.joint_states = msg
    def multiDoFCb(self, msg: Mdjt):
        if msg.points and msg.points[0].transforms: 
            pt = msg.points[0]
            self.sp.pose.position.x = pt.transforms[0].translation.x
            self.sp.pose.position.y = pt.transforms[0].translation.y
            self.sp.pose.position.z = pt.transforms[0].translation.z
        
    # --- Control Logic (Faithful Replica) ---
    def vector2Arrays(self, vector): return np.array([vector.x, vector.y, vector.z])
    def get_action(self, curr_act): self.currN = curr_act
    def sigmoid(self, s, v): return s / np.absolute(s) if np.absolute(s) > v else s / v

    def get_obs(self):
        curPos = self.vector2Arrays(self.local_pos.pose.position)
        curVel = self.vector2Arrays(self.local_vel.twist.linear)
        curAcc = self.vector2Arrays(self.imu.linear_acceleration) - self.gravity
        q = self.local_pos.pose.orientation
        (roll_curr, pitch_curr, yaw_curr) = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')
        
        try:
            joint_indices = [self.joint_states.joint_names.index(name) for name in self.MANIPULATOR_JOINT_NAMES]
            j_pos = np.array([self.joint_states.interface_values[0].values[i] for i in joint_indices])
            j_vel = np.array([self.joint_states.interface_values[1].values[i] for i in joint_indices])
        except (ValueError, IndexError, AttributeError):
            j_pos = np.zeros(self.MANIPULATOR_ACTION_DIM)
            j_vel = np.zeros(self.MANIPULATOR_ACTION_DIM)
            
        obs = np.concatenate([curPos, curVel, curAcc, np.array([roll_curr, pitch_curr, yaw_curr]), j_pos, j_vel, self.last_tau]).astype(np.float32)
        return obs, curAcc

    def a_des(self):
        _, accel = self.get_obs()
        dt = self.get_clock().now().nanoseconds * 1e-9 - self.pre_time; self.pre_time = self.pre_time + dt
        if dt > 0.01: dt = 0.01
        curPos = self.vector2Arrays(self.local_pos.pose.position); desPos = self.vector2Arrays(self.sp.pose.position)
        curVel = self.vector2Arrays(self.local_vel.twist.linear)
        errPos = curPos - desPos; errVel = curVel - self.desVel
        sv = errVel + np.multiply(self.Phi, errPos)
        
        if self.armed:
            self.Kp0 = (np.linalg.norm(sv) - np.multiply(self.alpha_0, self.Kp0)) * dt
            self.Kp0 = np.maximum(self.Kp0, 0.0001 * np.ones(3))
            
        Rho = self.Kp0; delTau = np.zeros(3)
        for i in range(3): delTau[i] = Rho[i] * self.sigmoid(sv[i], self.v)
        
        drone_currN = self.currN[:self.DRONE_ACTION_DIM]
        
        # Replicating the exact control law from the JAX base script
        des_a = (
            -np.multiply(self.Lam, sv)
            + self.M_bar * (self.desAcc + self.gravity - np.multiply(self.Phi, errVel))
            + drone_currN
            - delTau
        )

        if np.linalg.norm(des_a) > self.max_th:
            des_a = (self.max_th / np.linalg.norm(des_a)) * des_a
            
        return des_a
        
    def geo_con_(self):
        des_a = self.a_des()
        q = self.local_pos.pose.orientation
        rot_curr = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        zb_curr = rot_curr[:, 2]
        thrust = self.norm_thrust_const * des_a.dot(zb_curr)
        self.collective_thrust = np.clip(thrust, 0.0, self.max_throttle)
        rot_des = self.acc2quat(des_a, 0)
        quat_des = R.from_matrix(rot_des).as_quat()
        now = self.get_clock().now().to_msg()
        self.att_cmd.header.stamp = now
        self.att_cmd.orientation.x, self.att_cmd.orientation.y, self.att_cmd.orientation.z, self.att_cmd.orientation.w = quat_des
        self.att_cmd.thrust = float(self.collective_thrust)
        self.att_cmd.type_mask = 7

    def acc2quat(self, des_a, des_yaw):
        proj_xb_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        if np.linalg.norm(des_a) == 0.0: zb_des = np.array([0, 0, 1])
        else: zb_des = des_a / np.linalg.norm(des_a)
        cross_y = np.cross(zb_des, proj_xb_des)
        if np.linalg.norm(cross_y) < 1e-6:
            if abs(zb_des[2]) < 0.9: cross_y = np.cross(zb_des, np.array([0, 0, 1.0]))
            else: cross_y = np.cross(zb_des, np.array([0, 1.0, 0]))
        yb_des = cross_y / np.linalg.norm(cross_y)
        xb_des = np.cross(yb_des, zb_des); xb_des = xb_des / np.linalg.norm(xb_des)
        return np.transpose(np.array([xb_des, yb_des, zb_des]))

    def pub_att(self): self.geo_con_(); self.att_pub.publish(self.att_cmd)

    # Startup and Main Loop using the robust timer-based model
    def run_startup_sequence(self):
        self.startup_timer.cancel()
        while not self.arming_client.wait_for_service(timeout_sec=1.0): self.get_logger().info('Arming service not available, waiting...')
        self.get_logger().info("ARMING"); arm_req = CommandBool.Request(); arm_req.value = True
        self.arming_client.call_async(arm_req)
        self.streaming_timer = self.create_timer(1/30.0, self.stream_setpoints_callback); self.setpoint_stream_count = 0
    def stream_setpoints_callback(self):
        if not self.armed:
            if self.state.armed: self.armed = True; self.get_logger().info("Arming confirmed.")
            else: self.get_logger().info("Waiting for arming confirmation...")
        self.pub_att()
        self.setpoint_stream_count += 1
        if self.setpoint_stream_count > 100:
            self.streaming_timer.cancel()
            while not self.set_mode_client.wait_for_service(timeout_sec=1.0): self.get_logger().info('Set mode service not available, waiting...')
            self.get_logger().info("Setting OFFBOARD mode..."); mode_req = SetMode.Request(); mode_req.custom_mode = "OFFBOARD"
            self.set_mode_client.call_async(mode_req)
            self.control_loop_timer.reset(); self.get_logger().info("Controller is now in main loop.")
    
    def run_control_loop(self):
        PLANNING_BUDGET = 8
        if self.main_loop_counter % PLANNING_BUDGET == 0:
            obs_np, _ = self.get_obs()
            if obs_np.shape[0] == self.OBS_DIM:
                with torch.no_grad():
                    self.action_plan = self.policy(obs_np)
        
        if hasattr(self, 'action_plan'):
            action_idx = self.main_loop_counter % PLANNING_BUDGET
            full_5d_action = self.action_plan[action_idx]
            
            self.get_action(full_5d_action)
            self.pub_att() # Controls the drone
            
            # Publish manipulator correction
            h_manip = self.currN[self.DRONE_ACTION_DIM:]
            correction_msg = Float64MultiArray(data=h_manip.tolist())
            self.manip_correction_pub.publish(correction_msg)
            
            # Publish manipulator reference
            try:
                joint_indices = [self.joint_states.joint_names.index(name) for name in self.MANIPULATOR_JOINT_NAMES]
                current_joint_pos = [self.joint_states.interface_values[0].values[i] for i in joint_indices]
                ref_msg = Float64MultiArray(data=current_joint_pos)
                self.joint_ref_pub.publish(ref_msg)
            except (ValueError, IndexError): pass
            
            self.last_tau = full_5d_action
            
        self.main_loop_counter += 1

def main(args=None):
    rclpy.init(args=args)
    controller_node = Controller()
    rclpy.spin(controller_node)
    controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
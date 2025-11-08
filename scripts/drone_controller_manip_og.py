#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

# ROS Message Imports
from sensor_msgs.msg import Imu, JointState
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, Vector3
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import CommandTOL, CommandBool, SetMode
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory as Mdjt
from std_msgs.msg import Float32, Float64MultiArray

# Math and Policy Imports
import transforms3d.euler as tfe
import transforms3d.quaternions as tfq
from diffusion.diffusion_inference import Policy


class fcuModes:
    # This class is unchanged from your base script
    def __init__(self, node: Node):
        self.node = node
        self._cbg = ReentrantCallbackGroup()

    def _call_service_sync(self, client, req):
        if not client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().warn(f"Service {client.srv_name} not available yet.")
            if not client.wait_for_service(timeout_sec=10.0):
                 raise RuntimeError(f"Service {client.srv_name} not available.")
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut)
        if fut.result() is None:
            raise RuntimeError(f"Service call {client.srv_name} failed.")
        return fut.result()

    def setArm(self):
        client = self.node.create_client(CommandBool, "mavros/cmd/arming", callback_group=self._cbg)
        req = CommandBool.Request()
        req.value = True
        try:
            self._call_service_sync(client, req)
        except Exception as e:
            self.node.get_logger().error(f"Service arming call failed: {e}")

    def setOffboardMode(self):
        client = self.node.create_client(SetMode, "mavros/set_mode", callback_group=self._cbg)
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"
        try:
            self._call_service_sync(client, req)
        except Exception as e:
            self.node.get_logger().error(f"service set_mode call failed: {e}. Offboard Mode could not be set.")


class Controller(Node):
    def __init__(self):
        super().__init__("setpoint_node")

        # --- System Definition ---
        self.OBS_DIM = 21
        self.ACTION_DIM = 5
        self.DRONE_ACTION_DIM = 3
        self.MANIPULATOR_ACTION_DIM = 2
        self.MANIPULATOR_JOINT_NAMES = ['arm_to_L1', 'L1_to_L2']
        
        # --- State Variables ---
        self.state = State()
        self.sp = PoseStamped()
        self.sp.pose.position.x = 0.0
        self.sp.pose.position.y = 0.0
        self.sp.pose.position.z = 3.0
        self.imu = Imu()
        self.local_pos = PoseStamped()
        self.local_vel = TwistStamped()
        self.joint_states = DynamicJointState()
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
        self.att_cmd = AttitudeTarget()
        self.pre_time = self.get_clock().now().nanoseconds * 1e-9
        self.armed = False
        self.last_tau = np.zeros(self.ACTION_DIM)
        self.currN = np.zeros(self.ACTION_DIM)
        
        # --- Controller Gains and Parameters (Faithful Replication) ---
        self.Lam = np.array([2.0, 2.0, 8.0])
        self.Phi = np.array([1.5, 1.5, 1.2])
        self.M_bar = 1.0
        self.Kp0, self.alpha_0, self.v = (np.array([0.1, 0.1, 0.1]), np.array([1, 1, 1]), 0.1,)
        self.norm_thrust_const =  0.08
        self.max_th = 12.0
        self.max_throttle = 0.96
        self.gravity = np.array([0, 0, 9.8])

        # --- Publishers ---
        self.att_pub = self.create_publisher(AttitudeTarget, "/mavros/setpoint_raw/attitude", 10)
        self.manip_correction_pub = self.create_publisher(Float64MultiArray, '/diffusion/manipulator_correction', 10)
        self.joint_ref_pub = self.create_publisher(Float64MultiArray, '/joint_reference', 10)

        # --- Policy Loading ---
        self.get_logger().info("Loading DRONE+MANIPULATOR diffusion policy...")
        self.policy = Policy(config="experiment_jax.yaml", use_ema=True, rng=42)
        self.get_logger().info("Policy loaded. Warming up...")
        warmup_input = jnp.ones((self.OBS_DIM,), dtype=jnp.float32)
        self.policy(warmup_input)
        self.get_logger().info("Policy is ready.")

    # --- Callbacks ---
    def posCb(self, msg: PoseStamped):
        self.local_pos = msg

    def velCb(self, msg: TwistStamped):
        self.local_vel = msg

    def stateCb(self, msg: State):
        self.state = msg

    def accCB(self, msg: Imu):
        self.imu = msg
        
    def joint_state_cb(self, msg: DynamicJointState):
        self.joint_states = msg

    def multiDoFCb(self, msg: Mdjt):
        if msg.points and msg.points[0].transforms: 
            pt = msg.points[0]
            self.sp.pose.position.x = pt.transforms[0].translation.x
            self.sp.pose.position.y = pt.transforms[0].translation.y
            self.sp.pose.position.z = pt.transforms[0].translation.z
        
    # --- Control Logic ---
    def vector2Arrays(self, vector):
        return np.array([vector.x, vector.y, vector.z])
        
    def get_action(self, curr_act):
        self.currN = curr_act
        
    def sigmoid(self, s, v):
        return s / np.absolute(s) if np.absolute(s) > v else s / v

    def get_obs(self):
        curPos = self.vector2Arrays(self.local_pos.pose.position)
        curVel = self.vector2Arrays(self.local_vel.twist.linear)
        curAcc = self.vector2Arrays(self.imu.linear_acceleration) - self.gravity
        q = self.local_pos.pose.orientation
        (roll_curr, pitch_curr, yaw_curr) = tfe.quat2euler([q.w, q.x, q.y, q.z], axes='sxyz')
        
        try:
            joint_indices = [self.joint_states.joint_names.index(name) for name in self.MANIPULATOR_JOINT_NAMES]
            j_pos = np.array([self.joint_states.interface_values[0].values[i] for i in joint_indices])
            j_vel = np.array([self.joint_states.interface_values[1].values[i] for i in joint_indices])
        except (ValueError, IndexError, AttributeError):
            j_pos = np.zeros(self.MANIPULATOR_ACTION_DIM)
            j_vel = np.zeros(self.MANIPULATOR_ACTION_DIM)
            
        obs = np.concatenate([
            curPos, 
            curVel, 
            curAcc, 
            np.array([roll_curr, pitch_curr, yaw_curr]), 
            j_pos, 
            j_vel, 
            self.last_tau
        ]).astype(np.float32)
        
        return obs, curAcc

    def a_des(self):
        _, accel = self.get_obs()
        
        dt = self.get_clock().now().nanoseconds * 1e-9 - self.pre_time
        self.pre_time = self.pre_time + dt
        if dt > 0.01:
            dt = 0.01

        curPos = self.vector2Arrays(self.local_pos.pose.position)
        desPos = self.vector2Arrays(self.sp.pose.position)
        curVel = self.vector2Arrays(self.local_vel.twist.linear)
        curAcc = self.vector2Arrays(self.imu.linear_acceleration) - self.gravity
        
        errPos = curPos - desPos
        errVel = curVel - self.desVel
        sv = errVel + np.multiply(self.Phi, errPos)
        
        if self.armed:
            self.Kp0 = (np.linalg.norm(sv) - np.multiply(self.alpha_0, self.Kp0)) * dt
            self.Kp0 = np.maximum(self.Kp0, 0.0001 * np.ones(3))
            
        Rho = self.Kp0
        delTau = np.zeros(3)
        for i in range(3):
            delTau[i] = Rho[i] * self.sigmoid(sv[i], self.v)
        
        drone_currN = self.currN[:self.DRONE_ACTION_DIM]
        
        des_a = (
            -np.multiply(self.Lam, sv)
            + self.M_bar * (self.desAcc + self.gravity - np.multiply(self.Phi, errVel))
            + drone_currN
            - delTau
        )

        #print(drone_currN/des_a)

        if np.linalg.norm(des_a) > self.max_th:
            des_a = (self.max_th / np.linalg.norm(des_a)) * des_a
            
        return des_a
        
    def geo_con_(self):
        des_a = self.a_des()
        
        q = self.local_pos.pose.orientation
        rot_curr = tfq.quat2mat([q.w, q.x, q.y, q.z])
        
        zb_curr = rot_curr[:, 2]
        thrust = self.norm_thrust_const * des_a.dot(zb_curr)
        self.collective_thrust = np.clip(thrust, 0.0, self.max_throttle)
        
        rot_des = self.acc2quat(des_a, 0)
        
        w, x, y, z = tfq.mat2quat(rot_des)
        quat_des = [x, y, z, w]

        now = self.get_clock().now().to_msg()
        self.att_cmd.header.stamp = now
        self.att_cmd.orientation.x = quat_des[0]
        self.att_cmd.orientation.y = quat_des[1]
        self.att_cmd.orientation.z = quat_des[2]
        self.att_cmd.orientation.w = quat_des[3]
        self.att_cmd.thrust = float(self.collective_thrust)
        self.att_cmd.type_mask = 7

    def acc2quat(self, des_a, des_yaw):
        proj_xb_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        
        if np.linalg.norm(des_a) == 0.0:
            zb_des = np.array([0, 0, 1])
        else:
            zb_des = des_a / np.linalg.norm(des_a)
            
        cross_y = np.cross(zb_des, proj_xb_des)
        if np.linalg.norm(cross_y) == 0:
            if abs(zb_des[2]) < 0.9:
                cross_y = np.cross(zb_des, np.array([0, 0, 1.0]))
            else:
                cross_y = np.cross(zb_des, np.array([0, 1.0, 0]))
                
        yb_des = cross_y / np.linalg.norm(cross_y)
        xb_des = np.cross(yb_des, zb_des)
        xb_des = xb_des / np.linalg.norm(xb_des)

        rotmat = np.transpose(np.array([xb_des, yb_des, zb_des]))
        return rotmat

    def pub_att(self):
        self.geo_con_()
        self.att_pub.publish(self.att_cmd)


def main(argv):
    rclpy.init(args=argv)
    
    node = Controller()
    modes = fcuModes(node)
    
    # --- Subscriptions ---
    qos = qos_profile_sensor_data
    node.create_subscription(State, "/mavros/state", node.stateCb, 10)
    node.create_subscription(Imu, "/mavros/imu/data", node.accCB, qos)
    node.create_subscription(PoseStamped, "/mavros/local_position/pose", node.posCb, qos)
    node.create_subscription(TwistStamped, "/mavros/local_position/velocity_local", node.velCb, qos)
    node.create_subscription(Mdjt, "/command/trajectory", node.multiDoFCb, 10)
    node.create_subscription(DynamicJointState, '/dynamic_joint_states', node.joint_state_cb, qos)
    
    # --- Publishers ---
    sp_pub = node.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)

    # --- Startup Sequence ---
    node.get_logger().info("ARMING")
    while not node.state.armed and rclpy.ok():
        try:
            modes.setArm()
        except Exception as e:
            node.get_logger().warn(f"Arm attempt failed: {e}")
        rclpy.spin_once(node, timeout_sec=0.1)
    node.armed = True
    
    # Stream setpoints before switching mode
    for k in range(100):
        node.pub_att()
        rclpy.spin_once(node, timeout_sec=1.0 / 30.0)

    try:
        modes.setOffboardMode()
        node.get_logger().info("--------- OFFBOARD ---------")
    except Exception as e:
        node.get_logger().error(f"Failed to set offboard: {e}")

    # --- Main Loop ---
    i = 0
    PLANNING_BUDGET = 8
    action = None
    try:
        while rclpy.ok():
            if i % PLANNING_BUDGET == 0:
                obs = node.get_obs()[0]
                if obs.shape[0] == node.OBS_DIM:
                    action = node.policy(obs)

            if action is not None:
                action_idx = i % PLANNING_BUDGET
                full_5d_action = action[action_idx]
                
                # Update the full H vector
                node.get_action(full_5d_action)
                
                # Control the drone
                node.pub_att()
                
                # Publish manipulator correction
                h_manip = node.currN[node.DRONE_ACTION_DIM:]
                correction_msg = Float64MultiArray(data=h_manip.tolist())
                node.manip_correction_pub.publish(correction_msg)
                
                # Publish manipulator reference (hold position)
                try:
                    joint_indices = [node.joint_states.joint_names.index(name) for name in node.MANIPULATOR_JOINT_NAMES]
                    current_joint_pos = [node.joint_states.interface_values[0].values[i] for i in joint_indices]
                    ref_msg = Float64MultiArray(data=current_joint_pos)
                    node.joint_ref_pub.publish(ref_msg)
                except (ValueError, IndexError):
                    pass
                
                # Update last_tau for the next state observation
                node.last_tau = full_5d_action
                
            rclpy.spin_once(node, timeout_sec=1.0 / 60.0)
            i += 1
            
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down node")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main(sys.argv)
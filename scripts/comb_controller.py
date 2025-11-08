#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data

# ROS Message Imports
from sensor_msgs.msg import Imu
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode
from trajectory_msgs.msg import MultiDOFJointTrajectory as Mdjt
from std_msgs.msg import Float64MultiArray

# CORRECTED: Import the controller logic classes
from .drone_controller_manip import DroneControllerLogic
from .manip_asmc import ManipulatorControllerLogic

# Math and Policy Imports
import transforms3d.euler as tfe
from tf_transformations import *
from diffusion.diffusion_inference import Policy

# This class can remain as is, it's a utility for service calls.
class fcuModes:
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


class MasterControllerNode(Node):
    def __init__(self):
        super().__init__("master_controller_node")

        # --- System Definition ---
        self.OBS_DIM = 15
        self.ACTION_DIM = 3
        self.DRONE_ACTION_DIM = 3
        self.MANIPULATOR_ACTION_DIM = 2
        self.MANIPULATOR_JOINT_NAMES = ['arm_to_L1', 'L1_to_L2']
        self.N_MANIP_JOINTS = 3  # Total number of joints for manipulator controller (2 real + 1 dummy)

        # --- State Variables (The single source of truth for system state) ---
        self.state = State()
        self.sp = PoseStamped()
        self.sp.pose.position.x = 0.0
        self.sp.pose.position.y = 0.0
        self.sp.pose.position.z = 3.0
        self.imu = Imu()
        self.local_pos = PoseStamped()
        self.local_vel = TwistStamped()
        # FIX: self.joint_states was missing, but is needed for get_obs
        self.curr_pos = np.zeros(self.N_MANIP_JOINTS)
        self.curr_vel = np.zeros(self.N_MANIP_JOINTS)
        self.curr_eff = np.zeros(self.N_MANIP_JOINTS)

        self.des_pos_joints = np.zeros(self.N_MANIP_JOINTS)
        self.armed = False

        # --- Policy and Action Related Variables ---
        self.last_tau = np.zeros(self.ACTION_DIM)
        self.currN_drone = np.zeros(self.DRONE_ACTION_DIM)
        # FIX: The manipulator's disturbance vector needs 3 elements for the controller
        self.currN_manip = np.zeros(self.N_MANIP_JOINTS) 

        # --- Timers for calculating dt ---
        self.pre_time_drone = self.get_clock().now().nanoseconds * 1e-9
        self.pre_time_manip = self.get_clock().now().nanoseconds * 1e-9

        # --- Controller Logic Instantiation ---
        # CRITICAL FIX: Create instances of the logic classes
        self.drone_controller = DroneControllerLogic()
        self.manip_controller = ManipulatorControllerLogic(n_joints=self.N_MANIP_JOINTS)
        
        # --- Publishers ---
        self.att_pub = self.create_publisher(AttitudeTarget, "/mavros/setpoint_raw/attitude", 10)
        self.effort_pub = self.create_publisher(Float64MultiArray, '/arm_effort_controller/commands', 10)

        # --- Policy Loading ---
        self.get_logger().info("Loading DRONE+MANIPULATOR diffusion policy...")
        self.policy = Policy(config="experiment_jax.yaml", use_ema=True, rng=42)
        self.get_logger().info("Policy loaded. Warming up...")
        warmup_input = jnp.ones((self.OBS_DIM,), dtype=jnp.float32)
        self.policy(warmup_input)
        self.get_logger().info("Policy is ready.")

    def posCb(self, msg: PoseStamped):
        self.local_pos = msg

    def velCb(self, msg: TwistStamped):
        self.local_vel = msg

    def stateCb(self, msg: State):
        self.state = msg

    def accCB(self, msg: Imu):
        self.imu = msg
        
    def joint_state_cb(self, msg: DynamicJointState):
        L = min(len(msg.interface_values), self.N_MANIP_JOINTS)
        for i in range(L):
            vals = msg.interface_values[i].values
            if len(vals) >= 1:
                self.curr_pos[i] = vals[0]
            if len(vals) >= 2:
                self.curr_vel[i] = vals[1]
            if len(vals) >= 3:
                self.curr_eff[i] = vals[2]

    def multiDoFCb(self, msg: Mdjt):
        if msg.points and msg.points[0].transforms: 
            pt = msg.points[0]
            self.sp.pose.position.x = pt.transforms[0].translation.x
            self.sp.pose.position.y = pt.transforms[0].translation.y
            self.sp.pose.position.z = pt.transforms[0].translation.z

    def joint_reference_callback(self, msg: Float64MultiArray):
        num_inputs = min(len(msg.data), self.N_MANIP_JOINTS)
        self.des_pos_joints[:num_inputs] = np.array(msg.data[:num_inputs])

    def vector2Arrays(self, vector):
        return np.array([vector.x, vector.y, vector.z])
        
    def update_disturbance(self, curr_act):
        self.currN_drone = curr_act[:self.DRONE_ACTION_DIM]
        # Pad the 2D manipulator action with a zero for the dummy joint
        self.currN_manip = np.append(curr_act[self.DRONE_ACTION_DIM:], 0) 

        # print(self.currN_drone, self.currN_manip)

    def get_obs(self):
        """Constructs the observation vector for the policy."""
        curPos = self.vector2Arrays(self.local_pos.pose.position)
        curVel = self.vector2Arrays(self.local_vel.twist.linear)
        # FIX: The original drone controller had a gravity constant. Use it.
        curAcc = self.vector2Arrays(self.imu.linear_acceleration) - self.drone_controller.gravity
        q = self.local_pos.pose.orientation
        (roll_curr, pitch_curr, yaw_curr) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # This logic is from the original drone_controller_manip.py and is correct
        try:
            # joint_indices = [self.joint_states.joint_names.index(name) for name in self.MANIPULATOR_JOINT_NAMES]
            # Assumes positions are in the first interface_value, velocities in the second
            j_pos = np.array([self.curr_pos[i] for i in range(2)])
            j_vel = np.array([self.curr_vel[i] for i in range(2)])
        except (ValueError, IndexError, AttributeError):
            # Fallback if joint_states isn't populated yet or names don't match
            self.get_logger().warn("Could not parse joint states for observation.", throttle_duration_sec=5)
            j_pos = np.zeros(self.MANIPULATOR_ACTION_DIM)
            j_vel = np.zeros(self.MANIPULATOR_ACTION_DIM)
            
        obs = np.concatenate([
            curPos, curVel, curAcc, 
            np.array([roll_curr, pitch_curr, yaw_curr]), 
            # j_pos, j_vel, self.last_tau
            self.last_tau
        ]).astype(np.float32)
        
        return obs

    def run_drone_control(self):
        """Calculates and publishes the drone's attitude command."""
        dt = self.get_clock().now().nanoseconds * 1e-9 - self.pre_time_drone
        self.pre_time_drone = self.pre_time_drone + dt

        # Prepare numpy arrays from state messages
        curPos = self.vector2Arrays(self.local_pos.pose.position)
        curVel = self.vector2Arrays(self.local_vel.twist.linear)
        desPos = self.vector2Arrays(self.sp.pose.position)
        q_curr = np.array([self.local_pos.pose.orientation.x, self.local_pos.pose.orientation.y, self.local_pos.pose.orientation.z, self.local_pos.pose.orientation.w])

        # Call the controller logic
        a_des_result = self.drone_controller.a_des(curPos, curVel, desPos, self.currN_drone, dt, self.armed)
        quat_des, collective_thrust = self.drone_controller.geo_con(q_curr, a_des_result)
        # Publish the command
        att_cmd = AttitudeTarget()
        att_cmd.header.stamp = self.get_clock().now().to_msg()
        att_cmd.orientation.x = quat_des[0] # w is last in tfq, but first in ROS
        att_cmd.orientation.y = quat_des[1]
        att_cmd.orientation.z = quat_des[2]
        att_cmd.orientation.w = quat_des[3]
        att_cmd.thrust = float(collective_thrust)
        att_cmd.type_mask = 7 # Ignore angular rates
        self.att_pub.publish(att_cmd)

        # self.last_tau[:3] =  a_des_result


    def run_manip_control(self):
        """Calculates and publishes the manipulator's effort command."""
        dt = self.get_clock().now().nanoseconds * 1e-9 - self.pre_time_manip
        self.pre_time_manip = self.pre_time_manip + dt

        # Prepare numpy arrays from state messages
        try:
            # Need to get current pos/vel for all N_MANIP_JOINTS
            curr_pos_all = np.zeros(self.N_MANIP_JOINTS)
            curr_vel_all = np.zeros(self.N_MANIP_JOINTS)
            for i in range(self.N_MANIP_JOINTS):
                curr_pos_all[i] = self.curr_pos[i]
                curr_vel_all[i] = self.curr_vel[i]
        except (IndexError, AttributeError):
            # Not ready yet, skip control iteration
            return

        # Call the controller logic
        efforts = self.manip_controller.des_eff(
            curr_pos=curr_pos_all,
            des_pos=self.des_pos_joints,
            curr_vel=curr_vel_all,
            arm_currN=self.currN_manip,
            dt=dt
        )

        # Publish the command
        effort_msg = Float64MultiArray(data=efforts.tolist())
        self.effort_pub.publish(effort_msg)

        # self.last_tau[3:] = efforts[:2]  # Only first two joints correspond to policy action

def main(argv):
    rclpy.init(args=argv)
    
    # FIX: Use the correct class name
    node = MasterControllerNode()
    modes = fcuModes(node)
    
    # --- Subscriptions ---
    qos = qos_profile_sensor_data
    node.create_subscription(State, "/mavros/state", node.stateCb, 10)
    node.create_subscription(Imu, "/mavros/imu/data", node.accCB, qos)
    node.create_subscription(PoseStamped, "/mavros/local_position/pose", node.posCb, qos)
    node.create_subscription(TwistStamped, "/mavros/local_position/velocity_local", node.velCb, qos)
    node.create_subscription(Mdjt, "/command/trajectory", node.multiDoFCb, 10)
    node.create_subscription(DynamicJointState, '/dynamic_joint_states', node.joint_state_cb, qos)
    # FIX: Typo was '..', should be '.'
    node.create_subscription(Float64MultiArray, '/joint_reference', node.joint_reference_callback, 10)

    # --- Startup Sequence ---
    # Wait for a few messages to arrive to ensure state is not all zeros
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info("ARMING")
    while not node.state.armed and rclpy.ok():
        modes.setArm()
        rclpy.spin_once(node, timeout_sec=0.2)
    node.armed = True
    node.get_logger().info("ARMED")
    
    # Stream setpoints before switching mode
    for _ in range(100):
        node.run_drone_control()
        rclpy.spin_once(node, timeout_sec=1.0 / 50.0)

    try:
        modes.setOffboardMode()
        node.get_logger().info("--------- OFFBOARD ---------")
    except Exception as e:
        node.get_logger().error(f"Failed to set offboard: {e}")

    # --- Main Loop ---
    i = 0
    PLANNING_BUDGET = 8
    action = None
    # loop_rate = node.create_rate(60) # 60 Hz loop

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)

            if i % PLANNING_BUDGET == 0:
                obs = node.get_obs()
                if obs.shape[0] == node.OBS_DIM:
                    action = node.policy(obs)

            if action is not None:
                action_idx = i % PLANNING_BUDGET
                full_5d_action = action[action_idx]
                
                # Update the disturbance vectors for controllers
                node.update_disturbance(full_5d_action)
                
                # Run both controllers and publish commands
                node.run_drone_control()
                node.run_manip_control()
                               
                # Update last_tau for the next state observation
                node.last_tau = full_5d_action
                
            i += 1
            
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected.")
    finally:
        node.get_logger().info("Shutting down node.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main(sys.argv)
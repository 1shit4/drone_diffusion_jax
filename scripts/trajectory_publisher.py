#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist, Vector3, Quaternion
from std_msgs.msg import Header

# ==============================================================================
# HELPER AND TRAJECTORY MATH FUNCTIONS (UNCHANGED)
# ==============================================================================

def create_waypoint(pos, vel, acc):
    traj_point = MultiDOFJointTrajectoryPoint()
    rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    transform = Transform(translation=Vector3(x=pos[0], y=pos[1], z=pos[2]), rotation=rotation)
    velocity = Twist(linear=Vector3(x=vel[0], y=vel[1], z=vel[2]))
    acceleration = Twist(linear=Vector3(x=acc[0], y=acc[1], z=acc[2]))
    traj_point.transforms.append(transform)
    traj_point.velocities.append(velocity)
    traj_point.accelerations.append(acceleration)
    return traj_point

def get_hover_point(z_height=3.0):
    return create_waypoint([0.0, 0.0, z_height], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

def get_circular_trajectory_point(radius, speed, time):
    x, y, z = radius * np.cos(speed * time), radius * np.sin(speed * time), 3.0
    vx, vy, vz = -radius * speed * np.sin(speed * time), radius * speed * np.cos(speed * time), 0.0
    ax, ay, az = -radius * speed**2 * np.cos(speed * time), -radius * speed**2 * np.sin(speed * time), 0.0
    return create_waypoint([x, y, z], [vx, vy, vz], [ax, ay, az])

def get_line_trajectory_point(length, duration, time):
    half_duration = duration / 2; speed = length / half_duration
    if time <= half_duration: x, vx = speed * time, speed
    else: x, vx = length - (speed * (time - half_duration)), -speed
    y, z = 0.0, 3.0; vy, vz, ax, ay, az = 0.0, 0.0, 0.0, 0.0, 0.0
    return create_waypoint([x, y, z], [vx, vy, vz], [ax, ay, az])

def get_figure_eight_trajectory_point(scale, speed, time):
    x, y, z = scale * np.cos(speed * time), scale * np.sin(2 * speed * time) / 2, 3.0
    vx, vy, vz = -scale * speed * np.sin(speed * time), scale * speed * np.cos(2 * speed * time), 0.0
    ax, ay, az = -scale * speed**2 * np.cos(speed * time), -2 * scale * speed**2 * np.sin(2 * speed * time), 0.0
    return create_waypoint([x, y, z], [vx, vy, vz], [ax, ay, az])

def get_vertical_line_point(height, duration, time, z_start=3.0):
    half_duration = duration / 2; speed = height / half_duration
    if time <= half_duration: z, vz = z_start + (speed * time), speed
    else: z, vz = (z_start + height) - (speed * (time - half_duration)), -speed
    x, y = 0.0, 0.0; vx, vy, ax, ay, az = 0.0, 0.0, 0.0, 0.0, 0.0
    return create_waypoint([x, y, z], [vx, vy, vz], [ax, ay, az])

def get_looping_helical_point(radius, angular_speed, vertical_speed, height, time, z_start=3.0):
    x, y = radius * np.cos(angular_speed * time), radius * np.sin(angular_speed * time)
    vx, vy = -radius * angular_speed * np.sin(angular_speed * time), radius * angular_speed * np.cos(angular_speed * time)
    ax, ay = -radius * angular_speed**2 * np.cos(angular_speed * time), -radius * angular_speed**2 * np.sin(angular_speed * time)
    half_period = height / vertical_speed; full_period = 2 * half_period; cycle_time = time % full_period
    if cycle_time <= half_period: z, vz = z_start + (vertical_speed * cycle_time), vertical_speed
    else: z, vz = (z_start + height) - (vertical_speed * (cycle_time - half_period)), -vertical_speed
    az = 0.0
    return create_waypoint([x, y, z], [vx, vy, vz], [ax, ay, az])

# ==============================================================================
# ROS2 NODE IMPLEMENTATION
# ==============================================================================

class TrajectoryPublisherNode(Node):
    def __init__(self):
        super().__init__('trajectory_publisher_node')
        
        # THE FIX #1: Publishing to the correct topic that the controller subscribes to.
        self.publisher_ = self.create_publisher(MultiDOFJointTrajectory, '/command/trajectory', 10)
        
        self.timer_ = self.create_timer(1/30.0, self.publish_callback)
        self.start_time_ = self.get_clock().now().nanoseconds / 1e9

        # --- CHOOSE YOUR TRAJECTORY AND PARAMETERS HERE ---
        self.TRAJECTORY_TYPE = "CIRCLE"
        self.LINE_HEIGHT, self.LINE_DURATION = 3.0, 8.0
        self.RADIUS, self.ANGULAR_SPEED = 2.5, 0.5
        self.VERTICAL_SPEED, self.MAX_HEIGHT = 0.5, 3.0
        self.LINE_LENGTH = 5.0
        self.SCALE, self.SPEED = 4.0, 0.4
        
        self.get_logger().info(f"Trajectory Publisher is running. Publishing a {self.TRAJECTORY_TYPE} trajectory.")

    def publish_callback(self):
        current_time = (self.get_clock().now().nanoseconds / 1e9) - self.start_time_
        
        # THE FIX #2: Add modulo operator for looping trajectories.
        if self.TRAJECTORY_TYPE == "HOVER":
            trajectory_point = get_hover_point()
        elif self.TRAJECTORY_TYPE == "CIRCLE":
            trajectory_point = get_circular_trajectory_point(self.RADIUS, self.ANGULAR_SPEED, current_time)
        elif self.TRAJECTORY_TYPE == "LINE":
            looped_time = current_time % self.LINE_DURATION
            trajectory_point = get_line_trajectory_point(self.LINE_LENGTH, self.LINE_DURATION, looped_time)
        elif self.TRAJECTORY_TYPE == "FIGURE_EIGHT":
            trajectory_point = get_figure_eight_trajectory_point(self.SCALE, self.SPEED, current_time)
        elif self.TRAJECTORY_TYPE == "VERTICAL_LINE":
            looped_time = current_time % self.LINE_DURATION
            trajectory_point = get_vertical_line_point(self.LINE_HEIGHT, self.LINE_DURATION, looped_time)
        elif self.TRAJECTORY_TYPE == "HELIX_LOOP":
            trajectory_point = get_looping_helical_point(self.RADIUS, self.ANGULAR_SPEED, self.VERTICAL_SPEED, self.MAX_HEIGHT, current_time)
        else:
            self.get_logger().error(f"Invalid TRAJECTORY_TYPE: {self.TRAJECTORY_TYPE}")
            trajectory_point = get_hover_point() # Default to hover for safety
            
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
        trajectory_msg = MultiDOFJointTrajectory(header=header, points=[trajectory_point])
        self.publisher_.publish(trajectory_msg)

def main(args=None):
    rclpy.init(args=args)
    trajectory_publisher_node = TrajectoryPublisherNode()
    rclpy.spin(trajectory_publisher_node)
    trajectory_publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
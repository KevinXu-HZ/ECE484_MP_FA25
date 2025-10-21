#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState
from gazebo_msgs.msg import EntityState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util import euler_to_quaternion, quaternion_to_euler
import time


class vehicleController():

    def __init__(self, node=None):
        self.node = Node('vehicle_controller')
        self.own_node = True

        self.controlPub = self.node.create_publisher(AckermannDrive, "/ackermann_cmd", 1)
        self.prev_vel = 0
        self.L = 1.75
        self.log_acceleration = True  # Enable logging for Problem 5

        # Data logging for plots
        self.time_data = []
        self.acceleration_data = []
        self.trajectory_x = []
        self.trajectory_y = []
        self.start_time = time.time()

       


    # Tasks 1: Read the documentation https://docs.ros.org/en/ros2_packages/humble/api/simulation_interfaces/srv/GetEntityState.html 
    #       and https://docs.ros.org/en/ros2_packages/humble/api/gazebo_msgs/msg/EntityState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
     
    def extract_vehicle_info(self, currentPose):

        ####################### TODO: Your TASK 1 code starts Here #######################
        pos_x, pos_y, vel, yaw = 0, 0, 0, 0
        if currentPose.success:
            print("gem found")
    
            # get pos_x and pos_y
            point = currentPose.state.pose.position
            pos_x = point.x
            pos_y = point.y
    
            # get yaw
            orientation = currentPose.state.pose.orientation
            quat = [orientation.x, orientation.y, orientation.z, orientation.w]
            euler = quaternion_to_euler(quat)
            yaw = euler[2]
    
            #get vel
            twist = currentPose.state.twist
            vel_x = twist.linear.x
            vel_y = twist.linear.y
            # vel_z = twist.linear.z
            vel = (vel_x ** 2 + vel_y ** 2) ** 0.5
        else:
            print("error detected")
    
        ####################### TODO: Your Task 1 code ends Here #######################
    
        return pos_x, pos_y, vel, yaw # note that yaw is in radians


    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity

    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):

        ####################### TODO: Your TASK 2 code starts Here #######################
        # Optimized velocity control with look-ahead straight detection

        # Pure logic - no artificial limits
        straight_speed = 100.0     # Speed for straight sections
        curve_speed = 30.0         # Speed for curves
        straight_threshold = math.radians(5)  # Heading change threshold
        decel_rate = 100.0  # Desired deceleration rate (m/s²)

        # Calculate braking distance: d = (v_straight² - v_curve²) / (2 × a)
        braking_anticipation = (straight_speed**2 - curve_speed**2) / (2 * decel_rate)

        if len(future_unreached_waypoints) < 3:
            target_velocity = curve_speed
        else:
            # 1. Scan ahead to detect straight sections and upcoming curves
            straight_distance = 0.0
            curve_detected = False

            # Look ahead through waypoints
            for i in range(min(len(future_unreached_waypoints) - 2, 30)):  # Look ahead up to 30 waypoints
                # Calculate heading change at this waypoint
                v1_x = future_unreached_waypoints[i+1][0] - future_unreached_waypoints[i][0]
                v1_y = future_unreached_waypoints[i+1][1] - future_unreached_waypoints[i][1]
                v2_x = future_unreached_waypoints[i+2][0] - future_unreached_waypoints[i+1][0]
                v2_y = future_unreached_waypoints[i+2][1] - future_unreached_waypoints[i+1][1]

                angle1 = math.atan2(v1_y, v1_x)
                angle2 = math.atan2(v2_y, v2_x)
                heading_change = abs(angle2 - angle1)
                if heading_change > math.pi:
                    heading_change = 2 * math.pi - heading_change

                # Calculate segment distance
                segment_dist = math.sqrt(v1_x**2 + v1_y**2)

                if heading_change < straight_threshold:
                    # Still in straight section
                    if not curve_detected:
                        straight_distance += segment_dist
                else:
                    # Curve detected
                    curve_detected = True
                    break

            # 2. Calculate current position's curvature for immediate speed adjustment
            v1_x = future_unreached_waypoints[1][0] - future_unreached_waypoints[0][0]
            v1_y = future_unreached_waypoints[1][1] - future_unreached_waypoints[0][1]
            v2_x = future_unreached_waypoints[2][0] - future_unreached_waypoints[1][0]
            v2_y = future_unreached_waypoints[2][1] - future_unreached_waypoints[1][1]

            angle1 = math.atan2(v1_y, v1_x)
            angle2 = math.atan2(v2_y, v2_x)
            current_heading_change = abs(angle2 - angle1)
            if current_heading_change > math.pi:
                current_heading_change = 2 * math.pi - current_heading_change

            # 3. Simple decision: straight = fast, curve = slow
            if current_heading_change < straight_threshold:
                # Currently in straight
                if curve_detected and straight_distance < braking_anticipation:
                    # Curve approaching - slow down
                    target_velocity = curve_speed
                else:
                    # Clear straight - full speed
                    target_velocity = straight_speed
            else:
                # Currently in curve - slow speed
                target_velocity = curve_speed

        # 4. Asymmetric smoothing: gradual acceleration, quick braking
        if target_velocity > curr_vel:
            # Accelerating - use heavy smoothing to prevent flipping
            alpha = 0.01  # Very gradual acceleration
        else:
            # Braking - use lighter smoothing for responsive slowing
            alpha = 1  # Faster braking response

        smoothed_velocity = alpha * target_velocity + (1 - alpha) * curr_vel

        ####################### TODO: Your TASK 2 code ends Here #######################
        return smoothed_velocity

        
        

    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):
       
        ####################### TODO: Your TASK 3 code starts Here #######################
        # Pure Pursuit Algorithm Implementation
        # Using method 2: Set a constant lookahead distance and interpolate between future waypoints
        
        # Lookahead distance (can be tuned for better performance)
        lookahead_distance = 8.0
        
        # Find the lookahead point
        lookahead_point = self._find_lookahead_point(curr_x, curr_y, target_point, future_unreached_waypoints, lookahead_distance)
        
        if lookahead_point is None:
            # Fallback to target point if no lookahead point found
            lookahead_point = target_point
        
        # Calculate the angle between vehicle heading and lookahead line
        dx = lookahead_point[0] - curr_x
        dy = lookahead_point[1] - curr_y
        
        # Calculate the angle from vehicle to lookahead point
        alpha = math.atan2(dy, dx) - curr_yaw
        
        # Normalize angle to [-pi, pi]
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi
        
        # Calculate lookahead distance
        ld = math.sqrt(dx**2 + dy**2)
        
        # Pure Pursuit formula: δ = arctan(2L*sin(α)/ld)
        if ld > 0.1:  # Avoid division by zero
            target_steering = math.atan(2 * self.L * math.sin(alpha) / ld)
        else:
            target_steering = 0.0
        
        # Limit steering angle to reasonable bounds (typically ±30 degrees)
        max_steering = math.radians(30)
        target_steering = max(-max_steering, min(max_steering, target_steering))

        ####################### TODO: Your TASK 3 code ends Here #######################
        return target_steering
    
    def _find_lookahead_point(self, curr_x, curr_y, target_point, future_waypoints, lookahead_distance):
        """
        Find the lookahead point using constant lookahead distance.
        Interpolates between waypoints to find the exact point at lookahead_distance.
        """
        # Start with the target point and future waypoints
        all_waypoints = [target_point] + future_waypoints
        
        if len(all_waypoints) < 2:
            return target_point
        
        # Find the segment that contains the lookahead point
        cumulative_distance = 0.0
        prev_point = [curr_x, curr_y]
        
        for i, waypoint in enumerate(all_waypoints):
            # Calculate distance to current waypoint
            dx = waypoint[0] - prev_point[0]
            dy = waypoint[1] - prev_point[1]
            segment_distance = math.sqrt(dx**2 + dy**2)
            
            # Check if lookahead point is in this segment
            if cumulative_distance + segment_distance >= lookahead_distance:
                # Interpolate within this segment
                remaining_distance = lookahead_distance - cumulative_distance
                if segment_distance > 0:
                    ratio = remaining_distance / segment_distance
                    lookahead_x = prev_point[0] + ratio * dx
                    lookahead_y = prev_point[1] + ratio * dy
                    return [lookahead_x, lookahead_y]
                else:
                    return waypoint
            
            cumulative_distance += segment_distance
            prev_point = waypoint
        
        # If we've gone through all waypoints, return the last one
        return all_waypoints[-1]





    def execute(self, currentPose, target_point, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: GetEntityState response, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        
        if currentPose is None:
            print("Warning: No current pose data")
            return
            
        if len(future_unreached_waypoints) == 0:
            print("Warning: No waypoints available")
            return

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        # Log data for plotting
        if self.log_acceleration:
            acceleration = (curr_vel - self.prev_vel) * 100  # Since we are running at 100Hz
            current_time = time.time() - self.start_time
            self.time_data.append(current_time)
            self.acceleration_data.append(acceleration)
            self.trajectory_x.append(curr_x)
            self.trajectory_y.append(curr_y)

        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)

        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = float(target_velocity)
        newAckermannCmd.steering_angle = float(target_steering)

        self.controlPub.publish(newAckermannCmd)

        # Store current velocity for next iteration
        self.prev_vel = curr_vel

    def stop(self):
        """Stop the vehicle by setting speed to 0 and steering to 0"""
        try:
            newAckermannCmd = AckermannDrive()
            newAckermannCmd.speed = 0.0
            newAckermannCmd.steering_angle = 0.0
            self.controlPub.publish(newAckermannCmd)
            print("Controller: Stop command sent")

            # Save logged data when stopping
            if self.log_acceleration:
                self.save_data()
        except Exception as e:
            print(f"Controller: Error sending stop command: {e}")

    def save_data(self):
        """Save logged data to files for plotting"""
        import csv
        import os

        # Create data directory if it doesn't exist
        data_dir = os.path.expanduser("~/mp2_data")
        os.makedirs(data_dir, exist_ok=True)

        # Save acceleration data
        accel_file = os.path.join(data_dir, "acceleration_data.csv")
        with open(accel_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'acceleration'])
            for t, a in zip(self.time_data, self.acceleration_data):
                writer.writerow([t, a])

        # Save trajectory data
        traj_file = os.path.join(data_dir, "trajectory_data.csv")
        with open(traj_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            for x, y in zip(self.trajectory_x, self.trajectory_y):
                writer.writerow([x, y])

        print(f"Data saved to {data_dir}")
        print(f"  - {accel_file}")
        print(f"  - {traj_file}")
        
    def destroy(self):
        if self.own_node:

            self.node.destroy_node()


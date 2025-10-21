#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Float32MultiArray
import time
import sys
import os
import argparse

# Add the local module path for importing controller
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the bicycle model controller
from controller import BicycleModel


class VehicleNode(Node):
    def __init__(self, alternate_path=False):
        super().__init__('vehicle_node')
        
        # Initialize parameters
        self.declare_parameter('model_name', 'polaris')
        self.declare_parameter('use_alternate_route', alternate_path)
        
        # Get model name from parameter
        self.model_name = self.get_parameter('model_name').value
        
        # Initialize the bicycle model controller
        self.model = BicycleModel()
        
        if not alternate_path:
            # Default position list for running counter-clockwise starting right upper corner
            self.pos_list = [
                [100, 53], [80, 57], [60, 56], [50, 57], [40, 58], [35, 55], 
                [34, 44], [40, 39], [45, 40], [55, 40], [68, 40], [75, 30], 
                [75, 28], [83, 22], [104, 22], [110, 34], [102, 39], [96, 47]
            ]
        else:
            # Alternate position list
            if self.get_parameter('use_alternate_route').value:
                self.pos_list = [
                    [68, 40], [75, 30], [75, 28], [83, 22], [104, 22], [110, 34], 
                    [102, 39], [96, 47], [100, 53], [80, 57], [60, 56], [50, 57], 
                    [40, 58], [35, 55], [34, 44], [40, 39], [45, 40], [55, 40]
                ]
        
        # Initialize position index
        self.pos_idx = 0
        
        # Create initial target state
        self.target_state = ModelState()
        self.target_state.model_name = self.model_name
        self.target_state.pose.position.x = np.float64(self.pos_list[self.pos_idx][0] + 15 - 100)
        self.target_state.pose.position.y = np.float64(self.pos_list[self.pos_idx][1] + 100 - 100)
        
        # Create a timer for updating vehicle movement (30 Hz)
        self.timer = self.create_timer(0.033, self.update_vehicle)
        
        self.get_logger().info('Vehicle node initialized with waypoint following')
    
    def update_vehicle(self):
        """Update the vehicle state and handle waypoint following"""
        # Get current state

        curr_state = self.model.get_model_state()

        # If couldn't get state, try again next time
        if curr_state is None or not hasattr(curr_state, 'success') or not curr_state.success:
            self.get_logger().warn('Failed to get current model state')
            return
        
        # Calculate distance to target
        dist_to_target_x = abs(self.target_state.pose.position.x - curr_state.state.pose.position.x)
        dist_to_target_y = abs(self.target_state.pose.position.y - curr_state.state.pose.position.y)
        # Check if waypoint is reached
        if dist_to_target_x < 1.0 and dist_to_target_y < 1.0:
            # Move to next waypoint
            self.pos_idx = (self.pos_idx + 1) % len(self.pos_list)
            
            # Update target state
            self.target_state = ModelState()
            self.target_state.model_name = self.model_name
            self.target_state.pose.position.x = float(self.pos_list[self.pos_idx][0] + 15 - 100)
            self.target_state.pose.position.y = float(self.pos_list[self.pos_idx][1] + 100 - 100)
            
            self.get_logger().info(
                f"Reached waypoint, moving to: ({self.pos_list[self.pos_idx][0]}, {self.pos_list[self.pos_idx][1]})"
            )
        else:

            # Continue moving toward current target
            self.model.set_model_state(curr_state, self.target_state)


def main(argv):
    rclpy.init()
    node = VehicleNode(argv.alternate)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waypoint follower for simulated GEM vehicle.')
    
    parser.add_argument(
        '--alternate',
        help='Width of viewing window.',
        action="store_true")
    
    main(parser.parse_args())
#!/usr/bin/env python3

import sys
import os
import argparse

import numpy as np

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState, SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist
from scipy.spatial.transform import Rotation as R

class PoseSetterNode(Node):
    def __init__(self):
        super().__init__('set_pos')
        
        # Create service clients
        self.get_entity_state_client = self.create_client(GetEntityState, '/get_entity_state')
        self.set_entity_state_client = self.create_client(SetEntityState, '/set_entity_state')
        
        # Wait for services to be available
        while not self.get_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for get_entity_state service...')
            
        while not self.set_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_entity_state service...')

    def get_entity_state(self, entity_name='gem'):
        """Get the current state of an entity from Gazebo"""
        request = GetEntityState.Request()
        request.name = entity_name
        request.reference_frame = 'world'
        
        future = self.get_entity_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            return future.result()
        else:
            self.get_logger().error('Failed to get entity state')
            return None

    def set_entity_state(self, state):
        """Set the state of an entity in Gazebo"""
        request = SetEntityState.Request()
        request.state = state
        
        future = self.set_entity_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            if future.result().success:
                self.get_logger().info('Successfully set entity state')
                return True
            else:
                self.get_logger().warn('Failed to set entity state')
                return False
        else:
            self.get_logger().error('Service call failed')
            return False

    def set_position(self, x=0, y=0, heading=np.pi/2):
        """Set the position and orientation of the vehicle"""
        curr_state = self.get_entity_state()
        
        if curr_state is None:
            self.get_logger().error('Could not get current entity state')
            return False
        
        # Create a new entity state
        new_state = EntityState()
        new_state.name = 'gem'
        new_state.pose = curr_state.state.pose
        new_state.pose.position.x = float(x)
        new_state.pose.position.y = float(y)
        new_state.pose.position.z = 1.0
        
        # Convert euler angles to quaternion
        q = R.from_euler("zyx", [0, 0, heading]).as_quat()
        new_state.pose.orientation.w = q[0]
        new_state.pose.orientation.x = q[1]
        new_state.pose.orientation.y = q[2]
        new_state.pose.orientation.z = q[3]
        
        new_state.twist = curr_state.state.twist
        new_state.reference_frame = 'world'
        
        # Set the new state
        return self.set_entity_state(new_state)

def main(args=None):
    parser = argparse.ArgumentParser(description='Set the x, y position of the vehicle')
    
    x_default = 15
    y_default = 45
    heading_default = np.pi/2
    
    parser.add_argument('--x', type=float, help='x position of the vehicle', default=x_default)
    parser.add_argument('--y', type=float, help='y position of the vehicle', default=y_default)
    parser.add_argument('--heading', type=float, help='heading angle of the vehicle', default=heading_default)
    
    # Parse command line arguments
    parsed_args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create node and set position
    node = PoseSetterNode()
    success = node.set_position(x=parsed_args.x, y=parsed_args.y, heading=parsed_args.heading)
    
    # Clean up
    node.destroy_node()
    rclpy.shutdown()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

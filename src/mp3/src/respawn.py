#!/usr/bin/env python3

import sys
import os
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion
from ament_index_python.packages import get_package_share_directory

class RespawnNode(Node):
    def __init__(self):
        super().__init__('respawn')
        self.y = 45
        self.x = 15
        
        # Get package share directory
        mp3_pkg_dir = get_package_share_directory('mp3')
        self.path = mp3_pkg_dir
        
        print(self.path)
        
        # Create service clients
        self.delete_client = self.create_client(DeleteModel, "/gazebo/delete_model")
        self.spawn_urdf_client = self.create_client(SpawnModel, "/gazebo/spawn_urdf_model")
        self.spawn_sdf_client = self.create_client(SpawnModel, "/gazebo/spawn_sdf_model")
        
        # Wait for services
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for delete_model service...')
        
        while not self.spawn_urdf_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn_urdf_model service...')
            
        while not self.spawn_sdf_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn_sdf_model service...')
        
        self.respawn_models()
    
    def respawn_models(self):
        # Delete existing models
        try:
            delete_request = DeleteModel.Request()
            delete_request.model_name = "polaris"
            future = self.delete_client.call_async(delete_request)
            rclpy.spin_until_future_complete(self, future)
            
            delete_request.model_name = "marker"
            future = self.delete_client.call_async(delete_request)
            rclpy.spin_until_future_complete(self, future)
            
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        
        # Spawn polaris model
        try:
            urdf_path = os.path.join(self.path, "urdf", "polaris.urdf")
            with open(urdf_path, 'r') as f:
                urdf_content = f.read()
            
            spawn_request = SpawnModel.Request()
            spawn_request.model_name = "polaris"
            spawn_request.model_xml = urdf_content
            spawn_request.robot_namespace = "polaris"
            spawn_request.initial_pose = Pose(
                position=Point(x=self.x, y=self.y, z=1.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.707, w=0.707)
            )
            spawn_request.reference_frame = "world"
            
            future = self.spawn_urdf_client.call_async(spawn_request)
            rclpy.spin_until_future_complete(self, future)
            
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        
        # Spawn marker model
        try:
            sdf_path = os.path.join(self.path, "models", "marker", "model.sdf")
            with open(sdf_path, 'r') as f:
                sdf_content = f.read()
            
            spawn_request = SpawnModel.Request()
            spawn_request.model_name = "marker"
            spawn_request.model_xml = sdf_content
            spawn_request.robot_namespace = "marker"
            spawn_request.initial_pose = Pose(
                position=Point(x=self.x, y=self.y, z=1.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
            )
            spawn_request.reference_frame = "world"
            
            future = self.spawn_sdf_client.call_async(spawn_request)
            rclpy.spin_until_future_complete(self, future)
            
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RespawnNode()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

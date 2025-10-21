#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import GetEntityState
from tf2_ros import TransformBroadcaster
import math

class OdometryPublisher(Node):
    def __init__(self):
        super().__init__('gem_odometry_publisher')
        
        # Declare parameters
        self.declare_parameter('entity_name', 'gem')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('use_entity_state', True)
        
        self.entity_name = self.get_parameter('entity_name').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.use_entity_state = self.get_parameter('use_entity_state').value
        
        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Create odometry publisher
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        
        # Create service client for getting entity state
        if self.use_entity_state:
            self.get_entity_state_client = self.create_client(GetEntityState, '/get_entity_state')
            # Wait for service to be available
            while not self.get_entity_state_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Waiting for /get_entity_state service...')
        else:
            # Alternative: use get_model_state service
            from gazebo_msgs.srv import GetModelState
            self.get_model_state_client = self.create_client(GetModelState, '/get_model_state')
            while not self.get_model_state_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Waiting for /get_model_state service...')
        
        # Create timer for periodic updates
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f'GEM odometry publisher initialized for entity: {self.entity_name}')
        self.get_logger().info(f'Publish rate: {self.publish_rate} Hz')
    
    def timer_callback(self):
        """
        Timer callback to publish transforms and odometry
        """
        try:
            if self.use_entity_state:
                self.publish_entity_state()
            else:
                self.publish_model_state()
                
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {e}')
    
    def publish_entity_state(self):
        """Publish using get_entity_state service"""
        request = GetEntityState.Request()
        request.name = self.entity_name
        
        future = self.get_entity_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.1)
        
        if future.done():
            response = future.result()
            if response.success:
                self.publish_transforms(response.state)
                self.publish_odometry(response.state)
        #     else:
        #         # self.get_logger().warn(f'Failed to get entity state for {self.entity_name}')
        # else:
        #     # self.get_logger().warn('Service call timed out')
    
    def publish_model_state(self):
        """Publish using get_model_state service"""
        from gazebo_msgs.srv import GetModelState
        request = GetModelState.Request()
        request.model_name = self.entity_name
        
        future = self.get_model_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.1)
        
        if future.done():
            response = future.result()
            if response.success:
                self.publish_transforms(response.pose, response.twist)
                self.publish_odometry(response.pose, response.twist)
        #     else:
        #         self.get_logger().warn(f'Failed to get model state for {self.entity_name}')
        # else:
        #     self.get_logger().warn('Service call timed out')
    
    def publish_transforms(self, pose, twist=None):
        """
        Publish world->odom and odom->base_link transforms
        """
        # Publish world->odom transform (identity transform for now)
        world_to_odom = TransformStamped()
        world_to_odom.header.stamp = self.get_clock().now().to_msg()
        world_to_odom.header.frame_id = 'world'
        world_to_odom.child_frame_id = 'odom'
        world_to_odom.transform.translation.x = 0.0
        world_to_odom.transform.translation.y = 0.0
        world_to_odom.transform.translation.z = 0.0
        world_to_odom.transform.rotation.x = 0.0
        world_to_odom.transform.rotation.y = 0.0
        world_to_odom.transform.rotation.z = 0.0
        world_to_odom.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(world_to_odom)
        
        # Publish odom->base_link transform
        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = self.get_clock().now().to_msg()
        odom_to_base.header.frame_id = 'odom'
        odom_to_base.child_frame_id = 'base_link'
        odom_to_base.transform.translation.x = pose.position.x
        odom_to_base.transform.translation.y = pose.position.y
        odom_to_base.transform.translation.z = pose.position.z
        odom_to_base.transform.rotation.x = pose.orientation.x
        odom_to_base.transform.rotation.y = pose.orientation.y
        odom_to_base.transform.rotation.z = pose.orientation.z
        odom_to_base.transform.rotation.w = pose.orientation.w
        
        self.tf_broadcaster.sendTransform(odom_to_base)
    
    def publish_odometry(self, pose, twist=None):
        """
        Publish odometry message
        """
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        # Set pose
        odom_msg.pose.pose.position.x = pose.position.x
        odom_msg.pose.pose.position.y = pose.position.y
        odom_msg.pose.pose.position.z = pose.position.z
        odom_msg.pose.pose.orientation.x = pose.orientation.x
        odom_msg.pose.pose.orientation.y = pose.orientation.y
        odom_msg.pose.pose.orientation.z = pose.orientation.z
        odom_msg.pose.pose.orientation.w = pose.orientation.w
        
        # Set twist (velocity) if available
        if twist is not None:
            odom_msg.twist.twist.linear.x = twist.linear.x
            odom_msg.twist.twist.linear.y = twist.linear.y
            odom_msg.twist.twist.linear.z = twist.linear.z
            odom_msg.twist.twist.angular.x = twist.angular.x
            odom_msg.twist.twist.angular.y = twist.angular.y
            odom_msg.twist.twist.angular.z = twist.angular.z
        else:
            # Set zero velocity if twist not available
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0
            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0
        
        # Set covariance (identity matrix for now)
        for i in range(36):
            odom_msg.pose.covariance[i] = 0.0
            odom_msg.twist.covariance[i] = 0.0
        
        # Set diagonal elements
        odom_msg.pose.covariance[0] = 0.1  # x position variance
        odom_msg.pose.covariance[7] = 0.1  # y position variance
        odom_msg.pose.covariance[35] = 0.1  # yaw variance
        
        odom_msg.twist.covariance[0] = 0.1  # x velocity variance
        odom_msg.twist.covariance[7] = 0.1  # y velocity variance
        odom_msg.twist.covariance[35] = 0.1  # yaw rate variance
        
        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    
    odometry_publisher = OdometryPublisher()
    
    try:
        rclpy.spin(odometry_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        odometry_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState, SetEntityState
from gazebo_msgs.msg import ModelState, EntityState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from scipy.integrate import ode
from std_msgs.msg import Float32MultiArray

def func1(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    
    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx, dy, dtheta]


class BicycleModel(Node):
    def __init__(self):
        super().__init__('bicycle_model')
        
        # Create publishers
        self.waypoint_pub = self.create_publisher(ModelState, '/gem/waypoint', 1)
        self.control_pub = self.create_publisher(Float32MultiArray, '/gem/control', 1)
        
        # Create subscribers
        self.waypoint_sub = self.create_subscription(
            ModelState,
            '/gem/waypoint',
            self.__waypoint_handler,
            1)
        
        # Create service clients
        self.get_entity_state_client = self.create_client(GetEntityState, '/get_entity_state')
        self.set_entity_state_client = self.create_client(SetEntityState, '/set_entity_state')
        
        # Wait for services to be available
        # while not self.get_entity_state_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Waiting for /gazebo/get_entity_state service...')
        
        # while not self.set_entity_state_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        
        # Initialize waypoint list
        self.waypoint_list = []
        
        self.get_logger().info('Bicycle Model initialized')

    def get_model_state(self):
        """Get the current state of the model from Gazebo"""
        request = GetEntityState.Request()
        request.name = 'gem'
        request.reference_frame = 'world'

        
        future = self.get_entity_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            return future.result()
        else:
            self.get_logger().error('Failed to get entity state')
            return None

    def rear_wheel_model(self, ackermann_cmd):
        """Simulate the rear wheel bicycle model dynamics"""
        current_model_state = self.get_model_state()

        if current_model_state is None:
            return None

        vr = ackermann_cmd.speed
        delta = ackermann_cmd.steering_angle

        x = current_model_state.state.pose.position.x
        y = current_model_state.state.pose.position.y
        euler = self.quaternion_to_euler(
            current_model_state.state.pose.orientation.x,
            current_model_state.state.pose.orientation.y,
            current_model_state.state.pose.orientation.z,
            current_model_state.state.pose.orientation.w
        )
        theta = euler[2]

        init_r = [x, y, theta]    
        r = ode(func1)
        r.set_initial_value(init_r)
        r.set_f_params(vr, delta)
        val = r.integrate(r.t + 0.01)

        new_x = val[0]
        new_y = val[1]
        new_theta = val[2]
        return [new_x, new_y, new_theta]

    def rear_wheel_feedback(self, current_pose, target_pose):
        """Generate control commands based on current and target poses"""
        # Gain values
        k1 = 1
        k2 = 1
        k3 = 1
        
        current_euler = self.quaternion_to_euler(
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        )

        curr_x = current_pose.position.x
        curr_y = current_pose.position.y
        curr_theta = current_euler[2]

        # Handle target_pose from different message types
        if hasattr(target_pose, 'pose'):
            # If it's a ModelState message
            targ_x = target_pose.pose.position.x
            targ_y = target_pose.pose.position.y
        else:
            # If it's already a Pose message
            targ_x = target_pose.position.x
            targ_y = target_pose.position.y

        error_x = targ_x - curr_x
        error_y = targ_y - curr_y
        error_theta = (curr_theta - (np.arctan2(error_y, error_x) % (2 * np.pi))) % (np.pi * 2)
        if error_theta > np.pi:
            error_theta = error_theta - np.pi * 2
            
        vr = 10 * np.sqrt(error_x**2 + error_y**2)
        delta = -4 * error_theta

        # Apply constraints
        if delta > np.pi/3:
            delta = np.pi/3
        elif delta < -np.pi/3:
            delta = -np.pi/3

        if vr > 8:
            vr = 8

        # Create and return AckermannDrive command
        new_ackermann_cmd = AckermannDrive()
        new_ackermann_cmd.speed = np.float64(vr)
        new_ackermann_cmd.steering_angle = np.float64(delta)

        return new_ackermann_cmd

    def set_model_state(self, curr_state, target_state):
        """Set the model state based on bicycle model dynamics using SetEntityState service"""
        # For curr_state from GetEntityState, we need to access the nested state field
        current_pose = curr_state.state.pose if hasattr(curr_state, 'state') else curr_state.pose
        
        control = self.rear_wheel_feedback(current_pose, target_state)
        
        # Publish control command
        a = Float32MultiArray()
        a.data = [control.speed, control.steering_angle]

        # print("control: ", control)
        self.control_pub.publish(a)
        
        # Calculate new state using bicycle model
        values = self.rear_wheel_model(control)
        if values is None:
            self.get_logger().warn('Failed to calculate new state')
            return

        # Create EntityState for SetEntityState service
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = 'gem'
        request.state.pose.position.x = np.float64(values[0])
        request.state.pose.position.y = np.float64(values[1])
        request.state.pose.position.z = np.float64(0.006)
        
        q = self.euler_to_quaternion([values[2], 0, 0])
        request.state.pose.orientation.x = np.float64(q[0])
        request.state.pose.orientation.y = np.float64(q[1])
        request.state.pose.orientation.z = np.float64(q[2])
        request.state.pose.orientation.w = np.float64(q[3])
        
        request.state.twist.linear.x = np.float64(0)
        request.state.twist.linear.y = np.float64(0)
        request.state.twist.linear.z = np.float64(0)
        request.state.twist.angular.x = np.float64(0)
        request.state.twist.angular.y = np.float64(0)
        request.state.twist.angular.z = np.float64(0)
        request.state.reference_frame = 'world'
        
        #print("request.state: ", request.state)
        
        # Call the service

        #print("request: ", request)
        future = self.set_entity_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        #print("future: ", future)
        
        if future.result() is not None:
            if not future.result().success:
                self.get_logger().warn('Failed to set entity state')
        else:
            self.get_logger().error('Service call failed')

    def euler_to_quaternion(self, r):
        """Convert Euler angles to quaternion"""
        (yaw, pitch, roll) = (r[0], r[1], r[2])
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        
        return [roll, pitch, yaw]

    def __waypoint_handler(self, data):
        """Handle incoming waypoint messages"""
        self.waypoint_list.append(data)
        self.get_logger().debug(f"Received waypoint: ({data.pose.position.x}, {data.pose.position.y})")


def main(args=None):
    rclpy.init(args=args)
    node = BicycleModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 
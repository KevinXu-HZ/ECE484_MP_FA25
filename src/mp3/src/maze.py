import numpy as np
import turtle
import bisect
import argparse
from scipy.integrate import ode
import rclpy
import time
from rclpy.node import Node
from gazebo_msgs.srv import GetModelState, GetEntityState
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
from lidar_processing import LidarProcessing

class Maze:
    def __init__(
            self,
            maze=None,
            x_start=None,
            y_start=None,
            extensive=False):
        '''
        maze: 2D numpy array.
        passages are coded as a 4-bit number, with a bit value taking
        0 if there is a wall and 1 if there is no wall.
        The 1s register corresponds with a square's top edge,
        2s register the right edge,
        4s register the bottom edge,
        and 8s register the left edge.
        (numpy array)
        '''
        self.maze = maze
        self.num_rows = maze.shape[0]
        self.num_cols = maze.shape[1]
        self.fix_maze_boundary()
        # self.fix_wall_inconsistency()

        self.height = self.num_rows
        self.width = self.num_cols
        self.x_start = x_start
        self.y_start = y_start
        self.extensive = extensive

        self.turtle_registration()

    def turtle_registration(self):
        turtle.register_shape('tri', ((-3, -2), (0, 3), (3, -2), (0, 0)))

    def fix_maze_boundary(self):
        '''
        Make sure that the maze is bounded.
        '''
        for i in range(self.num_rows):
            self.maze[i,0] |= 8
            self.maze[i,-1] |= 2
        for j in range(self.num_cols):
            self.maze[0,j] |= 1
            self.maze[-1,j] |= 4

    def permissibilities(self, cell):
        '''
        Check if the directions of a given cell are permissible.
        Return:
        (up, right, down, left)
        '''
        cell_value = self.maze[cell[0], cell[1]]
        return (cell_value & 1 == 0, cell_value & 2 == 0, cell_value & 4 == 0, cell_value & 8 == 0)

    def colide_wall(self, y, x):
        '''
        Check if given position collide into a wall.
        Return:
        True if collide into a wall, else False
        '''

        if x >= 120 or x < 0 or y >= 75 or y < 0:
            return True
        cell_value = self.maze[y,x]
        if cell_value == 15:
            return True
        else:
            return False

    def show_maze(self):
        '''
        Display the maze
        '''

        turtle.setworldcoordinates(0, 0, self.width * 1.005, self.height * 1.005)

        wally = turtle.Turtle()
        wally.speed(0)
        wally.width(1.5)
        wally.hideturtle()
        turtle.tracer(0, 0)

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                permissibilities = self.permissibilities(cell = (i,j))
                turtle.up()
                wally.setposition((j, i))
                # Set turtle heading orientation
                # 0 - east, 90 - north, 180 - west, 270 - south
                wally.setheading(0)
                if not permissibilities[0]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(1)
                wally.setheading(90)
                wally.up()
                if not permissibilities[1]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(1)
                wally.setheading(180)
                wally.up()
                if not permissibilities[2]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(1)
                wally.setheading(270)
                wally.up()
                if not permissibilities[3]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(1)
                wally.up()

        turtle.update()

    def weight_to_color(self, weight):
        return '#%02x00%02x' % (int(weight * 255), int((1 - weight) * 255))

    def show_particles(self, particles, show_frequency = 10):
        turtle.shape('tri')

        for i, particle in enumerate(particles):
            if i % show_frequency == 0:
                turtle.setposition((particle.x, particle.y))
                turtle.setheading((particle.heading*180/np.pi)%360)
                turtle.color(self.weight_to_color(particle.weight))
                turtle.stamp()

        turtle.update()

    def show_estimated_location(self, particles):
        '''
        Show average weighted mean location of the particles.
        '''

        x_accum = 0
        y_accum = 0
        heading_cos_accum = 0
        heading_sin_accum = 0
        weight_accum = 0

        for particle in particles:
            weight_accum += particle.weight
            x_accum += particle.x * particle.weight
            y_accum += particle.y * particle.weight
            # heading_accum += ((particle.heading*180/np.pi)%360) * particle.weight
            heading_cos_accum += np.cos(particle.heading) * particle.weight
            heading_sin_accum += np.sin(particle.heading) * particle.weight

        if weight_accum == 0:
            raise RuntimeError("sum of particle weights are zero!")

        x_estimate = x_accum / weight_accum
        y_estimate = y_accum / weight_accum
        heading_sin_accum = heading_sin_accum / weight_accum
        heading_cos_accum = heading_cos_accum / weight_accum
        heading_estimate = np.arctan2(heading_sin_accum, heading_cos_accum) * 180 / np.pi

        turtle.color('orange')
        turtle.setposition(x_estimate, y_estimate)
        turtle.setheading(heading_estimate)
        turtle.shape('turtle')
        turtle.stamp()
        turtle.update()
        return [x_estimate, y_estimate, heading_estimate]

    def show_robot(self, robot):
        turtle.color('green')
        turtle.shape('turtle')
        turtle.shapesize(0.7, 0.7)
        turtle.setposition((robot.x, robot.y))
        turtle.setheading((robot.heading*180/np.pi)%360)
        turtle.stamp()
        turtle.update()

    def clear_objects(self):
        turtle.clearstamps()

    def sensor_model(self, coordinates, sensor_limit, orientation=0):
        x, y = coordinates

        # Measure distance between wall and vehicle in front direction
        pos_x = x
        pos_y = y
        front = 0
        dx = np.cos(orientation) * 1 - np.sin(orientation) * 0
        dy = np.sin(orientation) * 1 + np.cos(orientation) * 0
        while not self.colide_wall(int(round(pos_y)),int(round(pos_x))) and front < sensor_limit:
            pos_x = pos_x + dx
            pos_y = pos_y + dy
            front += 1

        # Measure distance between wall and vehicle in right direction
        pos_x = x
        pos_y = y
        right = 0
        dx = np.cos(orientation-np.pi/2) * 1 - np.sin(orientation-np.pi/2) * 0
        dy = np.sin(orientation-np.pi/2) * 1 + np.cos(orientation-np.pi/2) * 0
        while not self.colide_wall(int(round(pos_y)),int(round(pos_x))) and right < sensor_limit:
            pos_x = pos_x + dx
            pos_y = pos_y + dy
            right += 1

        # Measure distance between wall and vehicle in rear direction
        pos_x = x
        pos_y = y
        rear = 0
        dx = np.cos(orientation-np.pi) * 1 - np.sin(orientation-np.pi) * 0
        dy = np.sin(orientation-np.pi) * 1 + np.cos(orientation-np.pi) * 0
        while not self.colide_wall(int(round(pos_y)),int(round(pos_x))) and rear < sensor_limit:
            pos_x = pos_x + dx
            pos_y = pos_y + dy
            rear += 1

        # Measure distance between wall and vehicle in left direction
        pos_x = x
        pos_y = y
        left = 0
        dx = np.cos(orientation+np.pi/2) * 1 - np.sin(orientation+np.pi/2) * 0
        dy = np.sin(orientation+np.pi/2) * 1 + np.cos(orientation+np.pi/2) * 0
        while not self.colide_wall(int(round(pos_y)),int(round(pos_x))) and left < sensor_limit:
            pos_x = pos_x + dx
            pos_y = pos_y + dy
            left += 1
        
        if not self.extensive:
            return [
                front * 100,
                right * 100,
                rear * 100,
                left * 100
            ]

        #### TODO ####
        # Add the 4 additional sensor directions
        # Hint: look at above code for inspiration

        # Measure distance between wall and vehicle in front-left direction (45 degrees)
        pos_x = x
        pos_y = y
        front_left = 0
        dx = np.cos(orientation + np.pi/4) * 1 - np.sin(orientation + np.pi/4) * 0
        dy = np.sin(orientation + np.pi/4) * 1 + np.cos(orientation + np.pi/4) * 0
        while not self.colide_wall(int(round(pos_y)), int(round(pos_x))) and front_left < sensor_limit:
            pos_x = pos_x + dx
            pos_y = pos_y + dy
            front_left += 1

        # Measure distance between wall and vehicle in front-right direction (-45 degrees)
        pos_x = x
        pos_y = y
        front_right = 0
        dx = np.cos(orientation - np.pi/4) * 1 - np.sin(orientation - np.pi/4) * 0
        dy = np.sin(orientation - np.pi/4) * 1 + np.cos(orientation - np.pi/4) * 0
        while not self.colide_wall(int(round(pos_y)), int(round(pos_x))) and front_right < sensor_limit:
            pos_x = pos_x + dx
            pos_y = pos_y + dy
            front_right += 1

        # Measure distance between wall and vehicle in rear-left direction (135 degrees)
        pos_x = x
        pos_y = y
        rear_left = 0
        dx = np.cos(orientation + 3*np.pi/4) * 1 - np.sin(orientation + 3*np.pi/4) * 0
        dy = np.sin(orientation + 3*np.pi/4) * 1 + np.cos(orientation + 3*np.pi/4) * 0
        while not self.colide_wall(int(round(pos_y)), int(round(pos_x))) and rear_left < sensor_limit:
            pos_x = pos_x + dx
            pos_y = pos_y + dy
            rear_left += 1

        # Measure distance between wall and vehicle in rear-right direction (-135 degrees)
        pos_x = x
        pos_y = y
        rear_right = 0
        dx = np.cos(orientation - 3*np.pi/4) * 1 - np.sin(orientation - 3*np.pi/4) * 0
        dy = np.sin(orientation - 3*np.pi/4) * 1 + np.cos(orientation - 3*np.pi/4) * 0
        while not self.colide_wall(int(round(pos_y)), int(round(pos_x))) and rear_right < sensor_limit:
            pos_x = pos_x + dx
            pos_y = pos_y + dy
            rear_right += 1

        #### END ####

        return [
            front * 100,
            right * 100,
            rear * 100,
            left * 100,
            front_left * 100,
            front_right * 100,
            rear_left * 100,
            rear_right * 100
        ]            


class Particle():
    def __init__(
            self,
            x,
            y,
            maze,
            heading=None,
            weight=1.0,
            sensor_limit=None,
            noisy=False,
            gps_x_std=4,
            gps_y_std=4,
            gps_heading_std=4,
            gps_update=0.5):

        if heading is None:
            heading = np.random.uniform(0, 2 * np.pi)

        self.x = x
        self.y = y
        self.heading = heading
        self.weight = weight
        self.maze = maze
        self.sensor_limit = sensor_limit
        self.gps_x_std = gps_x_std
        self.gps_y_std = gps_y_std
        self.gps_heading_std = gps_heading_std
        self.gps_update = gps_update

        # Add random noise to the particle at initialization
        if noisy:
            std = 0.05
            self.x = self.add_noise(x=self.x, std=std)
            self.y = self.add_noise(x=self.y, std=std)
            self.heading = self.add_noise(x=self.heading, std=np.pi * 2 * 0.05)

        # Fix invalizd particles outside the map
        self.fix_invalid_particles()


    def fix_invalid_particles(self):

        # Fix invalid particles
        if self.x < 0:
            self.x = 0
        if self.x > self.maze.width:
            self.x = self.maze.width * 0.9999
        if self.y < 0:
            self.y = 0
        if self.y > self.maze.height:
            self.y = self.maze.height * 0.9999
        self.heading = self.heading % (np.pi*2)

    @property
    def state(self):
        return (self.x, self.y, self.heading)

    def add_noise(self, x, std):
        return x + np.random.normal(0, std)

    def read_sensor(self):
        return self.maze.sensor_model(
            coordinates=(self.x, self.y),
            orientation=self.heading,
            sensor_limit=self.sensor_limit
        )

    def try_move(self, offset, maze, noisy = False):
        curr_theta = self.heading
        curr_x = self.x
        curr_y = self.y
        dx = offset[0]
        dy = offset[1]
        dtheta = offset[2]
        dpos = np.sqrt(dx**2+dy**2)
        dtheta_world = offset[2] + curr_theta
        self.heading = (curr_theta+dtheta)%(np.pi*2)
        x = dpos*np.cos(dtheta_world)+curr_x
        y = dpos*np.sin(dtheta_world)+curr_y

        gj1 = int(self.x)
        gi1 = int(self.y)
        gj2 = int(x)
        gi2 = int(y)

        # Check if the particle is still in the maze
        if gi2 < 0 or gi2 >= maze.num_rows or gj2 < 0 or gj2 >= maze.num_cols:
            self.x = np.random.uniform(0, maze.width)
            self.y = np.random.uniform(0, maze.height)
            return False

        self.x = x
        self.y = y
        return True

class Robot(Particle):
    def __init__(
            self,
            x,
            y,
            maze,
            heading=None,
            sensor_limit=None,
            noisy=True,
            measurement_noise=False,
            extensive=False,
            node=None):

        super().__init__(
            x=x,
            y=y,
            maze=maze,
            heading=heading,
            sensor_limit=sensor_limit,
            noisy=noisy)
        
        # Create our own ROS2 node for service calls if none provided
        if node is None:
            if not rclpy.ok():
                rclpy.init()
            self.node = rclpy.create_node('robot_state_node')
            self._own_node = True
        else:
            self.node = node
            self._own_node = False
            
        # Create service client for getting entity state
        self._get_entity_client = self.node.create_client(GetEntityState, '/get_entity_state')
        
        # The actual Lidar mounted on the vehicle
        self.lidar = LidarProcessing(
            resolution=0.1,
            side_range=(-sensor_limit, sensor_limit),
            fwd_range=(-sensor_limit, sensor_limit),
            height_range=(-1.5, 0.5),
            extensive=extensive,
            node=self.node)
        
        self.measurement_noise = measurement_noise

        self._last = time.time()

    def getModelState(self):
        try:
            # Wait for service to be available
            if not self._get_entity_client.wait_for_service(timeout_sec=0.5):
                self.node.get_logger().info("Service /get_entity_state not available")
                return None
            
            # Create request
            request = GetEntityState.Request()
            request.name = 'gem'
            request.reference_frame = 'world'

            # Call service
            future = self._get_entity_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec = 0.5)
            # Return result
            if future.result() is not None and future.result().success:
                return future.result()
            else:
                self.node.get_logger().info("Failed to get entity state")
                return None
        
        except Exception as exc:
            self.node.get_logger().info(f"Service did not process request: {str(exc)}")
            return None
    
    def __del__(self):
        # Clean up the node if we created it
        if hasattr(self, '_own_node') and self._own_node:
            try:
                self.node.destroy_node()
            except:
                pass

    def read_sensor(self):
        # Read
        curr_state = self.getModelState()
        lidar_reading = self.lidar.processLidar()
        
        # Check if we have valid state data
        if curr_state is None:
            # If no state available, use current particle state
            x = self.x + self.maze.x_start - 100
            y = self.y + self.maze.y_start - 100
            self.heading = self.heading % (2*np.pi)
        else:
            # Use the actual robot state from Gazebo
            # GetEntityState response has state.pose, not just pose
            x = curr_state.state.pose.position.x
            y = curr_state.state.pose.position.y
            euler = self.quaternion_to_euler(curr_state.state.pose.orientation.x,
                                        curr_state.state.pose.orientation.y,
                                        curr_state.state.pose.orientation.z,
                                        curr_state.state.pose.orientation.w)
            self.heading = euler[2] % (2*np.pi)
            self.x = (x+100-self.maze.x_start)
            self.y = (y+100-self.maze.y_start)
        
        # If the lidar measurements are missing with 0.5 probability.
        if self.measurement_noise and np.random.random() < 0.5:
            lidar_reading = None

        if time.time() - self._last > 1 / self.gps_update:
            gps_reading = np.array([self.x, self.y, self.heading])
            gps_reading[0] += np.random.normal(0, self.gps_x_std)
            gps_reading[1] += np.random.normal(0, self.gps_y_std)
            gps_reading[2] = (gps_reading[2] + np.random.normal(0, self.gps_heading_std)) % (2 * np.pi)
            self._last = time.time()
        else:
            gps_reading = None

        return lidar_reading, gps_reading

    def quaternion_to_euler(self, x, y, z, w):
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

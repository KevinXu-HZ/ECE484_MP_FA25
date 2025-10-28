import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetEntityState, SetEntityState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode
import math
import random

def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    
    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]

class ParticleFilter:
    def __init__(self, bob, world, num_particles, sensor_limit, x_start, y_start, node):
        self.num_particles = num_particles  # The number of particles for the particle filter
        self.sensor_limit = sensor_limit    # The sensor limit of the sensor
        self.node = node                    # ROS 2 node for communication
        particles = []
        self.gps_reading = None

        for _ in range(num_particles):
            x = np.random.uniform(0, world.width)
            y = np.random.uniform(0, world.height)
            particles.append(Particle(x=x, y=y, maze=world, sensor_limit=sensor_limit))

        self.particles = particles          # Randomly assign particles at the begining
        self.bob = bob                      # The estimated robot state
        self.world = world                  # The map of the maze
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.set_entity_state_client = self.node.create_client(SetEntityState, '/set_entity_state')
        self.controlSub = self.node.create_subscription(Float32MultiArray, "/gem/control", self.__controlHandler, 10)
        self.get_model_state_client = self.node.create_client(GetEntityState, '/get_entity_state')
        self.control = []                   # A list of control signal from the vehicle

    def __controlHandler(self,data):
        """
        Description:
            Subscriber callback for /gem/control. Store control input from gem controller to be used in particleMotionModel.
        """
        tmp = list(data.data)
        self.control.append(tmp)

    def getModelState(self):
        """
        Description:
            Requests the current state of the polaris model when called
        Returns:
            modelState: contains the current model state of the polaris vehicle in gazebo
        """

        while not self.get_model_state_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Service not available, waiting again...')
            
        request = GetEntityState.Request()
        request.name = 'gem'
        
        try:
            future = self.get_model_state_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future)
            response = future.result()
            return response
        except Exception as e:
            self.node.get_logger().error(f"Service call failed: {e}")
            return None

    def updateWeight(self, lidar_readings):
        if lidar_readings is None:
            return
        
        #### TODO ####
        # Update the weight of each particle according to some function
        # (perhaps a gaussian kernel) that computes the score for each
        # particles' lidar measurement vs the lidar measurement from the robot.
        #
        # Make sure that the sum of all particle weights adds up to 1
        # after updating the weights.


        raise NotImplementedError("implement this!!!")


        #### END ####

    def resampleParticle(self):
        new_particles = []

        #### TODO ####
        # Resample current particles to generate a new set of particles.
        #
        # Things to consider:
        #   -   Resample particles based on the weight of each particle
        #
        #   -   If all the particles bunch up, then we will be stuck in a
        #       non-optimal solution. We can't be too certain! How can we
        #       mitigate this?
        #
        #   -   For problem [10] bonus points use gps measurements to get 
        #       a "rough" estimate of where you are. How can we use this
        #       to make sure the particle filter does not converge / stay
        #       in a non-optimal solution??
        #
        #       gps_x       = self.gps_reading[0]
        #       gps_y       = self.gps_reading[1]
        #       gps_heading = self.gps_reading[2]

        
        raise NotImplementedError("implement this!!!")


        #### END ####

        self.particles = new_particles

    def particleMotionModel(self):
        dt = 0.01   # might need adjusting depending on compute performance

        #### TODO ####
        # Estimate next state for each particle according to the control
        # input from the actual robot.
        # 
        # You can use an ODE function or the vehicle_dynamics function
        # provided at the top of this file.

        
        # raise NotImplementedError("implement this!!!")
        for particle in self.particles:
            vars = [particle.x, particle.y, particle.heading]

            for input in self.control:
                v = input[0]
                delta = input[1]

                derivatives = vehicle_dynamics(0,vars,v,delta)

                vars[0] += derivatives[0] * dt
                vars[1] += derivatives[1] * dt
                vars[2] += derivatives[2] * dt

                vars[2] = (vars[2] + np.pi)%(np.pi*2)-np.pi

            particle.x = vars[0]
            particle.y = vars[1]
            particle.heading = vars[2]
        #### END ####

        self.control = []


    def runFilter(self, show_frequency):
        """
        Description:
            Run PF localization
        """
        self.world.clear_objects()
        self.world.show_particles(self.particles, show_frequency=show_frequency)
        self.world.show_robot(self.bob)
        count = 0 
        while rclpy.ok():
            lidar_reading, gps_reading = self.bob.read_sensor()

            # ensure at least one positive gps reading before running the filter
            if gps_reading is not None:
                self.gps_reading = gps_reading
            if self.gps_reading is None:
                continue

            # if no control inputs have arrived, do nothing
            if len(self.control) == 0:
                continue

            #### TODO ####
            # 1. perform a particle motion step
            # 2. update weights based on measurements
            # 3. resample particles
            #
            # Hint: use class helper functions
            




            #### END ####

            if count % 2 == 0:
                #### TODO ####
                # Re-render world, make sure to clear previous objects first!





                #### END ####

                estimated_location = self.world.show_estimated_location(self.particles)
                err = math.sqrt((estimated_location[0] - self.bob.x) ** 2 + (estimated_location[1] - self.bob.y) ** 2)
                print(f":: step {count} :: err {err:.3f}")
            count += 1

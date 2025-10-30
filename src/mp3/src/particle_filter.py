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
        
        #### Measurement Update ####
        # Update the weight of each particle according to some function
        # (perhaps a gaussian kernel) that computes the score for each
        # particles' lidar measurement vs the lidar measurement from the robot.
        #
        # Make sure that the sum of all particle weights adds up to 1
        # after updating the weights.
        measurement = np.array(lidar_readings, dtype=np.float64)
        measurement = np.nan_to_num(
            measurement,
            nan=self.sensor_limit * 100.0,
            posinf=self.sensor_limit * 100.0,
            neginf=0.0
        )

        measurement_std = max(self.sensor_limit * 100.0 * 0.1, 1.0)
        inv_two_sigma_sq = 0.5 / (measurement_std ** 2)

        weights = []
        for particle in self.particles:
            expected = np.array(particle.read_sensor(), dtype=np.float64)
            expected = np.nan_to_num(
                expected,
                nan=self.sensor_limit * 100.0,
                posinf=self.sensor_limit * 100.0,
                neginf=0.0
            )
            error = measurement - expected
            squared_error = np.sum(error ** 2)
            weight = math.exp(-squared_error * inv_two_sigma_sq)
            if not np.isfinite(weight) or weight <= 0.0:
                weight = 1e-12
            particle.weight = weight
            weights.append(weight)

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            uniform_weight = 1.0 / self.num_particles
            for particle in self.particles:
                particle.weight = uniform_weight
        else:
            inv_total = 1.0 / weight_sum
            for particle in self.particles:
                particle.weight *= inv_total
        #### END ####

    def resampleParticle(self):
        new_particles = []

        #### Resampling Step ####
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

        weights = np.array([particle.weight for particle in self.particles], dtype=np.float64)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles
        else:
            weights /= weight_sum

        cumulative = np.cumsum(weights)
        step = 1.0 / self.num_particles
        start = random.random() * step
        positions = start + step * np.arange(self.num_particles)
        indices = np.searchsorted(cumulative, positions, side='right')

        for idx in indices:
            idx = min(idx, self.num_particles - 1)
            source = self.particles[idx]
            particle = Particle(
                x=source.x,
                y=source.y,
                heading=source.heading,
                maze=self.world,
                weight=1.0,
                sensor_limit=self.sensor_limit,
                noisy=False,
                gps_x_std=source.gps_x_std,
                gps_y_std=source.gps_y_std,
                gps_heading_std=source.gps_heading_std,
                gps_update=source.gps_update
            )

            particle.x += np.random.normal(0.0, 0.25)
            particle.y += np.random.normal(0.0, 0.25)
            particle.heading = (particle.heading + np.random.normal(0.0, np.deg2rad(3.0))) % (2 * np.pi)
            particle.fix_invalid_particles()
            particle.weight = 1.0 / self.num_particles
            new_particles.append(particle)

        if self.gps_reading is not None:
            effective_sample_size = 1.0 / np.sum((weights + 1e-16) ** 2)
            if effective_sample_size < 0.6 * self.num_particles:
                gps_x, gps_y, gps_heading = self.gps_reading
                gps_x_std = getattr(self.bob, "gps_x_std", 5.0)
                gps_y_std = getattr(self.bob, "gps_y_std", 5.0)
                gps_heading_std = getattr(self.bob, "gps_heading_std", np.deg2rad(15.0))

                num_reseed = max(1, int(0.05 * self.num_particles))
                reseed_indices = np.random.choice(self.num_particles, size=num_reseed, replace=False)
                for idx in reseed_indices:
                    particle = new_particles[idx]
                    particle.x = np.random.normal(gps_x, gps_x_std)
                    particle.y = np.random.normal(gps_y, gps_y_std)
                    particle.heading = (np.random.normal(gps_heading, gps_heading_std)) % (2 * np.pi)
                    particle.fix_invalid_particles()
        #### END ####

        self.particles = new_particles

    def particleMotionModel(self):
        dt = 0.01   # might need adjusting depending on compute performance

        #### Motion Model ####
        # Estimate next state for each particle according to the control
        # input from the actual robot.
        # 
        # You can use an ODE function or the vehicle_dynamics function
        # provided at the top of this file.

        if len(self.control) == 0:
            return

        speed_noise = 0.2
        steer_noise = np.deg2rad(1.0)
        drift_noise = 0.05
        heading_drift_noise = np.deg2rad(0.5)

        for particle in self.particles:
            x, y, heading = particle.x, particle.y, particle.heading
            for vr, delta in self.control:
                noisy_vr = vr + np.random.normal(0.0, speed_noise + 0.05 * abs(vr))
                noisy_delta = delta + np.random.normal(0.0, steer_noise + 0.05 * abs(delta))

                heading = (heading + noisy_delta * dt) % (2 * np.pi)
                distance = noisy_vr * dt
                x += distance * np.cos(heading)
                y += distance * np.sin(heading)

            x += np.random.normal(0.0, drift_noise)
            y += np.random.normal(0.0, drift_noise)
            heading = (heading + np.random.normal(0.0, heading_drift_noise)) % (2 * np.pi)

            particle.x = x
            particle.y = y
            particle.heading = heading
            particle.fix_invalid_particles()

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

            #### Filter Step ####
            # 1. perform a particle motion step
            # 2. update weights based on measurements
            # 3. resample particles
            #
            # Hint: use class helper functions
            self.particleMotionModel()
            if lidar_reading is not None:
                self.updateWeight(lidar_reading)
                self.resampleParticle()
            #### END ####

            if count % 2 == 0:
                #### Rendering ####
                # Re-render world, make sure to clear previous objects first!
                self.world.clear_objects()
                self.world.show_particles(self.particles, show_frequency=show_frequency)
                self.world.show_robot(self.bob)
                #### END ####

                estimated_location = self.world.show_estimated_location(self.particles)
                err = math.sqrt((estimated_location[0] - self.bob.x) ** 2 + (estimated_location[1] - self.bob.y) ** 2)
                print(f":: step {count} :: err {err:.3f}")
            count += 1

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

        # Convert readings to meters for a more interpretable noise scale
        lidar_arr = np.array(lidar_readings, dtype=np.float64) / 100.0
        sigma_m = 1.0  # Balanced for good weight discrimination
        sigma_sq = sigma_m ** 2

        # Update weight for each particle
        for particle in self.particles:
            # Check if particle is outside map bounds or inside a wall/obstacle
            out_of_bounds = (particle.x < 0 or particle.x >= self.world.width or
                           particle.y < 0 or particle.y >= self.world.height)

            if out_of_bounds:
                # Particle is outside the map
                particle.weight = 1e-100
            else:
                # Particle is within bounds, check if it's inside a wall
                ix = int(particle.x)
                iy = int(particle.y)
                if self.world.colide_wall(iy, ix):
                    # Particle is inside a wall
                    particle.weight = 1e-100
                else:
                    # Normal weight update based on sensor measurement
                    particle_readings = np.array(particle.read_sensor(), dtype=np.float64) / 100.0
                    diff = lidar_arr - particle_readings
                    squared_diff = np.dot(diff, diff)

                    # Use Gaussian kernel to compute weight
                    particle.weight = math.exp(-0.5 * squared_diff / sigma_sq)

        # Normalize weights so they sum to 1
        total_weight = sum(p.weight for p in self.particles)

        # Avoid division by zero
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            uniform_weight = 1.0 / len(self.particles)
            for particle in self.particles:
                particle.weight = uniform_weight

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

        # Low-variance resampling method with random particle injection
        # More stable than multinomial, reduces particle depletion
        N = self.num_particles
        part_array = self.particles

        # Calculate how many particles to sample vs inject randomly
        # Inject 5% random particles to prevent particle depletion
        num_random = max(int(N * 0.05), 1)
        num_resampled = N - num_random

        # Step 1: Build cumulative sum of weights
        cum_weights = np.zeros(N)
        cum_sum = 0.0
        for i in range(N):
            cum_sum += part_array[i].weight
            cum_weights[i] = cum_sum

        # Normalize to ensure it sums to exactly 1
        if cum_sum > 0:
            cum_weights = cum_weights / cum_sum
        else:
            cum_weights = np.linspace(0, 1, N)

        # Step 2: Low-variance resampling
        # Start at a random position and step through deterministically
        r = np.random.uniform(0, 1.0 / num_resampled)
        c = cum_weights[0]
        i = 0

        for m in range(num_resampled):
            u = r + m / num_resampled
            while u > c and i < N - 1:
                i += 1
                c = cum_weights[i]

            # Sample particle i with noise
            target_x = part_array[i].x
            target_y = part_array[i].y
            target_heading = part_array[i].heading
            new_particles.append(Particle(x = target_x, y = target_y, heading = target_heading,
                                        maze = self.world, weight = 1.0/N,
                                        sensor_limit = self.sensor_limit, noisy = True))

        # Step 3: Add random particles for diversity (prevents particle depletion)
        for _ in range(num_random):
            random_x = np.random.uniform(0, self.world.width)
            random_y = np.random.uniform(0, self.world.height)
            random_heading = np.random.uniform(0, 2 * np.pi)
            new_particles.append(Particle(x = random_x, y = random_y, heading = random_heading,
                                        maze = self.world, weight = 1.0/N,
                                        sensor_limit = self.sensor_limit, noisy = False))

        # raise NotImplementedError("implement this!!!")
        # for i,p in enumerate(self.particles):
        #     print(f"init[{i:03d}] x={p.x:.3f}, y={p.y:.3f}")

        #### END ####

        self.particles = new_particles
        # print([(p.x, p.y) for p in self.particles])
    
    def particleMotionModel(self):
        dt = 0.01   # control inputs arrive at 0.01s timestep (100 Hz)
        trans_noise_std = 0.15  # Balanced for tracking without too much drift
        rot_noise_std = np.deg2rad(8)  # Reduced to prevent divergence

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

            particle.x = vars[0] + np.random.normal(0, trans_noise_std)
            particle.y = vars[1] + np.random.normal(0, trans_noise_std)
            particle.heading = vars[2] + np.random.normal(0, rot_noise_std)

            # Ensure particle stays within map bounds after propagation
            particle.fix_invalid_particles()

            # If particle collides with a wall, it will get low weight and die out naturally
            # Don't randomly resample as that destroys the particle filter's convergence
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

            # Perform one full particle filter update cycle
            self.particleMotionModel()  # Propagate particles based on control inputs
            self.updateWeight(lidar_reading)  # Update weights based on sensor measurements
            self.resampleParticle()  # Resample particles based on weights

            # Update visualization after each complete cycle
            if count % 2 == 0:
                # Refresh visualization to reflect latest particle set
                self.world.clear_objects()
                self.world.show_particles(self.particles, show_frequency=show_frequency)
                self.world.show_robot(self.bob)

                estimated_location = self.world.show_estimated_location(self.particles)
                err = math.sqrt((estimated_location[0] - self.bob.x) ** 2 + (estimated_location[1] - self.bob.y) ** 2)
                # Debug output to verify particles exist
                avg_x = np.mean([p.x for p in self.particles])
                avg_y = np.mean([p.y for p in self.particles])
                print(f":: step {count} :: particles: {len(self.particles)}, avg_pos: ({avg_x:.1f}, {avg_y:.1f}), err {err:.3f}")
            count += 1

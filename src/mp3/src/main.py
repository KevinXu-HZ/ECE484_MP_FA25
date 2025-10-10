#!/usr/bin/env python3

import numpy as np 
import turtle
import argparse
import pickle
import rclpy
from rclpy.node import Node
from maze import Maze, Robot
from particle_filter import ParticleFilter
from ament_index_python.packages import get_package_share_directory
import os


class NavigatorNode(Node):
    def __init__(self):
        super().__init__('navigator')
        self.get_logger().info('Navigator node initialized')

def main(argv):
    rclpy.init()
    node = NavigatorNode()

    window = turtle.Screen()
    window.setup(width=argv.window_width, height=argv.window_height)

    # Creating the python map for the ECEB environment
    maze = np.zeros((200,200))
    
    # Load in obstacle data
    mp3_pkg_dir = get_package_share_directory('mp3')
    data_path = os.path.join(mp3_pkg_dir, 'data', 'obstacle_list.data')
    
    try:
        with open(data_path, 'rb') as filehandle:
            obstacle = pickle.load(filehandle)
    except FileNotFoundError:
        with open('obstacle_list.data', 'rb') as filehandle:
            obstacle = pickle.load(filehandle)

    # Decode obstacle data
    for (x,y) in obstacle:
        maze[y + 100, x + 100] = 1

    y_start = 100
    x_start = 15
    width = 120
    height = 75
    maze_ted = np.zeros((height, width), np.int8)
    for i in range(y_start, y_start + height):
        for j in range(x_start, x_start + width):
            if maze[i, j] == 1:
                    maze_ted[i - y_start, j - x_start] |= 15
            else:
                if (i == 0):
                    maze_ted[i - y_start, j - x_start] |= 1
                elif (i == maze.shape[1] - 1):
                    maze_ted[i - y_start, j - x_start] |= 4
                else:
                    if maze[i + 1, j] == 1:
                        maze_ted[i - y_start, j - x_start] |= 4
                    if maze[i - 1, j] == 1:
                        maze_ted[i - y_start, j - x_start] |= 1
                
                if (j == 0):
                    maze_ted[i - y_start, j - x_start] |= 8
                elif(j == maze.shape[1] - 1):
                    maze_ted[i - y_start, j - x_start] |= 2
                else:
                    if maze[i, j + 1] == 1:
                        maze_ted[i - y_start, j - x_start] |= 2
                    if maze[i, j - 1] == 1:
                        maze_ted[i - y_start, j - x_start] |= 8
    world = Maze(
        maze=maze_ted,
        x_start=x_start,
        y_start=y_start,
        extensive=argv.extensive_lidar)
    world.show_maze()

    bob = Robot(
        x=0,
        y=0,
        heading=0,
        maze=world,
        sensor_limit=argv.sensor_limit,
        measurement_noise=argv.measurement_noise,
        extensive=argv.extensive_lidar,
        node=node)

    ParticleFilter(
        bob=bob,
        world=world,
        num_particles=argv.num_particles,
        sensor_limit=argv.sensor_limit,
        x_start=x_start,
        y_start=y_start,
        node=node).runFilter(argv.show_frequency)
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Particle filter in maze.')

    parser.add_argument(
        '--window_width',
        type=int,
        help='Width of viewing window.',
        default=1280)
    
    parser.add_argument(
        '--window_height',
        type=int,
        help='Height of viewing window.',
        default=720)
    
    parser.add_argument(
        '--num_particles',
        type=int,
        help='Number of particles used in particle filter.',
        default=250)
    
    parser.add_argument(
        '--sensor_limit',
        type=float,
        help='The distance in Gazebo the sensor can sense.',
        default=15)

    parser.add_argument(
        '--measurement_noise',
        help='Adding noise to the lidar measurement.',
        action='store_true')

    parser.add_argument(
        '--gps_x_std',
        type=float,
        help="The standard deviation of gaussian noise added to the gps_x signal.",
        default=5)

    parser.add_argument(
        '--gps_y_std',
        type=float,
        help="The standard deviation of gaussian noise added to the gps_y signal.",
        default=5)
    
    parser.add_argument(
        '--gps_heading_std',
        type=float,
        help="The standard deviation of gaussian noise added to the gps_heading signal.",
        default=np.pi/8)

    parser.add_argument(
        '--gps_update',
        type=float,
        help="Update rate of the gps sensor (Hz).",
        default=1)

    parser.add_argument(
        '--extensive_lidar',
        help="Uses 8 lidar beams instead of 4.",
        action='store_true')

    parser.add_argument(
        '--show_frequency',
        type=int,
        help="Number of particles to display during rendering.",
        default=5)

    main(parser.parse_args())
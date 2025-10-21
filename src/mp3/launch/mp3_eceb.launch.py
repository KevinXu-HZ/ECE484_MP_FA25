#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Package directories
    mp3_pkg_dir = get_package_share_directory('mp3')
    
    # Launch arguments
    paused_arg = DeclareLaunchArgument(
        'paused',
        default_value='false',
        description='Start Gazebo paused'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Start Gazebo GUI'
    )
    
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Start Gazebo headless'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Debug mode'
    )
    
    # Include Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('mp3'),
                'worlds',
                'mp3_eceb.world'
            ]),
            'gui': LaunchConfiguration('gui'),
            'paused': LaunchConfiguration('paused'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'verbose': 'false',
        }.items()
    )
    
    # Robot description parameter
    robot_description = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': PathJoinSubstitution([
                FindPackageShare('mp3'),
                'urdf',
                'polaris.urdf'
            ])
        }],
        output='screen'
    )
    
    # Spawn polaris robot
    spawn_polaris = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'polaris',
            '-topic', 'robot_description',
            '-x', '15',
            '-y', '45',
            '-z', '1',
            '-Y', '1.57079632679'
        ],
        output='screen'
    )
    
    # Spawn marker
    spawn_marker = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'marker',
            '-file', PathJoinSubstitution([
                FindPackageShare('mp3'),
                'models',
                'marker',
                'model.sdf'
            ]),
            '-x', '15',
            '-y', '45',
            '-z', '0'
        ],
        output='screen'
    )
    
    # Vehicle movement node
    vehicle_node = Node(
        package='mp3',
        executable='vehicle.py',
        name='vehicle_node',
        output='screen'
    )
    
    return LaunchDescription([
        paused_arg,
        use_sim_time_arg,
        gui_arg,
        headless_arg,
        debug_arg,
        
        gazebo_launch,
        robot_description,
        spawn_polaris,
        spawn_marker,
        vehicle_node,
    ]) 
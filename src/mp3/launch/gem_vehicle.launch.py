#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Package directories
    mp3_pkg_dir = get_package_share_directory('mp3')
    gem_description_pkg_dir = get_package_share_directory('gem_description')
    gazebo_ros_pkg_dir = get_package_share_directory('gazebo_ros')
    
    # Launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='/',
        description='Namespace for the robot'
    )
    
    cmd_timeout_arg = DeclareLaunchArgument(
        'cmd_timeout',
        default_value='4.0',
        description='Command timeout'
    )
    
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
    
    verbose_arg = DeclareLaunchArgument(
        'verbose',
        default_value='false',
        description='Verbose output'
    )
    
    world_name_arg = DeclareLaunchArgument(
        'world_name',
        default_value='smaller_track.world',
        description='World file name'
    )
    
    # Vehicle pose arguments
    x_arg = DeclareLaunchArgument(
        'x',
        default_value='15.0',
        description='Initial x position'
    )
    
    y_arg = DeclareLaunchArgument(
        'y',
        default_value='45.0',
        description='Initial y position'
    )
    
    z_arg = DeclareLaunchArgument(
        'z',
        default_value='10.0',
        description='Initial z position'
    )
    
    yaw_arg = DeclareLaunchArgument(
        'yaw',
        default_value='1.57079632679',
        description='Initial yaw orientation'
    )
    
    # Include GEM description launch (XML format)
    gem_description_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gem_description'),
                'launch',
                'gem_description.launch'
            ])
        ]),
        launch_arguments={
            'namespace': LaunchConfiguration('namespace')
        }.items()
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
            'verbose': LaunchConfiguration('verbose'),
        }.items()
    )
    
    # Spawn robot
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'gem',
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-Y', LaunchConfiguration('yaw'),
        ],
        output='screen',
    )
    
    # Include GEM odometry publisher
    gem_odometry_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gem_gazebo'),
                'launch',
                'odometry_publisher.launch.py'
            ])
        ]),
        launch_arguments={
            'entity_name': 'gem',
            'publish_rate': '50.0',
            'use_entity_state': 'true'
        }.items()
    )
    
    # Static transform from map to world (identity transform)
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'world'],
        output='screen'
    )
    
    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{
            'rate': 0.1,
            'use_gui': False
        }],
        output='screen'
    )
    
    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('mp3'),
            'config',
            'mp3.rviz'

        ])],
        condition=IfCondition(LaunchConfiguration('gui')),
        output='screen'
    )
    

    return LaunchDescription([
        namespace_arg,
        cmd_timeout_arg,
        paused_arg,
        use_sim_time_arg,
        gui_arg,
        headless_arg,
        debug_arg,
        verbose_arg,
        world_name_arg,
        x_arg,
        y_arg,
        z_arg,
        yaw_arg,
        
        gem_description_launch,
        gazebo_launch,
        spawn_robot,
        gem_odometry_launch,
        static_transform_publisher,
        joint_state_publisher,
        rviz,
    ]) 
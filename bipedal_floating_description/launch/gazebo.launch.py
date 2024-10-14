from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import (OnExecutionComplete, OnProcessExit,
                                OnProcessIO, OnProcessStart, OnShutdown)


def generate_launch_description():
    use_sim_time_arg = LaunchConfiguration('use_sim_time')  

    share_dir = get_package_share_directory('bipedal_floating_description')

    world = LaunchConfiguration('world')
    world_file_name = 'emptyy.world'
    world_path = os.path.join(share_dir, 'worlds', world_file_name)
    declare_world_cmd = DeclareLaunchArgument(
    name='world',
    default_value=world_path,
    description='Full path to the world model file to load')

    declear_use_sim_time = DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use sim time if true')


    xacro_file = os.path.join(share_dir, 'urdf', 'bipedal_floating.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_urdf = robot_description_config.toxml()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': robot_urdf, 
            'use_sim_time': use_sim_time_arg}
        ]
    )

    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher'
    )

    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gzserver.launch.py'
            ])
        ]),
        launch_arguments={
            'pause': 'true',
            'world': world,
        }.items()
    )
    # pkg_gazebo_ros = FindPackageShare(package='gazebo_ros').find('gazebo_ros')   

    # gazebo_server = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
    #     launch_arguments={'world': world}.items())



    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gzclient.launch.py'
            ])
        ])
    )

    urdf_spawn_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'bipedal_floating',
            '-topic', 'robot_description'
        ],
        output='screen'
    )


    joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active','joint_state_broadcaster'],
        output='screen'
    )

    velocity_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active','velocity_controller'],
        output='screen'
    )

    joint_trajectory_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active','joint_trajectory_controller'],
        output='screen'
    )

    init_pose = ExecuteProcess(
        cmd=[
            "ros2", "topic", "pub", "-1", "/joint_trajectory_controller/joint_trajectory",
            "trajectory_msgs/msg/JointTrajectory",
            "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'base_link'}, joint_names: ['L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 'L_Ankle_Pitch', 'L_Ankle_Roll', 'R_Hip_Roll', 'R_Hip_Yaw', 'R_Knee_Pitch', 'R_Hip_Pitch', 'R_Ankle_Pitch', 'L_Hip_Roll', 'R_Ankle_Roll'], points: [{positions: [0.0, -0.37, 0.74, -0.36, 0.0, 0.0, 0.0, 0.74, -0.37, -0.36, 0.0, 0.0], velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 0, nanosec: 0}}]}"
        ],
        output='screen'
    )

    effort_controllers = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active','effort_controllers'],
        output='screen'
        )




    return LaunchDescription([
        declear_use_sim_time,
        declare_world_cmd,
        robot_state_publisher_node,
        # joint_state_publisher_node,
        gazebo_server,
        gazebo_client,
        urdf_spawn_node,

        RegisterEventHandler(
            OnProcessExit(
                target_action=urdf_spawn_node,
                on_exit=[joint_state_broadcaster]
            )
        ),

        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_state_broadcaster,
                on_exit=[effort_controllers]
            )
        ),

        # RegisterEventHandler(
        #     OnProcessExit(
        #         target_action=joint_state_broadcaster,
        #         on_exit=[velocity_controller]
        #     )
        # ),
        # RegisterEventHandler(
        #     OnProcessExit(
        #         target_action=joint_state_broadcaster,
        #         on_exit=[joint_trajectory_controller]
        #     )
        # ),
        # RegisterEventHandler(
        #     OnProcessExit(
        #         target_action=joint_state_broadcaster,
        #         on_exit=[init_pose]
        #     )
        # ),


    ])

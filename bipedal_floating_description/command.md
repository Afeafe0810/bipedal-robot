ros2 topic pub -1 /joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory  "{
    header: {
        stamp: {sec: 0, nanosec: 0},
        frame_id: 'base_link'
    },
    joint_names: [
        'L_Hip_Roll', 'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch',
        'L_Ankle_Pitch', 'L_Ankle_Roll', 'R_Hip_Roll', 'R_Hip_Yaw',
        'R_Hip_Pitch', 'R_Knee_Pitch', 'R_Ankle_Pitch', 'R_Ankle_Roll'
    ],
    points: [
        {
            positions: [0.0, 0.0, -0.37, 0.74, -0.36, 0.0, 0.0, 0.0, -0.37, 0.74, -0.36, 0.0],
            velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            time_from_start: {sec: 0, nanosec: 0}
        },
    ]
}"


ros2 service call /ATTACHLINK linkattacher_msgs/srv/AttachLink "{model1_name: 'bipedal_floating', link1_name: 'r_foot_1', model2_name: 'ground_plane', link2_name: 'link'}"

ros2 service call /DETACHLINK linkattacher_msgs/srv/DetachLink "{model1_name: 'bipedal_floating', link1_name: 'r_foot_1', model2_name: 'ground_plane', link2_name: 'link'}"



ros2 launch bipedal_floating_description gazebo.launch.py

ros2 run bipedal_floating_description controller


colcon build --packages-select bipedal_floating_description


#order
['L Hip_Yaw', 'L Hip_Pitch', 'L Knee_Pitch', 'L Ankle_Pitch', 'L Ankle Roll', 'R Hip Roll', 'R Hip Yaw', 'R Knee Pitch', 'R Hip Pitch', 'R Ankle Pitch', 'L Hip Roll', 'R Ankle Roll']
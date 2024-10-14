import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray 

from sensor_msgs.msg import JointState

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import pinocchio as pin
import qpsolvers

import pink
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask
import meshcat_shapes
import qpsolvers

import numpy as np
np.set_printoptions(precision=2)

from sys import argv
from os.path import dirname, join, abspath
import os
import copy
import math
from scipy.spatial.transform import Rotation as R

import pandas as pd

class UpperLevelController(Node):

    def __init__(self):
        super().__init__('upper_level_controllers')

        self.position_publisher = self.create_publisher(Float64MultiArray , '/position_controller/commands', 10)
        self.velocity_publisher = self.create_publisher(Float64MultiArray , '/velocity_controller/commands', 10)
        self.effort_publisher = self.create_publisher(Float64MultiArray , '/effort_controllers/commands', 10)
        self.vcmd_publisher = self.create_publisher(Float64MultiArray , '/velocity_command/commands', 10)
        self.l_gravity_publisher = self.create_publisher(Float64MultiArray , '/l_gravity', 10)
        self.r_gravity_publisher = self.create_publisher(Float64MultiArray , '/r_gravity', 10)
        # self.Q0 = np.array([0.0, 0.0, -0.37, 0.74, -0.36, 0.0, 0.0, 0.0, -0.37, 0.74, -0.36, 0.0])
        # self.Q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.37, 0.74, -0.36, 0.0, 0.0, 0.0, -0.37, 0.74, -0.36, 0.0])

        self.joint_trajectory_controller = self.create_publisher(JointTrajectory , '/joint_trajectory_controller/joint_trajectory', 10)

        self.joint_states_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10)
        self.joint_states_subscriber  # prevent unused variable warning

        #joint_state(subscribe data)
        self.jp_sub = np.zeros(12)
        self.jv_sub = np.zeros(12)

        #joint_velocity_cal
        self.joint_position_past = np.zeros((12,1))

        #joint_velocity_filter (jv = after filter)
        self.jv = np.zeros((12,1))
        self.jv_p = np.zeros((12,1))
        self.jv_pp = np.zeros((12,1))
        self.jv_sub_p = np.zeros((12,1))
        self.jv_sub_pp = np.zeros((12,1))



        self.state_subscriber = self.create_subscription(
            Float64MultiArray,
            'state_topic',
            self.state_callback,
            10
        )
        self.state_subscriber  # prevent unused variable warning


        self.robot = self.load_URDF("/home/ldsc/ros2_ws/src/bipedal_floating_description/urdf/bipedal_floating.pin.urdf")
        
        # Initialize meschcat visualizer
        self.viz = pin.visualize.MeshcatVisualizer(
            self.robot.model, self.robot.collision_model, self.robot.visual_model
        )
        self.robot.setVisualizer(self.viz, init=False)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()


        # Set initial robot configuration
        print(self.robot.model)
        print(self.robot.q0)
        self.init_configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)
        self.viz.display(self.init_configuration.q)

        # Tasks initialization for IK
        self.tasks = self.tasks_init()

        self.timer_period = 0.01 # seconds
        self.timer = self.create_timer(self.timer_period, self.main_controller_callback)

        self.state = 0
        self.tt = 0

        #path_data_norman(減過後的軌跡)

        self.L_ref_data = pd.read_csv('/home/ldsc/Path_norman/LX.csv', header=None).values
        self.R_ref_data = pd.read_csv('/home/ldsc/Path_norman/RX.csv', header=None).values
        self.count = 0
        self.stance = 2

        #ALIP
        #--measurement
        self.model_state_x = np.zeros((2,1))
        self.model_state_x_past = np.zeros((2,1))
        self.model_state_y = np.zeros((2,1))
        self.model_state_y_past = np.zeros((2,1))
        #--compensator
        self.ob_xly = np.zeros((2,1))
        self.ob_xly_past = np.zeros((2,1))
        self.ob_ylx = np.zeros((2,1))
        self.ob_ylx_past = np.zeros((2,1))
        #--torque
        self.ap = 0.0
        self.ap_past = 0.0
        self.ar = 0.0
        self.ar_past = 0.0

        
    def load_URDF(self, urdf_path):
        robot = pin.RobotWrapper.BuildFromURDF(
                        filename=urdf_path,
                        package_dirs=["."],
                        # root_joint=pin.JointModelFreeFlyer(),
                        root_joint=None,
                        )
        
        print(f"URDF description successfully loaded in {robot}")

        #左單支撐腳
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/stance_l.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        self.stance_l_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + self.stance_l_model.name)
        # Create data required by the algorithms
        self.stance_l_data = self.stance_l_model.createData()

        #右單支撐腳
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/stance_r_gravity.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        self.stance_r_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + self.stance_r_model.name)
        # Create data required by the algorithms
        self.stance_r_data = self.stance_r_model.createData()

        #雙足模型_以左腳建起
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/bipedal_l_gravity.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        self.bipedal_l_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + self.bipedal_l_model.name)
        # Create data required by the algorithms
        self.bipedal_l_data = self.bipedal_l_model.createData()

        #雙足模型_以右腳建起
        pinocchio_model_dir = "/home/ldsc/ros2_ws/src"
        urdf_filename = pinocchio_model_dir + '/bipedal_floating_description/urdf/bipedal_r_gravity.xacro' if len(argv)<2 else argv[1]
        # Load the urdf model
        self.bipedal_r_model  = pin.buildModelFromUrdf(urdf_filename)
        print('model name: ' + self.bipedal_r_model.name)
        # Create data required by the algorithms
        self.bipedal_r_data = self.bipedal_r_model.createData()

        return robot
        
    def tasks_init(self):
        # Tasks initialization for IK
        left_foot_task = FrameTask(
            "l_foot",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        pelvis_task = FrameTask(
            "base_link",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        right_foot_task = FrameTask(
            "r_foot_1",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        posture_task = PostureTask(
            cost=1e-1,  # [cost] / [rad]
        )
        tasks = {
            # 'left_foot_task': left_foot_task,
            'pelvis_task': pelvis_task,
            # 'right_foot_task': right_foot_task,
            'posture_task': posture_task,
        }
        return tasks

    def state_callback(self,msg):
        
        self.state = msg.data[0]

    def collect_joint_data(self):
        joint_position = copy.deepcopy(self.jp_sub)
        joint_velocity = copy.deepcopy(self.jv_sub)

        joint_position = np.reshape(joint_position,(12,1))
        joint_velocity = np.reshape(joint_velocity,(12,1))

        return joint_position,joint_velocity

    def joint_velocity_cal(self,joint_position):
        joint_position_now = copy.deepcopy(joint_position)
        joint_velocity_cal = (joint_position_now - self.joint_position_past)/self.timer_period
        self.joint_position_past = joint_position_now

        joint_velocity_cal = np.reshape(joint_velocity_cal,(12,1))
        return joint_velocity_cal

    def joint_velocity_filter(self,joint_velocity):
        
        jv_sub = copy.deepcopy(joint_velocity)

        self.jv = 1.1580*self.jv_p - 0.4112*self.jv_pp + 0.1453*self.jv_sub_p + 0.1078*self.jv_sub_pp

        self.jv_pp = copy.deepcopy(self.jv_p)
        self.jv_p = copy.deepcopy(self.jv)
        self.jv_sub_pp = copy.deepcopy(self.jv_sub_p)
        self.jv_sub_p = copy.deepcopy(jv_sub)

        return self.jv

    def joint_states_callback(self, msg):
        
        # Original ndarray order
        original_order = [
            'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 'L_Ankle_Pitch', 
            'L_Ankle_Roll', 'R_Hip_Roll', 'R_Hip_Yaw', 'R_Knee_Pitch', 
            'R_Hip_Pitch', 'R_Ankle_Pitch', 'L_Hip_Roll', 'R_Ankle_Roll'
        ]

        # Desired order
        desired_order = [
            'L_Hip_Roll', 'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 
            'L_Ankle_Pitch', 'L_Ankle_Roll', 'R_Hip_Roll', 'R_Hip_Yaw', 
            'R_Hip_Pitch', 'R_Knee_Pitch', 'R_Ankle_Pitch', 'R_Ankle_Roll'
        ]

        if len(msg.velocity) == 12:
            velocity_order_dict = {joint: value for joint, value in zip(original_order, np.array(msg.velocity))}

            self.jv_sub = np.array([velocity_order_dict[joint] for joint in desired_order])

        if len(msg.position) == 12:
            position_order_dict = {joint: value for joint, value in zip(original_order, np.array(msg.position))}
            self.jp_sub = np.array([position_order_dict[joint] for joint in desired_order])

        self.collect_joint_data()


        # print(msg.position)
        # print(msg.velocity)

        # print("-------")

    def xyz_rotation(self,axis,theta):
        cos = math.cos
        sin = math.sin
        R = np.array((3,3))
        if axis == 'x':
            R = np.array([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]])
        elif axis == 'y':
            R = np.array([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])
        elif axis == 'z':
            R = np.array([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])
        return R    

    def rotation_matrix(self,joint_position):
        jp = copy.deepcopy(joint_position)

        #骨盆姿態(要確認！)
        self.RP = np.array([[1,0,0],[0,1,0],[0,0,1]])
        # 各關節角度
        Theta1 = jp[0,0] #L_Hip_Roll
        Theta2 = jp[1,0]
        Theta3 = jp[2,0]
        Theta4 = jp[3,0]
        Theta5 = jp[4,0]
        Theta6 = jp[5,0] #L_Ankle_Roll

        Theta7 = jp[6,0] #R_Hip_Roll
        Theta8 = jp[7,0]
        Theta9 = jp[8,0]
        Theta10 = jp[9,0]
        Theta11 = jp[10,0]
        Theta12 = jp[11,0] #R_Ankle_Roll

        #calculate rotation matrix
        self.L_R01 = self.xyz_rotation('x',Theta1) #L_Hip_roll
        self.L_R12 = self.xyz_rotation('z',Theta2)
        self.L_R23 = self.xyz_rotation('y',Theta3)
        self.L_R34 = self.xyz_rotation('y',Theta4)
        self.L_R45 = self.xyz_rotation('y',Theta5)
        self.L_R56 = self.xyz_rotation('x',Theta6) #L_Ankle_roll

        self.R_R01 = self.xyz_rotation('x',Theta7) #R_Hip_roll
        self.R_R12 = self.xyz_rotation('z',Theta8)
        self.R_R23 = self.xyz_rotation('y',Theta9)
        self.R_R34 = self.xyz_rotation('y',Theta10)
        self.R_R45 = self.xyz_rotation('y',Theta11)
        self.R_R56 = self.xyz_rotation('x',Theta12) #R_Ankle_roll

    def relative_axis(self):
        self.AL1 = self.RP@(np.array([[1],[0],[0]])) #L_Hip_roll
        self.AL2 = self.RP@self.L_R01@(np.array([[0],[0],[1]])) 
        self.AL3 = self.RP@self.L_R01@self.L_R12@(np.array([[0],[1],[0]])) 
        self.AL4 = self.RP@self.L_R01@self.L_R12@self.L_R23@(np.array([[0],[1],[0]]))
        self.AL5 = self.RP@self.L_R01@self.L_R12@self.L_R23@self.L_R34@(np.array([[0],[1],[0]])) 
        self.AL6 = self.RP@self.L_R01@self.L_R12@self.L_R23@self.L_R34@self.L_R45@(np.array([[1],[0],[0]])) #L_Ankle_Roll
        # print("AL1: ",self.AL1)
        # print("AL2: ",self.AL2)
        # print("AL3: ",self.AL3)
        # print("AL4: ",self.AL4)
        # print("AL5: ",self.AL5)
        # print("AL6: ",self.AL6)  

        self.AR1 = self.RP@(np.array([[1],[0],[0]])) #R_Hip_roll
        self.AR2 = self.RP@self.R_R01@(np.array([[0],[0],[1]])) 
        self.AR3 = self.RP@self.R_R01@self.R_R12@(np.array([[0],[1],[0]])) 
        self.AR4 = self.RP@self.R_R01@self.R_R12@self.R_R23@(np.array([[0],[1],[0]]))
        self.AR5 = self.RP@self.R_R01@self.R_R12@self.R_R23@self.R_R34@(np.array([[0],[1],[0]])) 
        self.AR6 = self.RP@self.R_R01@self.R_R12@self.R_R23@self.R_R34@self.R_R45@(np.array([[1],[0],[0]])) #R_Ankle_Roll
        # print("AR1: ",self.AR1)
        # print("AR2: ",self.AR2)
        # print("AR3: ",self.AR3)
        # print("AR4: ",self.AR4)
        # print("AR5: ",self.AR5)
        # print("AR6: ",self.AR6) 

    def get_position(self,configuration):
        self.pelvis = configuration.get_transform_frame_to_world("pelvis_link")
        # print("p",pelvis)
        self.l_hip_roll = configuration.get_transform_frame_to_world("l_hip_yaw_1")
        self.l_hip_yaw = configuration.get_transform_frame_to_world("l_hip_pitch_1")
        self.l_hip_pitch = configuration.get_transform_frame_to_world("l_thigh_1")
        self.l_knee_pitch = configuration.get_transform_frame_to_world("l_shank_1")
        self.l_ankle_pitch = configuration.get_transform_frame_to_world("l_ankle_1")
        self.l_ankle_roll = configuration.get_transform_frame_to_world("l_foot_1")
        self.l_foot = configuration.get_transform_frame_to_world("l_foot")
        # print("l_foot:",l_foot.translation)
        self.r_hip_roll = configuration.get_transform_frame_to_world("r_hip_yaw_1")
        self.r_hip_yaw = configuration.get_transform_frame_to_world("r_hip_pitch_1")
        self.r_hip_pitch = configuration.get_transform_frame_to_world("r_thigh_1")
        self.r_knee_pitch = configuration.get_transform_frame_to_world("r_shank_1")
        self.r_ankle_pitch = configuration.get_transform_frame_to_world("r_ankle_1")
        self.r_ankle_roll = configuration.get_transform_frame_to_world("r_foot_1")
        self.r_foot = configuration.get_transform_frame_to_world("r_foot")
        # print("r_foot:",r_foot.translation)
          
    def get_posture(self):
        cos = math.cos
        sin = math.sin

        pelvis_p = np.reshape(copy.deepcopy(self.pelvis.translation),(3,1))
        pelvis_p[0,0] = pelvis_p[0,0] - 0.007
        l_foot_p = np.reshape(copy.deepcopy(self.l_foot.translation),(3,1))
        r_foot_p = np.reshape(copy.deepcopy(self.r_foot.translation),(3,1))

        pelvis_o = copy.deepcopy(self.pelvis.rotation)
        l_foot_o = copy.deepcopy(self.l_foot.rotation)
        r_foot_o = copy.deepcopy(self.r_foot.rotation)

        pR = R.from_matrix(pelvis_o).as_euler('zyx', degrees=False)   
        P_Yaw = pR[0]
        P_Pitch = pR[1]
        P_Roll = pR[2]

        lR = R.from_matrix(l_foot_o).as_euler('zyx', degrees=False) 
        L_Yaw = lR[0]
        L_Pitch = lR[1]
        L_Roll = lR[2]
        
        rR = R.from_matrix(r_foot_o).as_euler('zyx', degrees=False) 
        R_Yaw = rR[0]
        R_Pitch = rR[1]
        R_Roll = rR[2]

        self.PX = np.array([[pelvis_p[0,0]],[pelvis_p[1,0]],[pelvis_p[2,0]],[P_Roll],[P_Pitch],[P_Yaw]])
        self.LX = np.array([[l_foot_p[0,0]],[l_foot_p[1,0]],[l_foot_p[2,0]],[L_Roll],[L_Pitch],[L_Yaw]])
        self.RX = np.array([[r_foot_p[0,0]],[r_foot_p[1,0]],[r_foot_p[2,0]],[R_Roll],[R_Pitch],[R_Yaw]])

        self.L_Body_transfer = np.array([[cos(L_Pitch)*cos(L_Yaw), -sin(L_Yaw),0],
                                [cos(L_Pitch)*sin(L_Yaw), cos(L_Yaw), 0],
                                [-sin(L_Pitch), 0, 1]])  
        
        self.R_Body_transfer = np.array([[cos(R_Pitch)*cos(R_Yaw), -sin(R_Yaw),0],
                                [cos(R_Pitch)*sin(R_Yaw), cos(R_Yaw), 0],
                                [-sin(R_Pitch), 0, 1]])  
        
        # print("PX",self.PX)
        # print("LX",self.LX)

    def left_leg_jacobian(self):
        pelvis = np.reshape(copy.deepcopy(self.pelvis.translation),(3,1))
        l_hip_roll = np.reshape(copy.deepcopy(self.l_hip_roll.translation),(3,1))
        l_hip_yaw = np.reshape(copy.deepcopy(self.l_hip_yaw.translation),(3,1))
        l_hip_pitch = np.reshape(copy.deepcopy(self.l_hip_pitch.translation),(3,1))
        l_knee_pitch = np.reshape(copy.deepcopy(self.l_knee_pitch.translation),(3,1))
        l_ankle_pitch = np.reshape(copy.deepcopy(self.l_ankle_pitch.translation),(3,1))
        l_ankle_roll = np.reshape(copy.deepcopy(self.l_ankle_roll.translation),(3,1))
        l_foot = np.reshape(copy.deepcopy(self.l_foot.translation),(3,1))
        # print("l_hip_roll:",l_hip_roll)
        # print("l_hip_yaw:",l_hip_yaw)
        # print("l_hip_pitch:",l_hip_pitch)
        # print("l_knee_pitch:",l_knee_pitch)
        # print("l_ankle_pitch:",l_ankle_pitch)
        # print("l_ankle_roll:",l_ankle_roll)
        # print("l_foot:",l_foot)


        # print("2",l_knee_pitch,l_ankle_pitch,l_ankle_roll)
        # l_foot = copy.deepcopy(l_foot)
        JL1 = np.cross(self.AL1,(l_foot-l_hip_roll),axis=0)
        JL2 = np.cross(self.AL2,(l_foot-l_hip_yaw),axis=0)
        JL3 = np.cross(self.AL3,(l_foot-l_hip_pitch),axis=0)
        JL4 = np.cross(self.AL4,(l_foot-l_knee_pitch),axis=0)
        JL5 = np.cross(self.AL5,(l_foot-l_ankle_pitch),axis=0)
        JL6 = np.cross(self.AL6,(l_foot-l_ankle_roll),axis=0)

        JLL_upper = np.hstack((JL1, JL2,JL3,JL4,JL5,JL6))
        JLL_lower = np.hstack((self.AL1,self.AL2,self.AL3,self.AL4,self.AL5,self.AL6))    
        self.JLL = np.vstack((JLL_upper,JLL_lower))  
        # print(self.JLL)

        #排除支撐腳腳踝對末端速度的影響
        self.JLL44 = np.reshape(self.JLL[2:,0:4],(4,4))  
        self.JLL42 = np.reshape(self.JLL[2:,4:],(4,2))

        return self.JLL

    def right_leg_jacobian(self):
        pelvis = np.reshape(copy.deepcopy(self.pelvis.translation),(3,1))
        r_hip_roll = np.reshape(copy.deepcopy(self.r_hip_roll.translation),(3,1))
        r_hip_yaw = np.reshape(copy.deepcopy(self.r_hip_yaw.translation),(3,1))
        r_hip_pitch = np.reshape(copy.deepcopy(self.r_hip_pitch.translation),(3,1))
        r_knee_pitch = np.reshape(copy.deepcopy(self.r_knee_pitch.translation),(3,1))
        r_ankle_pitch = np.reshape(copy.deepcopy(self.r_ankle_pitch.translation),(3,1))
        r_ankle_roll = np.reshape(copy.deepcopy(self.r_ankle_roll.translation),(3,1))
        r_foot = np.reshape(copy.deepcopy(self.r_foot.translation),(3,1))
        # print("1:",l_hip_roll,l_hip_yaw,l_hip_pitch)
        # print("2",l_knee_pitch,l_ankle_pitch,l_ankle_roll)
        # l_foot = copy.deepcopy(l_foot)
        JR1 = np.cross(self.AR1,(r_foot-r_hip_roll),axis=0)
        JR2 = np.cross(self.AR2,(r_foot-r_hip_yaw),axis=0)
        JR3 = np.cross(self.AR3,(r_foot-r_hip_pitch),axis=0)
        JR4 = np.cross(self.AR4,(r_foot-r_knee_pitch),axis=0)
        JR5 = np.cross(self.AR5,(r_foot-r_ankle_pitch),axis=0)
        JR6 = np.cross(self.AR6,(r_foot-r_ankle_roll),axis=0)

        JRR_upper = np.hstack((JR1,JR2,JR3,JR4,JR5,JR6))
        JRR_lower = np.hstack((self.AR1,self.AR2,self.AR3,self.AR4,self.AR5,self.AR6))    
        self.JRR = np.vstack((JRR_upper,JRR_lower))  
        # print(self.JRR)

        #排除支撐腳腳踝對末端速度的影響
        self.JRR44 = np.reshape(self.JRR[2:,0:4],(4,4))  
        self.JRR42 = np.reshape(self.JRR[2:,4:],(4,2))
        return self.JRR

    def ref_cmd(self):
        #pelvis
        # #放到右腳上
        # P_Y_ref = -0.1
        #放到左腳上
        P_Y_ref = 0.1

        P_X_ref = 0.0
        P_Z_ref = 0.58
        P_Roll_ref = 0.0
        P_Pitch_ref = 0.0
        P_Yaw_ref = 0.0

        self.PX_ref = np.array([[P_X_ref],[P_Y_ref],[P_Z_ref],[P_Roll_ref],[P_Pitch_ref],[P_Yaw_ref]])

        #left_foot
        # #右腳測試時
        # L_X_ref = 0.007
        # L_Y_ref = 0.03
        # L_Z_ref = 0.05
        #左腳測試時
        L_X_ref = 0.007
        L_Y_ref = 0.1
        L_Z_ref = 0.02

        L_Roll_ref = 0.0
        L_Pitch_ref = 0.0
        L_Yaw_ref = 0.0
        
        self.LX_ref = np.array([[L_X_ref],[L_Y_ref],[L_Z_ref],[L_Roll_ref],[L_Pitch_ref],[L_Yaw_ref]])

        #right_foot
        # # 右腳測試時
        # R_X_ref = 0.007
        # R_Y_ref = -0.1
        # R_Z_ref = 0.02
        #左腳測試時
        R_X_ref = 0.007
        R_Y_ref = -0.03
        R_Z_ref = 0.05

        R_Roll_ref = 0.0
        R_Pitch_ref = 0.0
        R_Yaw_ref = 0.0

        self.RX_ref = np.array([[R_X_ref],[R_Y_ref],[R_Z_ref],[R_Roll_ref],[R_Pitch_ref],[R_Yaw_ref]])

    def calculate_err(self):
        PX_ref = copy.deepcopy(self.PX_ref)
        LX_ref = copy.deepcopy(self.LX_ref)
        RX_ref = copy.deepcopy(self.RX_ref)
        PX = copy.deepcopy(self.PX)
        LX = copy.deepcopy(self.LX)
        RX = copy.deepcopy(self.RX)

        # print(self.L_ref_data[:,0])
        
        if self.state==2 and self.count < 1800:
            #foot_trajectory(by norman)
            L_ref = (np.reshape(self.L_ref_data[:,self.count],(6,1)))
            R_ref = (np.reshape(self.R_ref_data[:,self.count],(6,1)))
        else:
            #foot_trajectory(by myself)
            L_ref = LX_ref - PX_ref 
            R_ref = RX_ref - PX_ref

        L = LX - PX
        R = RX - PX 

        Le_dot = 20*(L_ref - L)
        Re_dot = 20*(R_ref - R)

        Lroll_error_dot = Le_dot[3,0]
        Lpitch_error_dot = Le_dot[4,0]
        Lyaw_error_dot = Le_dot[5,0]
        WL_x = self.L_Body_transfer[0,0]*Lroll_error_dot + self.L_Body_transfer[0,1]*Lpitch_error_dot
        WL_y = self.L_Body_transfer[1,0]*Lroll_error_dot + self.L_Body_transfer[1,1]*Lpitch_error_dot
        WL_z = self.L_Body_transfer[2,0]*Lroll_error_dot + self.L_Body_transfer[2,2]*Lyaw_error_dot

        Le_2 = np.array([[Le_dot[0,0]],[Le_dot[1,0]],[Le_dot[2,0]],[WL_x],[WL_y],[WL_z]])

        Rroll_error_dot = Re_dot[3,0]
        Rpitch_error_dot = Re_dot[4,0]
        Ryaw_error_dot = Re_dot[5,0]
        WR_x = self.R_Body_transfer[0,0]*Rroll_error_dot + self.R_Body_transfer[0,1]*Rpitch_error_dot
        WR_y = self.R_Body_transfer[1,0]*Rroll_error_dot + self.R_Body_transfer[1,1]*Rpitch_error_dot
        WR_z = self.R_Body_transfer[2,0]*Rroll_error_dot + self.R_Body_transfer[2,2]*Ryaw_error_dot

        Re_2 = np.array([[Re_dot[0,0]],[Re_dot[1,0]],[Re_dot[2,0]],[WR_x],[WR_y],[WR_z]])

        #切換重力補償模型
        if abs(L[1,0]) <0.05:
            self.stance = 1
        elif abs(R[1,0]) <0.05:
            self.stance = 0
        else:
            self.stance = 2
        print("stance:",self.stance)
        return Le_2,Re_2,self.stance
    
    def velocity_cmd(self,Le_2,Re_2,jv_f):

        L2 = copy.deepcopy(Le_2)
        R2 = copy.deepcopy(Re_2)
        v =  copy.deepcopy(jv_f) #joint_velocity

        # print(L2)
        if self.state == 3:   #(左支撐腳腳踝動態排除測試)
            L2_41 = np.reshape(L2[2:,0],(4,1)) #L2 z to wz
            VL56 =  np.reshape(v[4:6,0],(2,1)) #左腳腳踝速度
            
            L2_41_cal = L2_41 - self.JLL42@VL56
            
            lw_41_d = np.dot(np.linalg.pinv(self.JLL44),L2_41_cal)
            lw_21_d = np.zeros((2,1))

            Lw_d = np.vstack((lw_41_d,lw_21_d))
            Rw_d = np.dot(np.linalg.pinv(self.JRR),R2) 
        elif self.state == 4:   #(右支撐腳腳踝動態排除測試)
            R2_41 = np.reshape(R2[2:,0],(4,1)) #R2 z to wz
            VR56 =  np.reshape(v[10:,0],(2,1)) #右腳腳踝速度
            
            R2_41_cal = R2_41 - self.JRR42@VR56
            
            rw_41_d = np.dot(np.linalg.pinv(self.JRR44),R2_41_cal)
            rw_21_d = np.zeros((2,1))

            Lw_d = np.dot(np.linalg.pinv(self.JLL),L2) 
            Rw_d = np.vstack((rw_41_d,rw_21_d))
        else:
            Lw_d = np.dot(np.linalg.pinv(self.JLL),L2) 
            Rw_d = np.dot(np.linalg.pinv(self.JRR),R2) 
        

        return Lw_d,Rw_d
    
    def gravity_compemsate(self,joint_position):
        jp_l = np.reshape(copy.deepcopy(joint_position[0:6,0]),(6,1)) #左腳
        jp_r = np.reshape(copy.deepcopy(joint_position[6:,0]),(6,1))  #右腳

        kl = 0.3
        kr = 0.3

        #雙支撐
        if self.stance == 2:
            kl = 1
            kr = 1
            jp_l = np.flip(-jp_l,axis=0)
            jv_l = np.zeros((6,1))
            c_l = np.zeros((6,1))
            l_leg_gravity = np.reshape(-pin.rnea(self.stance_l_model, self.stance_l_data, jp_l,jv_l,(c_l)),(6,1))  
            l_leg_gravity = np.flip(l_leg_gravity,axis=0)

            jp_r = np.flip(-jp_r,axis=0)
            jv_r = np.zeros((6,1))
            c_r = np.zeros((6,1))
            r_leg_gravity = np.reshape(-pin.rnea(self.stance_r_model, self.stance_r_data, jp_r,jv_r,(c_r)),(6,1))  
            r_leg_gravity = np.flip(r_leg_gravity,axis=0)
        
        #右腳為支撐腳(右腳關節翻轉加負號)
        elif self.stance == 0: 
            kr = 1.5
            jp_r = np.flip(-jp_r,axis=0)
            jp = np.vstack((jp_r,jp_l))
            jv = np.zeros((12,1))
            cin = np.zeros((12,1))
            leg_gravity = np.reshape(pin.rnea(self.bipedal_r_model, self.bipedal_r_data, jp,jv,(cin)),(12,1))  
    
            l_leg_gravity = np.reshape(leg_gravity[6:,0],(6,1))
            r_leg_gravity = np.reshape(-leg_gravity[0:6,0],(6,1)) #加負號(相對關係)
            r_leg_gravity = np.flip(r_leg_gravity,axis=0)

        #左腳為支撐腳(左腳關節翻轉加負號)
        elif self.stance == 1:
            kl = 2
            jp_l = np.flip(-jp_l,axis=0)
            jp = np.vstack((jp_l,jp_r))
            jv = np.zeros((12,1))
            cin = np.zeros((12,1))
            leg_gravity = np.reshape(pin.rnea(self.bipedal_l_model, self.bipedal_l_data, jp,jv,(cin)),(12,1))  
    
            l_leg_gravity = np.reshape(-leg_gravity[0:6,0],(6,1)) #加負號(相對關係)
            l_leg_gravity = np.flip(l_leg_gravity,axis=0)
            r_leg_gravity = np.reshape(leg_gravity[6:,0],(6,1))

        else:
            l_leg_gravity = np.zeros((6,1))
            r_leg_gravity = np.zeros((6,1))
        
        self.l_gravity_publisher.publish(Float64MultiArray(data=l_leg_gravity))
        self.r_gravity_publisher.publish(Float64MultiArray(data=r_leg_gravity))
        
        return l_leg_gravity,r_leg_gravity,kl,kr
    
    def balance(self,joint_position,l_leg_gravity_compensate,r_leg_gravity_compensate):
        #balance the robot to initial state by p_control
        jp = copy.deepcopy(joint_position)
        p = np.array([[0.0],[0.0],[-0.37],[0.74],[-0.37],[0.0],[0.0],[0.0],[-0.37],[0.74],[-0.37],[0.0]])
        l_leg_gravity = copy.deepcopy(l_leg_gravity_compensate)
        r_leg_gravity = copy.deepcopy(r_leg_gravity_compensate)

        torque = np.zeros((12,1))
        torque[0,0] = 20*(p[0,0]-jp[0,0]) + l_leg_gravity[0,0]
        torque[1,0] = 20*(p[1,0]-jp[1,0]) + l_leg_gravity[1,0]
        torque[2,0] = 40*(p[2,0]-jp[2,0]) + l_leg_gravity[2,0]
        torque[3,0] = 60*(p[3,0]-jp[3,0]) + l_leg_gravity[3,0]
        torque[4,0] = 60*(p[4,0]-jp[4,0]) + l_leg_gravity[4,0]
        torque[5,0] = 40*(p[5,0]-jp[5,0]) 

        torque[6,0] = 25*(p[6,0]-jp[6,0]) + r_leg_gravity[0,0]
        torque[7,0] = 20*(p[7,0]-jp[7,0]) + r_leg_gravity[1,0]
        torque[8,0] = 40*(p[8,0]-jp[8,0]) + r_leg_gravity[2,0]
        torque[9,0] = 60*(p[9,0]-jp[9,0]) + r_leg_gravity[3,0]
        torque[10,0] = 60*(p[10,0]-jp[10,0]) + r_leg_gravity[4,0]
        torque[11,0] = 50*(p[11,0]-jp[11,0]) 
        self.effort_publisher.publish(Float64MultiArray(data=torque))

    def swing_leg(self,joint_position,joint_velocity,l_leg_vcmd,r_leg_vcmd,l_leg_gravity_compensate,r_leg_gravity_compensate,kl,kr):
        print("swing_mode")
        self.tt += 0.0314
        jp = copy.deepcopy(joint_position)
        jv = copy.deepcopy(joint_velocity)
        vl_cmd = copy.deepcopy(l_leg_vcmd)
        vr_cmd = copy.deepcopy(r_leg_vcmd)
        l_leg_gravity = copy.deepcopy(l_leg_gravity_compensate)
        r_leg_gravity = copy.deepcopy(r_leg_gravity_compensate)

        # #L_leg_velocity
        # vl = np.reshape(copy.deepcopy(joint_velocity[:6,0]),(6,1))

        p = np.array([[0.0],[0.0],[-0.37],[0.74],[-0.37],[0.0],[0.0],[0.0],[-0.37],[0.74],[-0.37],[0.0]])

        torque = np.zeros((12,1))

        torque[0,0] = kl*(vl_cmd[0,0]-jv[0,0]) + l_leg_gravity[0,0]
        torque[1,0] = kl*(vl_cmd[1,0]-jv[1,0]) + l_leg_gravity[1,0]
        torque[2,0] = kl*(vl_cmd[2,0]-jv[2,0]) + l_leg_gravity[2,0]
        torque[3,0] = kl*(vl_cmd[3,0]-jv[3,0]) + l_leg_gravity[3,0]
        torque[4,0] = kl*(vl_cmd[4,0]-jv[4,0]) + l_leg_gravity[4,0]
        torque[5,0] = kl*(vl_cmd[5,0]-jv[5,0]) + l_leg_gravity[5,0]

        torque[6,0] = kr*(vr_cmd[0,0]-jv[6,0]) + r_leg_gravity[0,0]
        torque[7,0] = kr*(vr_cmd[1,0]-jv[7,0]) + r_leg_gravity[1,0]
        torque[8,0] = kr*(vr_cmd[2,0]-jv[8,0]) + r_leg_gravity[2,0]
        torque[9,0] = kr*(vr_cmd[3,0]-jv[9,0]) + r_leg_gravity[3,0]
        torque[10,0] = kr*(vr_cmd[4,0]-jv[10,0]) + r_leg_gravity[4,0]
        torque[11,0] = kr*(vr_cmd[5,0]-jv[11,0]) + r_leg_gravity[5,0]
        self.effort_publisher.publish(Float64MultiArray(data=torque))

        vcmd_data = np.array([[vl_cmd[0,0]],[vl_cmd[1,0]],[vl_cmd[2,0]],[vl_cmd[3,0]],[vl_cmd[4,0]],[vl_cmd[5,0]]])
        self.vcmd_publisher.publish(Float64MultiArray(data=vcmd_data))
        jv_collect = np.array([[jv[0,0]],[jv[1,0]],[jv[2,0]],[jv[3,0]],[jv[4,0]],[jv[5,0]]])
        self.velocity_publisher.publish(Float64MultiArray(data=jv_collect))#檢查收到的速度(超髒)

        return torque

    def walking(self,joint_position,joint_velocity,l_leg_vcmd,r_leg_vcmd,l_leg_gravity_compensate,r_leg_gravity_compensate,kl,kr):
        print("walking")
        self.count += 1 
        jv = copy.deepcopy(joint_velocity)
        vl_cmd = copy.deepcopy(l_leg_vcmd)
        vr_cmd = copy.deepcopy(r_leg_vcmd)
        l_leg_gravity = copy.deepcopy(l_leg_gravity_compensate)
        r_leg_gravity = copy.deepcopy(r_leg_gravity_compensate)

        torque = np.zeros((12,1))

        torque[0,0] = 1.1*(vl_cmd[0,0]-jv[0,0]) + l_leg_gravity[0,0]
        torque[1,0] = (vl_cmd[1,0]-jv[1,0])
        torque[2,0] = (vl_cmd[2,0]-jv[2,0]) + l_leg_gravity[2,0]
        torque[3,0] = kl*(vl_cmd[3,0]-jv[3,0]) + l_leg_gravity[3,0]
        torque[4,0] = kl*(vl_cmd[4,0]-jv[4,0]) + l_leg_gravity[4,0]
        torque[5,0] = kl*(vl_cmd[5,0]-jv[5,0]) + l_leg_gravity[5,0]

        torque[6,0] = 1.1*(vr_cmd[0,0]-jv[6,0])+ r_leg_gravity[0,0]
        torque[7,0] = (vr_cmd[1,0]-jv[7,0])
        torque[8,0] = (vr_cmd[2,0]-jv[8,0]) + r_leg_gravity[2,0]
        torque[9,0] = kr*(vr_cmd[3,0]-jv[9,0]) + r_leg_gravity[3,0]
        torque[10,0] = kr*(vr_cmd[4,0]-jv[10,0])
        torque[11,0] = kr*(vr_cmd[5,0]-jv[11,0]) + r_leg_gravity[5,0]

        self.effort_publisher.publish(Float64MultiArray(data=torque))

    def com_position(self,joint_position,stance_type):
        #get com position
        jp_l = np.reshape(copy.deepcopy(joint_position[0:6,0]),(6,1)) #左腳
        jp_r = np.reshape(copy.deepcopy(joint_position[6:,0]),(6,1))  #右腳
        
        #得到支撐狀態
        stance = copy.deepcopy(stance_type)

        #右腳為支撐腳
        if stance == 0:
            jp_r = np.flip(-jp_r,axis=0)
            joint_angle = np.vstack((jp_r,jp_l))
            pin.centerOfMass(self.bipedal_r_model,self.bipedal_r_data,joint_angle)
            com_in_wf = np.reshape(self.bipedal_r_data.com[0],(3,1))
            r_foot_in_wf = np.array([[0.007],[-0.1],[0]])
            com_in_rf = com_in_wf - r_foot_in_wf
            com_in_lf = np.zeros((3,1))

        #左腳為支撐腳
        elif stance == 1:
            jp_l = np.flip(-jp_l,axis=0)
            joint_angle = np.vstack((jp_l,jp_r))
            pin.centerOfMass(self.bipedal_l_model,self.bipedal_l_data,joint_angle)
            com_in_wf = np.reshape(self.bipedal_l_data.com[0],(3,1))
            l_foot_in_wf = np.array([[0.007],[0.1],[0]])
            com_in_lf = com_in_wf - l_foot_in_wf
            com_in_rf = np.zeros((3,1))
        
        return com_in_lf,com_in_rf 

    def alip_test(self,joint_velocity,l_leg_vcmd,r_leg_vcmd,l_leg_gravity_compensate,r_leg_gravity_compensate,kl,kr,stance_type,joint_torque,com_in_lf,com_in_rf):
        print("alip_mode")
        jv = copy.deepcopy(joint_velocity)
        vl_cmd = copy.deepcopy(l_leg_vcmd)
        vr_cmd = copy.deepcopy(r_leg_vcmd)
        l_leg_gravity = copy.deepcopy(l_leg_gravity_compensate)
        r_leg_gravity = copy.deepcopy(r_leg_gravity_compensate)
        #支撐狀態
        stance = copy.deepcopy(stance_type) 
        #獲取關節扭矩資訊
        torque = copy.deepcopy(joint_torque) 
        #獲取量測值
        CX_l = copy.deepcopy(com_in_lf)
        CX_r = copy.deepcopy(com_in_rf)
        
        #左腳
        #左腳ALIP模型測試
        if self.stance == 1:
            #質心速度
            #量測值
            Xc_mea = CX_l[0,0]
            Ly_mea = 0
            self.model_state_x = np.array([[Xc_mea],[Ly_mea]])
            #模型(m=9 H=0.4 Ts=0.005)
            Ax = np.array([[1,0.001389],[0.4415,1]])
            Bx = np.array([[0],[0.005]])
            Cx = np.array([[1,0],[0,1]])  
            #--LQR
            Kx = np.array([[298.1259,15.9944]])
            Lx = np.array([[0.0808,0.0014],[0.4415,0.1505]]) 
            #--compensator
            self.ob_xly = Ax@self.ob_xly_past + self.ap_past*Bx + Lx@(self.model_state_x_past - Cx@self.ob_xly_past)
            #----calculate toruqe
            self.ap = torque[4,0]
            #----update
            self.model_state_x_past = self.model_state_x
            self.ob_xly_past = self.ob_xly
            self.ap_past = self.ap
        
        print("mea_data:",self.model_state_x)
        print("obs_data:",self.ob_xly)

    def main_controller_callback(self):

        joint_position,joint_velocity = self.collect_joint_data()
        joint_velocity_cal = self.joint_velocity_cal(joint_position)
        jv_f = self.joint_velocity_filter(joint_velocity_cal)

        self.position_publisher.publish(Float64MultiArray(data=joint_position))#檢查收到的位置(普)
        # jv_collect = np.array([[joint_velocity[5,0]],[joint_velocity_cal[5,0]],[jv_f[5,0]]])
        # self.velocity_publisher.publish(Float64MultiArray(data=jv_collect))#檢查收到的速度(超髒)
        
        self.rotation_matrix(joint_position)

        self.relative_axis()

        configuration = pink.Configuration(self.robot.model, self.robot.data,joint_position)
        self.get_position(configuration)
        self.get_posture()
        self.viz.display(configuration.q)

        l_leg_gravity,r_leg_gravity,kl,kr = self.gravity_compemsate(joint_position)

        JLL = self.left_leg_jacobian()
        JRR = self.right_leg_jacobian()

        self.ref_cmd()

        Le_2,Re_2,stance = self.calculate_err()

        VL,VR = self.velocity_cmd(Le_2,Re_2,jv_f)
        
        if self.state == 0:   
            self.balance(joint_position,l_leg_gravity,r_leg_gravity)

        elif self.state == 1:
            torque = self.swing_leg(joint_position,jv_f,VL,VR,l_leg_gravity,r_leg_gravity,kl,kr)

            if stance == 0 or stance == 1 :
                com_in_lf,com_in_rf = self.com_position(joint_position,stance)
                self.alip_test(jv_f,VL,VR,l_leg_gravity,r_leg_gravity,kl,kr,stance,torque, com_in_lf,com_in_rf)

        elif self.state == 2:
            self.walking(joint_position,jv_f,VL,VR,l_leg_gravity,r_leg_gravity,kl,kr)
            
        # v = np.vstack((VL,VR))





        # # for trajectory controller
        # p = joint_position + self.timer_period*v
        # v = np.reshape(v,(12))
        # p = np.reshape(p,(12))
        # p = np.array([0.0,0.0,-0.37,0.74,-0.36,0.0,0.0,0.0,-0.37,0.74,-0.36,0.0])
        # trajectory_msg  = JointTrajectory()
        # # trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        # trajectory_msg.header.frame_id= 'base_link'
        # trajectory_msg.joint_names = [
        #     'L_Hip_Roll', 'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 
        #     'L_Ankle_Pitch', 'L_Ankle_Roll', 'R_Hip_Roll', 'R_Hip_Yaw', 
        #     'R_Hip_Pitch', 'R_Knee_Pitch', 'R_Ankle_Pitch', 'R_Ankle_Roll'
        # ]
        # point = JointTrajectoryPoint()
        # point.positions = list(p)
        # point.velocities = list(v)
        # point.time_from_start = rclpy.duration.Duration(seconds=self.timer_period).to_msg()
        # trajectory_msg.points.append(point)
        # self.joint_trajectory_controller.publish(trajectory_msg)



def main(args=None):
    rclpy.init(args=args)

    upper_level_controllers = UpperLevelController()

    rclpy.spin(upper_level_controllers)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    upper_level_controllers.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

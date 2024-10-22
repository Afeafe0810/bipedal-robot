import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray 

from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState

from nav_msgs.msg import Odometry

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
from copy import deepcopy
from math import cos, sin, cosh, sinh, pi, sqrt
from scipy.spatial.transform import Rotation

import pandas as pd
import csv

from linkattacher_msgs.srv import AttachLink
from linkattacher_msgs.srv import DetachLink

#========================常數==================================#
original_joint_order = ['L_Hip_az', 'L_Hip_ay', 'L_Knee_ay', 'L_Ankle_ay', 'L_Ankle_ax', 'R_Hip_ax',
                        'R_Hip_az', 'R_Knee_ay', 'R_Hip_ay', 'R_Ankle_ay', 'L_Hip_ax', 'R_Ankle_ax']
desired_joint_order = ['L_Hip_ax', 'L_Hip_az', 'L_Hip_ay', 'L_Knee_ay', 'L_Ankle_ay', 'L_Ankle_ax',
                       'R_Hip_ax', 'R_Hip_az', 'R_Hip_ay', 'R_Knee_ay', 'R_Ankle_ay', 'R_Ankle_ax']

timer_period = 0.01 # seconds 跟joint state update rate&取樣次數有關
DS_timeLength = 2 # 雙支撐的時間總長

#理想機器人狀態
m = 9    #機器人下肢總重
H = 0.45 #理想質心高度
W = 0.2 #兩腳底間距
g = 9.81 #重力
l = sqrt(g/H)
T = 0.5 #支撐間隔時長

def zyx_analy_2_gmtry_w(ay:float, az:float) -> np.ndarray:
    '''我們的歐拉角是以zyx做定義, 這個函式把解析角速度轉成幾何角速度'''
    return np.array([[cos(ay)*cos(az), -sin(az), 0],
                     [cos(ay)*sin(az),  cos(az), 0],
                     [-sin(ay),         0,       1]])

class Allcoodinate():
    '''儲存所有座標軸的位置、速度、旋轉矩陣相對其他frame'''
    def __init__(self) -> None:
        #=========base_in_wf_callback=========#
        self.p_base_in_wf = None
        self.r_base2wf = None
        
        self.__p_com_in_lfrf_past = np.zeros((6,1)) #只是用來做濾波而已
        self.__v_com_in_lfrf_past = np.zeros((6,1))
        self.__vf_com_in_lfrf_past = np.zeros((6,1))
    
    def get_all_frame_needed(self, config:pink.Configuration, bipedal_floating_model, bipedal_floating_data, jp:np.ndarray) -> None:
        #=========================== in pf==================================#
        self.get_in_pf(config, bipedal_floating_model, bipedal_floating_data, jp)
        self.get_in_wf() #利用座標轉換求得_in_wf
        self.jointframe_rotation(jp)
        self.get_unitVector_of_jv()
            
    def get_in_pf(self, config:pink.Configuration, bipedal_floating_model, bipedal_floating_data, jp:np.ndarray):
        
        def config_to_linkplace():
            '''利用self.robot的config求出各點的齊次矩陣'''
            for linkname in linknames:
                homo_pt_in_pf = config.get_transform_frame_to_world(linkname) #骨盆對pink_wf齊次矩陣
                
                p_pt_in_pf = homo_pt_in_pf.translation
                r_pt2pf = homo_pt_in_pf.rotation
                
                a_pt_in_pf = Rotation.from_matrix(r_pt2pf).as_euler('zyx', degrees=False) [::-1] #把旋轉矩陣換成歐拉角zyx,並轉成ax,ay,az
                
                pa_pt_in_pf=np.vstack(( p_pt_in_pf, a_pt_in_pf ))
                
                yield pa_pt_in_pf, r_pt2pf
                
        linknames = [
            "pelvis_link", "l_foot", "r_foot", 
            "l_hip_yaw_1", "l_hip_pitch_1", "l_thigh_1", "l_shank_1", "l_ankle_1", "l_foot_1",
            "r_hip_yaw_1", "r_hip_pitch_1", "r_thigh_1", "r_shank_1", "r_ankle_1", "r_foot_1"
            ]
        #===========================用pinocchio求質心==================================# 
        pin.centerOfMass(bipedal_floating_model, bipedal_floating_data, jp) # 會變更引數的值, 不會return
        
        self.p_com_in_pf = np.reshape(bipedal_floating_data.com[0],(3,1)) #todo 可以確認看看不同模型建立出來的質心會不會不一樣
        
        #=============================================================#
        (
            (self.pa_pel_in_pf, self.r_pel2pf), #骨盆
            (self.pa_lf_in_pf , self.r_lf2pf ), #左腳掌
            (self.pa_rf_in_pf , self.r_rf2pf ), #右腳掌
            
            (self.pa_lj1_in_pf, self.r_lj12pf), #由骨盆往下第一個左邊關節
            (self.pa_lj2_in_pf, self.r_lj22pf),
            (self.pa_lj3_in_pf, self.r_lj32pf),
            (self.pa_lj4_in_pf, self.r_lj42pf),
            (self.pa_lj5_in_pf, self.r_lj52pf),
            (self.pa_lj6_in_pf, self.r_lj62pf),
            
            (self.pa_rj1_in_pf, self.r_rj12pf), #由骨盆往下第一個右邊關節
            (self.pa_rj2_in_pf, self.r_rj22pf),
            (self.pa_rj3_in_pf, self.r_rj32pf),
            (self.pa_rj4_in_pf, self.r_rj42pf),
            (self.pa_rj5_in_pf, self.r_rj52pf),
            (self.pa_rj6_in_pf, self.r_rj62pf) 
            )= list(config_to_linkplace())
                     
    def get_in_wf(self):
        
        def inpf_to_inwf():
            r_pf2pel = self.r_pel2pf.T
            for pt in pts:
                p_pel_in_pf = self.pa_pel_in_pf[0:3,0]
                
                p_pt_in_pf = self.__dict__[f'pa_{pt}_in_pf'][0:3]
                r_pt2pf = self.__dict__[f'r_{pt}2pf']
                
                p_pt_in_wf = self.r_pel2wf @ r_pf2pel @(p_pt_in_pf - p_pel_in_pf) + self.p_pel_in_wf
                r_pt2wf = self.r_pel2wf @ r_pf2pel @ r_pt2pf
                
                if pt in ('lf','rf'):
                    #==========利用旋轉矩陣算出角度'''========#
                    a_pt_in_wf = Rotation.from_matrix(r_pt2wf).as_euler('zyx', degrees=False) [::-1] #把旋轉矩陣換成歐拉角zyx,並轉成ax,ay,az
                    yield p_pt_in_wf, r_pt2wf, a_pt_in_wf
                else:
                    yield p_pt_in_wf, r_pt2wf
                
        #==========利用base_in_wf_callback訂閱到的base_in_wf的位態求出pel_in_wf的位態========#
        self.p_pel_in_base = np.vstack(( 0.0, 0.0, 0.598 ))
        
        self.p_pel_in_wf = self.r_base2wf @ self.p_pel_in_base + self.p_base_in_wf
        self.r_pel2wf = deepcopy(self.r_base2wf) #因為兩者是平移建立的
        
        #==========利用訂閱推導出的的pel_in_wf,算出pf下的向量,再轉到pel,再轉到wf'''========#
        pts = [
            'com', 'lf', 'rf',
            'lj1', 'lj2', 'lj3', 'lj4', 'lj5', 'lj6',
            'rj1', 'rj2', 'rj3', 'rj4', 'rj5', 'rj6'
            ]
        
        (
            (self.p_com_in_wf, self.r_com2wf), #骨盆
            (self.p_lf_in_wf , self.r_lf2wf, self.a_lf_in_wf ), #左腳掌
            (self.p_rf_in_wf , self.r_rf2wf, self.a_rf_in_wf ), #右腳掌
            
            (self.p_lj1_in_wf, self.r_lj12wf), #由骨盆往下第一個左邊關節
            (self.p_lj2_in_wf, self.r_lj22wf),
            (self.p_lj3_in_wf, self.r_lj32wf),
            (self.p_lj4_in_wf, self.r_lj42wf),
            (self.p_lj5_in_wf, self.r_lj52wf),
            (self.p_lj6_in_wf, self.r_lj62wf),
            
            (self.p_rj1_in_wf, self.r_rj12wf), #由骨盆往下第一個右邊關節
            (self.p_rj2_in_wf, self.r_rj22wf),
            (self.p_rj3_in_wf, self.r_rj32wf),
            (self.p_rj4_in_wf, self.r_rj42wf),
            (self.p_rj5_in_wf, self.r_rj52wf),
            (self.p_rj6_in_wf, self.r_rj62wf) 
            ) = list(inpf_to_inwf())
    
    def jointframe_rotation(self,jp):
        '''得到兩兩相鄰的關節frame彼此的關係'''
        def xyz_rotation(axis:str, theta:float) ->np.ndarray:
            '''關節繞自己軸旋轉->和前一個frame的旋轉矩陣'''
            if axis == 'x':
                return np.array([[1, 0,           0],
                                [0, cos(theta), -sin(theta)],
                                [0, sin(theta),  cos(theta)]])
            if axis == 'y':
                return np.array([[cos(theta), 0, sin(theta)],
                                [0,          1, 0],
                                [-sin(theta),0, cos(theta)]])
            if axis == 'z':
                return np.array([[cos(theta), -sin(theta),0],
                                [sin(theta), cos(theta) ,0],
                                [0,          0,          1]])
                
        def jointframe_rotation_generator():
            '''生成器, 嗯,我也不知道要怎麼命名'''
            for i,joint_name in enumerate( desired_joint_order ):
                axis = joint_name[-1]
                yield xyz_rotation(axis,jp[i,0])
                
        (
            self.r_lj12pel, self.r_lj22lj1, self.r_lj32lj2, 
            self.r_lj42lj3, self.r_lj52lj4, self.r_lj62lj5,
            
            self.r_rj12pel, self.r_rj22rj1, self.r_rj32rj2, 
            self.r_rj42rj3, self.r_rj52rj4, self.r_rj62rj5
            
            ) = list(jointframe_rotation_generator())
        
    def get_unitVector_of_jv(self):
        '''求每個關節的旋轉軸在wf的方向->為關節角動量的方向'''
        r_pf2wf = self.r_pel2wf @ self.r_pel2pf.T
        
        self.u_axis_lj1_in_wf = r_pf2wf @ np.vstack(( 1, 0, 0 )) #最後面的向量是指旋轉軸向量in自己的frame
        self.u_axis_lj2_in_wf = r_pf2wf @ self.r_lj12pf @ np.vstack(( 0, 0, 1 ))
        self.u_axis_lj3_in_wf = r_pf2wf @ self.r_lj12pf @ self.r_lj22lj1 @ np.vstack(( 0, 1, 0 ))
        self.u_axis_lj4_in_wf = r_pf2wf @ self.r_lj12pf @ self.r_lj22lj1 @ self.r_lj32lj2 @ np.vstack(( 0, 1, 0 ))
        self.u_axis_lj5_in_wf = r_pf2wf @ self.r_lj12pf @ self.r_lj22lj1 @ self.r_lj32lj2 @ self.r_lj42lj3 @ np.vstack(( 0, 1, 0 ))
        self.u_axis_lj6_in_wf = r_pf2wf @ self.r_lj12pf @ self.r_lj22lj1 @ self.r_lj32lj2 @ self.r_lj42lj3 @ self.r_lj52lj4 @ np.vstack(( 1, 0, 0 ))
        
        self.u_axis_rj1_in_wf = r_pf2wf @ np.vstack(( 1, 0, 0 ))
        self.u_axis_rj2_in_wf = r_pf2wf @ self.r_rj12pf @ np.vstack(( 0, 0, 1 ))
        self.u_axis_rj3_in_wf = r_pf2wf @ self.r_rj12pf @ self.r_rj22rj1 @ np.vstack(( 0, 1, 0 ))
        self.u_axis_rj4_in_wf = r_pf2wf @ self.r_rj12pf @ self.r_rj22rj1 @ self.r_rj32rj2 @ np.vstack(( 0, 1, 0 ))
        self.u_axis_rj5_in_wf = r_pf2wf @ self.r_rj12pf @ self.r_rj22rj1 @ self.r_rj32rj2 @ self.r_rj42rj3 @ np.vstack(( 0, 1, 0 ))
        self.u_axis_rj6_in_wf = r_pf2wf @ self.r_rj12pf @ self.r_rj22rj1 @ self.r_rj32rj2 @ self.r_rj42rj3 @ self.r_rj52rj4 @ np.vstack(( 1, 0, 0 ))
    
    def get_pL_com_in_foot(self):
        """ 用wf的相對關係求出com_in_ft, 再差分得到角動量, 在ALIP軌跡規劃的切換瞬間用到"""
        p_com_in_lf = self.r_lf2wf.T @ (self.p_com_in_wf - self.p_lf_in_wf)
        p_com_in_rf = self.r_rf2wf.T @ (self.p_com_in_wf - self.p_rf_in_wf)
        
        p_com_in_lfrf = np.vstack((p_com_in_lf, p_com_in_rf)) #照lf rf的順序疊起來
        
        
        v_com_in_lfrf = (p_com_in_lfrf - self.__p_com_in_lfrf_past) / timer_period
        vf_com_in_lfrf = 0.7408* self.__vf_com_in_lfrf_past+ 0.2592*self.__v_com_in_lfrf_past  #濾過後的速度(5Hz)
        
        Ly_com_in_lf = m * vf_com_in_lfrf[0,0] * H
        Ly_com_in_rf = m * vf_com_in_lfrf[3,0] * H
        Lx_com_in_lf = -m * vf_com_in_lfrf[1,0] * H
        Lx_com_in_rf = -m * vf_com_in_lfrf[4,0] * H
        
        self.xLy_com_in_lf = np.vstack(( p_com_in_lfrf[0,0], Ly_com_in_lf )) #輸出
        self.yLx_com_in_lf = np.vstack(( p_com_in_lfrf[1,0], Lx_com_in_lf ))
        self.xLy_com_in_rf = np.vstack(( p_com_in_lfrf[3,0], Ly_com_in_rf ))
        self.yLx_com_in_rf = np.vstack(( p_com_in_lfrf[4,0], Lx_com_in_rf ))
        
        self.__p_com_in_lfrf_past = p_com_in_lfrf #update
        self.__v_com_in_lfrf_past = v_com_in_lfrf
        self.__vf_com_in_lfrf_past = vf_com_in_lfrf
        
class UpperLevelController(Node):
    def __init__(self):
                 
        def publisher_create():
            '''effort publisher是ROS2-control的力矩, 負責控制各個關節的力矩->我們程式的目的就是為了pub他'''
            self.publisher['position'] = self.create_publisher(Float64MultiArray , '/position_controller/commands', 10)
            self.publisher['velocity'] = self.create_publisher(Float64MultiArray , '/velocity_controller/commands', 10)
            self.publisher['effort'] = self.create_publisher(Float64MultiArray , '/effort_controllers/commands', 10) #這個才是重點
            self.publisher['velocity_cmd'] = self.create_publisher(Float64MultiArray , '/velocity_command/commands', 10)
            self.publisher['l_gravity'] = self.create_publisher(Float64MultiArray , '/l_gravity', 10)
            self.publisher['r_gravity'] = self.create_publisher(Float64MultiArray , '/r_gravity', 10)
            self.publisher['alip_x'] = self.create_publisher(Float64MultiArray , '/alip_x_data', 10)
            self.publisher['alip_y'] = self.create_publisher(Float64MultiArray , '/alip_y_data', 10)
            self.publisher['torque_l'] = self.create_publisher(Float64MultiArray , '/torqueL_data', 10)
            self.publisher['torque_r'] = self.create_publisher(Float64MultiArray , '/torqueR_data', 10)
            self.publisher['ref'] = self.create_publisher(Float64MultiArray , '/ref_data', 10)
            self.publisher['joint_trajectory_controller'] = self.create_publisher(JointTrajectory , '/joint_trajectory_controller/joint_trajectory', 10)
            self.publisher['pel'] = self.create_publisher(Float64MultiArray , '/px_data', 10)
            self.publisher['com'] = self.create_publisher(Float64MultiArray , '/com_data', 10)
            self.publisher['lf'] = self.create_publisher(Float64MultiArray , '/lx_data', 10)
            self.publisher['rf'] = self.create_publisher(Float64MultiArray , '/rx_data', 10)
           
        def subscriber_create():
            callcount=0 #每5次run一次 main_callback,做decimate(down sampling)降低振盪
            #========================存放訂閱的資料,每5次輸出給self==================================#
            sub = {
                'p_base_in_wf': None,
                'r_base2wf': None,
                'state': None,
                'jp': None
                }
            def base_in_wf_callback(msg):
                ''' 訂閱base_in_wf的位置和旋轉矩陣'''
                base = msg.pose.pose.position
                quaters_base = msg.pose.pose.orientation ##四元數法
                quaters_base = Rotation.from_quat([quaters_base.x, quaters_base.y, quaters_base.z, quaters_base.w])
                sub['p_base_in_wf'] = np.vstack(( base.x, base.y, base.z ))
                sub['r_base2wf'] = quaters_base.as_matrix()
            
            def state_callback(msg):
                """ 接收我們手動pub出的state """
                sub['state'] = msg.data[0]
                    
            def joint_states_callback(msg):
                '''把訂閱到的關節位置、差分與飽和限制算出速度,並轉成我們想要的順序'''
                nonlocal callcount
                callcount += 1
                
                def diff2velocity(jp:np.ndarray, jp_p:np.ndarray) -> np.ndarray:
                    '''差分出速度,加上飽和限制在[-0.75, 0.75]'''                                 
                    jv = (jp - jp_p)/timer_period
                    for i in range(len(jv)):
                        if jv[i]>= 0.75:
                            jv[i] = 0.75
                        elif jv[i]<= -0.75:
                            jv[i] = -0.75
                    return jv
                                                                    
                if len(msg.position) == 12: # 將關節順序轉成我們想要的
                    jp_dict = {joint:value for joint,value in zip(original_joint_order, msg.position)}
                    sub['jp'] =  np.vstack(( jp_dict[joint] for joint in desired_joint_order ))
                    
                if callcount == 5:
                    #========================把sub的資料全部深複製成self的property,使得跑main_callback的時候不會中途被改變==================================#
                    self.pt.p_base_in_wf = deepcopy(sub['p_base_in_wf'])
                    self.pt.r_base2wf = deepcopy(sub['r_base2wf'])
                    self.state = deepcopy(sub['state'])
                    self.jp = deepcopy(sub['jp'])
                    #========================把5次的點差分出速度==================================#
                    self.jv = diff2velocity(self.jp, self.__jp_p) 
                    #========================關節速度濾波==================================#
                    self.jvf = 0.0063*self.__jvf_p - 0.0001383*self.__jvf_pp + 1.014*self.__jv_p - 0.008067*self.__jv_pp #100Hz
                    #========================更新==================================#
                    self.__jvf_pp, self.__jvf_p, self.__jv_pp, self.__jv_p = self.__jvf_p, self.jvf, self.__jv_p, self.jv
                    #==========================================================#
                    self.main_callback()
                    callcount = 0
                       
            self.subscriber['base'] = self.create_subscription(Odometry, '/odom', base_in_wf_callback, 10) #base_state_subscribe
            self.subscriber['state'] = self.create_subscription(Float64MultiArray, 'state_topic', state_callback, 10)
            self.subscriber['joint_states'] = self.create_subscription(JointState, '/joint_states', joint_states_callback, 10) #joint_state_subscribe
 
        def load_URDF():
            '''導入5個模型和1個有運動學和動力學接口的robot'''
            pinocchio_model_dir = "/home/ldsc/ros2_ws/src/bipedal_floating_description/urdf/"
            
            robot = pin.RobotWrapper.BuildFromURDF( #提供完整的運動學和動力學接口
                filename = pinocchio_model_dir + "bipedal_floating.pin.urdf",
                package_dirs = ["."],
                # root_joint=pin.JointModelFreeFlyer(),
                root_joint = None,
                )
            yield robot
            
            print(f"URDF description successfully loaded in {self.robot}")
            
            urdf_filenames = ['bipedal_floating', 'stance_l', 'stance_r_gravity', 'bipedal_l_gravity', 'bipedal_r_gravity']
            #分別是            從骨盆建模,           左單支撐一隻腳, 右單支撐一隻腳,      雙足模型_以左腳建起,    雙足模型_以右腳建起
            
            for urdf_filename in urdf_filenames:
                filedir = pinocchio_model_dir + urdf_filename + '.xacro'
                
                model = pin.buildModelFromUrdf(filedir) #Load urdf模型，只生成模型,不包關高階的數據和功能
                print("model name: " + model.name )
                data = model.createData() # Create data required by the algorithms
                yield model, data
                   
        def tasks_init():
            '''可能不會用到, 但要刪掉前要先試試看'''
            # Tasks initialization for IK
            left_foot_task = FrameTask("l_foot", position_cost=1.0, orientation_cost=1.0,)
            pelvis_task = FrameTask("base_link", position_cost=1.0, orientation_cost=0.0,)
            right_foot_task = FrameTask("r_foot_1", position_cost=1.0, orientation_cost=1.0,)
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
        
        def attach_links(model1_name, link1_name, model2_name, link2_name):
            req = AttachLink.Request()
            req.model1_name = model1_name
            req.link1_name = link1_name
            req.model2_name = model2_name
            req.link2_name = link2_name

            self.future = self.attach_link_client.call_async(req)

        def detach_links(model1_name, link1_name, model2_name, link2_name):
            req = DetachLink.Request()
            req.model1_name = model1_name
            req.link1_name = link1_name
            req.model2_name = model2_name
            req.link2_name = link2_name

            self.future = self.detach_link_client.call_async(req)
        
        
        #========================初始化一些重要的property==================================#
        self.publisher = {} # 存放create的publisher
        self.subscriber = {} # 存放create的subscriber
        self.pt = Allcoodinate() #放各個frame
        
        self.jp = None #現在的關節角度數組
        self.jv = None #現在的關節速度數組(不乾淨)
        self.jvf = None #現在的關節速度數組(濾波)
        self.__jp_p = np.zeros((12,1)) #用來差分或濾波
        self.__jv_p = np.zeros((12,1))
        self.__jv_pp = np.zeros((12,1))
        self.__jvf_p = np.zeros((12,1))
        self.__jvf_pp = np.zeros((12,1))
        
        self.state = None #放我們pub的mode
        self.contact = {'lf':True, 'rf':True} #左右腳是否接觸地面
        
        self.cf = None #放支撐腳： 左腳'lf', 右腳'rf', 雙支撐'2f'
        self.sw = None #放擺動腳: 左腳'lf', 右腳'rf', 雙支撐''(雙支撐狀態應該不會用到)
        
        self.DS_time = 0 #雙支撐的時間進度
        self.contact_t = 0.0 #接觸時間進度
        
        #==========================================================#
        
        super().__init__('upper_level_controllers') #創建了一個叫做'upper_level_controllers'的節點        
        publisher_create()
        subscriber_create()

        (
            self.robot,
            (self.bipedal_floating_model, self.bipedal_floating_data),
            (self.stance_l_model, self.stance_l_data),
            (self.stance_r_model, self.stance_r_data),
            (self.bipedal_l_model, self.bipedal_l_data),
            (self.bipedal_r_model, self.bipedal_r_data),
        ) = list(load_URDF())
        
        # Initialize meschcat visualizer
        self.viz = pin.visualize.MeshcatVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model)
        self.robot.setVisualizer(self.viz, init=False)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        
        # Set initial robot configuration
        print(self.robot.model)
        print(self.robot.q0)
        self.init_configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)
        self.viz.display(self.init_configuration.q)

        # Tasks initialization for IK
        self.tasks = tasks_init()
        
        self.attach_link_client = self.create_client(AttachLink, '/ATTACHLINK') #會把腳底frame跟大地frame在當下的距離固定住
        self.detach_link_client = self.create_client(DetachLink, '/DETACHLINK')
    
    def main_callback(self):

        def com_foot2position_in_pf(jp):
            ''' 回傳(對左腳質點位置,對右腳的,對骨盆的) p.s. 不管是哪個模型,原點都在兩隻腳(相距0.2m)中間'''
            #get com position   
            jp_l, jp_r = jp[:6],jp[6:] #分別取出左右腳的由骨盆往下的joint_position
            
            #右腳為支撐腳的模型
            jp_from_rf = np.vstack(( -jp_r[::-1], jp_l )) #從右腳掌到左腳掌的順序,由於jp_r從右腳對骨盆變成骨盆對右腳,所以要負號
            pin.centerOfMass( self.bipedal_r_model, self.bipedal_r_data, jp_from_rf)
            p_com_in_pf_from_rf = np.reshape(self.bipedal_r_data.com[0],(3,1))
            p_rf_in_pf = np.array([[0.0],[-0.1],[0.0]]) #原點在兩隻腳正中間
            p_rf2com_in_pf = p_com_in_pf_from_rf - p_rf_in_pf

            #左腳為支撐腳的模型
            jp_from_lf = -jp_from_rf[::-1]
            pin.centerOfMass(self.bipedal_l_model, self.bipedal_l_data, jp_from_lf)
            p_com_in_pf_from_lf = np.reshape(self.bipedal_l_data.com[0],(3,1))
            p_lf_in_pf = np.array([[0.0],[0.1],[0]])
            p_lf2com_in_pf = p_com_in_pf_from_lf - p_lf_in_pf

            return p_lf2com_in_pf, p_rf2com_in_pf
        
        def judge_step_firmly():
            '''當腳掌高度(z)在wf<0.01當作踩穩，之後可能要改'''
            self.contact['lf'] == (self.p_lf_in_wf[2,0] <= 0.01)
            self.contact['rf'] == (self.p_rf_in_wf[2,0] <= 0.01)
                
        def stance_change(pa_lf2pel_in_pf:np.ndarray,  pa_rf2pel_in_pf:np.ndarray ): #todo還是看不太懂,要記得改
            """決定不同狀態時支撐模式怎麼做切換"""
            #========================當state0時, 骨盆距離腳的側向(y)在0.06以內, 決定支撐腳==================================#
            if self.state == 0: #判斷單雙支撐
                if abs(pa_lf2pel_in_pf[1,0])<=0.06:
                    self.cf = 'lf'
                elif abs(pa_rf2pel_in_pf[1,0])<=0.06:
                    self.cf = 'rf' #單支撐
                else:
                    self.cf = '2f' #雙支撐
                    
            #========================當state1時, 一開始維持雙支撐, 兩秒後換左腳單支撐==================================#
            elif self.state == 1:
                if self.DS_time <= DS_timeLength:
                    self.cf = '2f' #開始雙支撐
                    self.DS_time += timer_period #更新時間
                    print("DS",self.DS_time)
                else:
                    self.DS_time = 1
                    self.cf = 'lf'
                    self.RSS_time = 0.01 #??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            
            #========================當state30時, #地面踩夠久才換支撐腳==================================#
            elif self.state == 30:
                if abs( self.contact_t-0.5 )<=0.005:
                    if self.cf == 'lf':
                        self.cf = 'rf'
                    elif self.cf == 'rf':
                        self.cf = 'lf'

        #========================建立現在的模型==================================#
        config = pink.Configuration(self.robot.model, self.robot.data, self.jp) 
        self.viz.display(config.q)

        #========================得到各個frame座標==================================#
        self.pt.get_all_frame_needed(config, self.bipedal_floating_model, self.bipedal_floating_data, self.jp)
        
        #得到相對姿勢
        pa_lf2pel_in_pf = self.pt.pa_pel_in_pf - self.pt.pa_lf_in_pf #從pink拿骨盆座標下相對於左右腳掌的骨盆位置
        pa_rf2pel_in_pf = self.pt.pa_pel_in_pf - self.pt.pa_rf_in_pf
        p_lf2com_in_pf, p_rf2com_in_pf = com_foot2position_in_pf(self.jp)

        #===========================================================
        judge_step_firmly()#判斷是否踩穩
        stance_change(pa_lf2pel_in_pf, pa_rf2pel_in_pf) #怎麼切支撐狀態要改!!!!!
        
        #========================軌跡規劃==================================#
        self.trajRef_planning()
        
        torq_j1234_cf, torq_j1234_sw = self.outerloop()
        torq_j56_cf, torq_j56_sw = self.ankle_control()

    def trajRef_planning(self):
        
        def ALIP_trajRef_planning():
            """ 用ALIP規劃出cf,sw,pel _in_wf的軌跡"""
            p_cf_in_wf = self.pt.__dict__[f"p_{self.cf}_in_wf"]
            r_cf2wf = self.pt.__dict__[f"r_{self.cf}2wf"]
            r_wf2cf = r_cf2wf.T
            
            def ALIP_MAT(axis:str, t:float) -> np.ndarray:
                """理想ALIP動態矩陣"""
                if axis == 'x':
                    return np.array([[ cosh(l*t),       sinh(l*t)/(m*H*l) ], 
                                    [ m*H*l*sinh(l*t), cosh(l*t) ]])
                elif axis == 'y':
                    return np.array([[ cosh(l*t),       -sinh(l*t)/(m*H*l) ],
                                    [ -m*H*l*sinh(l*t), cosh(l*t) ]])
            
            def getNextStep_xy_sw2com_in_cf(T:float) -> np.ndarray:
                '''得到下一步擺動腳的落地點'''
                Ly_com_in_cf_T = ( ALIP_MAT('x',T) @ xLy0_com_in_cf[0:2,0] )[1,0] #下步換腳瞬間的角動量
                Lx_com_in_cf_T = ( ALIP_MAT('y',T) @ yLx0_com_in_cf[0:2,0] )[1,0]
                
                
                des_vx_com_in_cf_2T = 0.15 #下下步換腳瞬間的理想Vx
                des_Ly_com_in_cf_2T = m*des_vx_com_in_cf_2T*H #下下步換腳瞬間的理想Ly
                des_Lx_com_in_cf_2T = (0.5*m*H*W)*(l*sinh(l*T))/(1+cosh(l*T)) #下下步換腳瞬間的理想Lx
                if self.sw == 'rf': #下步支撐腳是rf的話要負的
                    des_Lx_com_in_cf_2T = - des_Lx_com_in_cf_2T
                    
                x_sw2com_in_cf_T = (des_Ly_com_in_cf_2T - cosh(l*T) * Ly_com_in_cf_T) / (m*H*l*sinh(l*T)) #下步換腳瞬間的位置
                y_sw2com_in_cf_T = (des_Lx_com_in_cf_2T - cosh(l*T) * Lx_com_in_cf_T) /-(m*H*l*sinh(l*T))
                return np.vstack(( x_sw2com_in_cf_T, y_sw2com_in_cf_T ))
                        
            def get_com_trajpt_in_cf(t:float) -> np.ndarray :
                '''得出到換腳前的com_in_cf的軌跡點'''
                ref_xLy_com_in_cf = ( ALIP_MAT('x',t) @ xLy0_com_in_cf[0:2,0] )
                ref_yLx_com_in_cf = ( ALIP_MAT('y',t) @ yLx0_com_in_cf[0:2,0] )
                ref_p_com_in_cf = np.vstack((ref_xLy_com_in_cf[0,0], ref_yLx_com_in_cf[0,0], H ))
                return ref_p_com_in_cf, ref_xLy_com_in_cf, ref_yLx_com_in_cf

            def get_com2sw_trajpt_in_cf(t:float, p0_sw2com_in_cf:np.ndarray, xy_sw2com_in_cf_T:np.ndarray) -> np.ndarray:
                '''給初始點和下一點, 用弦波連成換腳前的sw_in_cf的軌跡點'''
                tn = t/T
                zCL = 0.02 #踏步高度
                ref_p_sw2com_in_cf = np.vstack((
                    0.5*( (1+cos(pi*tn))*p0_sw2com_in_cf[0:2] + (1-cos(pi*tn))*xy_sw2com_in_cf_T ),
                    4*zCL*(tn-0.5)**2 + (H-zCL)
                ))
                return - ref_p_sw2com_in_cf #scom2sw
            
            if self.state != self.state_past or self.cf != self.cf_past: #切換的瞬間,時間設成0,拿取初始值
                self.contact_t = 0.0 #接觸時間歸零
                self.pt.get_pL_com_in_foot()
                xLy0_com_in_cf = deepcopy( self.pt.__dict__[f"xLy_com_in_{self.cf}"] ) #初始值,一步還沒走完前不能被改變
                yLx0_com_in_cf = deepcopy( self.pt.__dict__[f"yLx_com_in_{self.cf}"] )
                
                p0_sw2com_in_cf = r_wf2cf @ (self.pt.p_com_in_wf - self.pt.p_sw_in_wf) 
                xy_sw2com_in_cf_T = getNextStep_xy_sw2com_in_cf(T)

            ref_p_com_in_cf, ref_xLy_com_in_cf, ref_yLx_com_in_cf = get_com_trajpt_in_cf(self.contact_t) #得到cf下的軌跡點
            ref_p_com2sw_in_cf = get_com2sw_trajpt_in_cf(self.contact_t, p0_sw2com_in_cf, xy_sw2com_in_cf_T)
            ref_p_sw_in_wf = r_cf2wf @ ref_p_com2sw_in_cf + self.pt.p_com_in_wf #sw參考軌跡
            
            ref_p_com_in_wf = r_cf2wf @ ref_p_com_in_cf + p_cf_in_wf #com參考軌跡
            p_com2pel_in_wf = self.pt.p_pel_in_wf - self.pt.p_com_in_wf #com和pel的相對位置
            ref_p_pel_in_wf = p_com2pel_in_wf + ref_p_com_in_wf #pel參考軌跡
            ref_p_pel_in_wf[2] = 0.55
            
            #===========================主要的參考軌跡==================================#
            self.ref_pa_pel_in_wf = np.vstack(( ref_p_pel_in_wf, 0, 0, 0 ))
            self.__dict__[f"ref_pa_{self.cf}_in_wf"] = np.vstack(( p_cf_in_wf, 0, 0, 0 ))
            self.__dict__[f"ref_pa_{self.sw}_in_wf"] = np.vstack(( ref_p_sw_in_wf, 0, 0, 0 ))
            
            #===========================忘了==================================#
            self.ref_pa_com_in_cf = ref_p_com_in_cf #這是幹麻的？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
            self.ref_xLy_com_in_cf, self.ref_yLx_com_in_cf = ref_xLy_com_in_cf, ref_yLx_com_in_cf #用於補償
            
        if self.state == 0:
            self.ref_pa_pel_in_wf = np.vstack(( 0.0,  0.0, 0.55, 0.0, 0.0, 0.0 ))
            self.ref_pa_lf_in_wf  = np.vstack(( 0.0,  0.1, 0.0,  0.0, 0.0, 0.0 ))
            self.ref_pa_rf_in_wf  = np.vstack(( 0.0, -0.1, 0.0,  0.0, 0.0, 0.0 ))
              
        elif self.state == 1: #左右腳都放在0.1, state1:骨盆在0.5DDT內移到左邊0.06(左腳)
            
            targetstate1_y_pel_in_wf = 0.06

            if self.DS_time > 0.0 and self.DS_time <= 0.5*DS_timeLength:
                ref_y_pel_in_wf = targetstate1_y_pel_in_wf * self.DS_time/ (0.5*DS_timeLength)
            else:
                ref_y_pel_in_wf = targetstate1_y_pel_in_wf

            self.ref_pa_pel_in_wf = np.vstack(( 0.0,  ref_y_pel_in_wf, 0.55, 0.0, 0.0, 0.0 ))
            self.ref_pa_lf_in_wf  = np.vstack(( 0.0,  0.1,             0.0,  0.0, 0.0, 0.0 ))
            self.ref_pa_rf_in_wf  = np.vstack(( 0.0, -0.1,             0.0,  0.0, 0.0, 0.0 ))
        
        elif self.state == 30: #ALIP模式
            ALIP_trajRef_planning() #即時規劃出參考命令(下一個軌跡點)
            self.contact_t += timer_period

    

        
    def outerloop(self):
        
        def calculateErr2endVeocity():
            '''計算對骨盆的支撐腳、擺動腳in wf的error, 並經過Pcontrol再轉成幾何速度輸出'''
            kp=20 #p control #todo擺動腳之後要改成PI
            #===========================得到參考命令和量測值==================================#
            ref_pa_pel2cf_in_wf = self.__dict__[f"ref_pa_{self.cf}_in_wf"] - self.ref_pa_pel_in_wf
            ref_pa_pel2sw_in_wf = self.__dict__[f"ref_pa_{self.sw}_in_wf"] - self.ref_pa_pel_in_wf
            pa_pel2cf_in_wf = self.pt.__dict__[f"pa_{self.cf}_in_wf"] - self.pt.pa_pel_in_wf
            pa_pel2sw_in_wf = self.pt.__dict__[f"pa_{self.sw}_in_wf"] - self.pt.pa_pel_in_wf
            
            #===========================經過加法器計算error==================================#
            err_pa_pel2cf_in_wf = ref_pa_pel2cf_in_wf - pa_pel2cf_in_wf
            err_pa_pel2sw_in_wf = ref_pa_pel2sw_in_wf - pa_pel2sw_in_wf

            #===========================經過kp==================================#
            derr_pa_pel2cf_in_wf = kp*err_pa_pel2cf_in_wf
            derr_pa_pel2sw_in_wf = kp*err_pa_pel2sw_in_wf
            
            #===========================轉換成geometry端末角速度==================================#
            w_pel2cf_in_wf = zyx_analy_2_gmtry_w(*ref_pa_pel2cf_in_wf[4:,0]) @ derr_pa_pel2cf_in_wf[-3:] #analytical轉成Geometry
            w_pel2sw_in_wf = zyx_analy_2_gmtry_w(*ref_pa_pel2sw_in_wf[4:,0]) @ derr_pa_pel2sw_in_wf[-3:]

            vw_pel2cf_in_wf = np.vstack(( derr_pa_pel2cf_in_wf[:3], w_pel2cf_in_wf )) 
            vw_pel2sw_in_wf = np.vstack(( derr_pa_pel2sw_in_wf[:3], w_pel2sw_in_wf ))

            return vw_pel2cf_in_wf, vw_pel2sw_in_wf
        
        def Jacobian() -> tuple[np.ndarray, np.ndarray]:
            '''算出左關節對左腳掌, 右關節對右腳掌的Jacobian'''
            pt = self.pt
            #===========================左腳位置和方向的Jacobian==================================#
            J_jl2lf_in_wf = np.array([ [], [], [], [], [], [] ]) #先設6*1全空的一個向量
            for i in range(1,6+1): 
                u_axisli_in_wf = pt.__dict__[f"u_axisl{i}_in_wf"]
                p_jli_in_wf = pt.__dict__[f"p_jl{i}_in_wf"]
                p_jli2lf_in_wf = self.p_lf_in_wf - p_jli_in_wf
                Jpi = np.cross(u_axisli_in_wf, p_jli2lf_in_wf, axis=0 )
                Jai = u_axisli_in_wf
                Ji = np.vstack(( Jpi, Jai ))
                J_jl2lf_in_wf = np.hstack(J_jl2lf_in_wf, Ji)
            
            #===========================右腳位置和方向的Jacobian==================================#
            J_jr2rf_in_wf = np.array([ [], [], [], [], [], [] ]) #先設6*1全空的一個向量
            for i in range(1,6+1):
                u_axisri_in_wf = pt.__dict__[f"u_axisr{i}_in_wf"]
                p_jri_in_wf = pt.__dict__[f"p_jr{i}_in_wf"]
                p_jri2rf_in_wf = self.p_rf_in_wf - p_jri_in_wf
                Jpi = np.cross(u_axisri_in_wf, p_jri2rf_in_wf, axis=0 )
                Jai = u_axisri_in_wf
                Ji = np.vstack(( Jpi, Jai ))
                J_jr2rf_in_wf = np.hstack(J_jr2rf_in_wf, Ji)

            return J_jl2lf_in_wf, J_jr2rf_in_wf
        
        def endVelocity2jointVelocity(vw_pel2cf_in_wf, vw_pel2sw_in_wf):
            '''轉成支撐腳、擺動腳膝上4關節速度命令'''
            if self.state == 30: #ALIP mode
                J_jl2lf_in_wf, J_jr2rf_in_wf = Jacobian()
                J = {'lf': J_jl2lf_in_wf,
                     'rf':J_jr2rf_in_wf}
                jv_ankle_of = {'lf':self.jntFilt['v'][4:6], #左右腳踝關節轉速
                               'rf':self.jntFilt['v'][10:]}
                
                #===========================支撐腳膝上四關節: 控骨盆z, axyz==================================#
                vzwxyz_pel2cf_in_wf = vw_pel2cf_in_wf[2:]
                J_ankle_to_vzwxyz_cf = J[self.cf][2:, 4:]
                J_knee_to_vzwxyz_cf = J[self.cf][2:, :4]
                #---------------------------經過加法器扣除掉腳踝關節的影響----------------------------------#
                vzwxyz_indep_pel2cf_in_wf = vzwxyz_pel2cf_in_wf - J_ankle_to_vzwxyz_cf @ jv_ankle_of[self.cf] #扣掉腳踝關節(i.e.膝下關節)的影響
                #---------------------------經過J^-1----------------------------------#
                J_vzwxyz_cf_inv = np.linalg.pinv( J_knee_to_vzwxyz_cf )
                cmd_v_j1234_cf = J_vzwxyz_cf_inv @ vzwxyz_indep_pel2cf_in_wf
                
                #===========================擺動腳膝上四關節: 控落點xy, az==================================#
                vxywz_pel2sw_in_wf = np.vstack([ vw_pel2sw_in_wf[i] for i in (0,1,-1) ])
                J_ankle_to_vxywz_sw = np.vstack([ J[self.sw][i, 4:] for i in (0,1,-1) ])
                J_knee_to_vxywz_sw = np.vstack([ J[self.sw][i, :4] for i in (0,1,-1) ])
                #---------------------------經過加法器扣除掉腳踝關節的影響----------------------------------#
                vxy_wz_indep_pel2sw_in_wf = vxywz_pel2sw_in_wf - J_ankle_to_vxywz_sw @ jv_ankle_of[self.sw] #扣掉腳踝關節(i.e.膝下關節)的影響
                #---------------------------經過J^-1----------------------------------#
                J_vxy_wz_inv = np.linalg.pinv( J_knee_to_vxywz_sw )
                cmd_v_j1234_sw = J_vxy_wz_inv @ vxy_wz_indep_pel2sw_in_wf
                
                return cmd_v_j1234_cf, cmd_v_j1234_sw
        
        vw_pel2cf_in_wf, vw_pel2sw_in_wf = calculateErr2endVeocity()
        cmd_v_j1234_cf, cmd_v_j1234_sw = endVelocity2jointVelocity(vw_pel2cf_in_wf, vw_pel2sw_in_wf)
        
        #===========================publish出去==================================#
        # cmd_v = {self.cf: np.vstack(( cmd_v_j1234_cf,0,0 )),
        #          self.sw: np.vstack(( cmd_v_j1234_sw,0,0 ))
        # }
        # self.publisher['velocity_cmd'].publish(Float64MultiArray(data=cmd_v['']))
        # vcmd_data = np.array([[vl_cmd[0,0]],[vl_cmd[1,0]],[vl_cmd[2,0]],[vl_cmd[3,0]],[vl_cmd[4,0]],[vl_cmd[5,0]]])
        # self.vcmd_publisher.publish(Float64MultiArray(data=vcmd_data))
        # jv_collect = np.array([[jv[0,0]],[jv[1,0]],[jv[2,0]],[jv[3,0]],[jv[4,0]],[jv[5,0]]])
        # self.velocity_publisher.publish(Float64MultiArray(data=jv_collect))#檢查收到的速度(超髒)
        
        
        return self.innerloop(cmd_v_j1234_cf, cmd_v_j1234_sw)
    
    def innerloop(self,cmd_v_j1234_cf, cmd_v_j1234_sw):
        jv = {'lf':self.jntFilt['v'][:6], #左右腳踝關節轉速
              'rf':self.jntFilt['v'][6:]}
        #===========================膝上四關節進入加法器算誤差==================================#
        err_v_j1234_cf = cmd_v_j1234_cf - jv[self.cf][:4]
        err_v_j1234_sw = cmd_v_j1234_sw - jv[self.sw][:4]
        #===========================經過膝上四關節的kp==================================#
        ksw = np.vstack(( 1, 1, 1, 1))
        if self.contact[self.cf]:
            kcf = np.vstack(( 1.2, 1.2, 1.2, 1.5))
        else:
            kcf = np.vstack(( 1.2, 1.2, 1.2, 1.2))
        
        a_j1234_cf = kcf * err_v_j1234_cf #點乘
        a_j1234_sw = ksw * err_v_j1234_sw #點乘
        #===========================順向動力學==================================#
        torq_j1234_cf = self.forward_dynamics(a_j1234_cf,'cf')
        torq_j1234_sw = self.forward_dynamics(a_j1234_sw,'sw')
        
        return torq_j1234_cf, torq_j1234_sw
        
    def forward_dynamics(self, a_j1234:np.ndarray, ft:str):
        '''內環的動力學計算,目前只計算重力矩, 慣性矩陣設為單位矩陣'''
        def gravity():
            '''計算重力矩, 則設加速度、速度=0'''
            jp_l, jp_r = self.jntSub[:6],self.jntSub[6:] #分別取出左右腳的由骨盆往下的joint_position
            jv = np.zeros((6,1))
            ja = np.zeros((6,1))
            
            #雙支撐的重力矩
            L_DS_gravity = np.reshape(pin.rnea(self.stance_l_model, self.stance_l_data, -jp_l[::-1], jv, ja ),(6,1))  
            L_DS_gravity = - L_DS_gravity[::-1] #順序換回去

            R_DS_gravity = np.reshape(pin.rnea(self.stance_r_model, self.stance_r_data, -jp_r[::-1], jv, ja ),(6,1))  
            R_DS_gravity = - R_DS_gravity[::-1]
            
            DS_gravity = np.vstack((L_DS_gravity, R_DS_gravity))
            
            #右腳單支撐的重力矩
            jp_RSS = np.vstack((-jp_r[::-1],jp_l))
            Leg_RSS_gravity = np.reshape(pin.rnea(self.bipedal_r_model, self.bipedal_r_data, jp_RSS,jv,ja ),(12,1))  
            RSS_gravity = np.vstack(( Leg_RSS_gravity[6:], -Leg_RSS_gravity[6::-1] ))

            #左腳單支撐的重力矩
            jp_LSS = np.vstack(( -jp_l[::-1],jp_r ))
            Leg_LSS_gravity = np.reshape(pin.rnea(self.bipedal_l_model, self.bipedal_l_data, jp_LSS,jv,ja ),(12,1))  
            LSS_gravity = np.vstack(( -Leg_LSS_gravity[6::-1], Leg_LSS_gravity[6:] ))

            if self.cf == 'rf':
                Leg_gravity = 0.3*DS_gravity + 0.75*RSS_gravity
       
            elif self.cf == 'lf':
                Leg_gravity =  0.3*DS_gravity + 0.75*LSS_gravity

            l_leg_gravity, r_leg_gravity = Leg_gravity[:6], Leg_gravity[6:]
            
            self.publisher['l_gravity'].publish(Float64MultiArray(data=l_leg_gravity))
            self.publisher['r_gravity'].publish(Float64MultiArray(data=r_leg_gravity))
            
            return l_leg_gravity,r_leg_gravity
        #===========================計算關節重力矩G==================================#
        gravity_torq_jlf, gravity_torq_jrf = gravity()
        gravity_torq_j1234 = {'lf':gravity_torq_jlf[0:4],
                              'rf':gravity_torq_jrf[0:4]
        }
        #===========================慣量矩陣H==================================#
        inertia_mat = np.identity(4)
        torq_j1234 = inertia_mat @ a_j1234 + gravity_torq_j1234[ft]
        return torq_j1234
    
    def ankle_control(self):
        kp = kp; kd = 0.2
        
        ref_axy_ankle = np.zeros((2,1))
        axy_cf_in_wf = self.pt.__dict__[f"a_{self.cf}_in_wf[:2]"]
        axy_sw_in_wf = self.pt.__dict__[f"a_{self.sw}_in_wf[:2]"]
        
        #===========================膝下兩關節進入加法器算誤差==================================#
        err_ax_cf_in_wf, err_ay_cf_in_wf = (ref_axy_ankle - axy_cf_in_wf).flatten()
        err_ax_sw_in_wf, err_ay_sw_in_wf = (ref_axy_ankle - axy_sw_in_wf).flatten()
        #===========================PD控制力矩==================================#
        if err_ay_cf_in_wf > 0:
            torq_j5_cf = kp * err_ay_cf_in_wf + kd
        else:
            torq_j5_cf = kp * err_ay_cf_in_wf - kd    
        if err_ax_cf_in_wf > 0:
            torq_j6_cf = kp * err_ax_cf_in_wf + kd
        else:
            torq_j6_cf = kp * err_ax_cf_in_wf - kd

            
        if err_ay_sw_in_wf > 0:
            torq_j5_sw = kp * err_ay_sw_in_wf + kd
        else:
            torq_j5_sw = kp * err_ay_sw_in_wf - kd   
        if err_ax_sw_in_wf > 0:
            torq_j6_sw = kp * err_ax_sw_in_wf + kd
        else:
            torq_j6_sw = kp * err_ax_sw_in_wf - kd
                
        return np.vstack(( torq_j5_cf, torq_j6_cf)), np.vstack(( torq_j5_sw, torq_j6_sw))
    
    def estimator(self):
       
        xLy_com_in_cf = self.pt.__dict__[f"xLy_com_in_{self.cf}"] #這次的量測值
        yLx_com_in_cf = self.pt.__dict__[f"yLx_com_in_{self.cf}"]
        
        xLy_com_in_cf_past = self.pt.__dict__[f"xLy_com_in_{self.cf}_past"] #上次的量測值
        xLy_com_in_sw_past = self.pt.__dict__[f"xLy_com_in_{self.sw}_past"]
        yLx_com_in_cf_past = self.pt.__dict__[f"yLx_com_in_{self.cf}_past"]
        yLx_com_in_sw_past = self.pt.__dict__[f"yLx_com_in_{self.sw}_past"]
        
        ob_xLy_com_in_cf_past = self.pt.__dict__[f"ob_xLy_com_in_{self.cf}_past"] #上次的觀測值
        ob_xLy_com_in_sw_past = self.pt.__dict__[f"ob_xLy_com_in_{self.sw}_past"]
        ob_yLx_com_in_cf_past = self.pt.__dict__[f"ob_yLx_com_in_{self.cf}_past"]
        ob_yLx_com_in_sw_past = self.pt.__dict__[f"ob_yLx_com_in_{self.sw}_past"]
        
        
        ux_cf_past = self.__dict__[f"u_x{self.cf}_past"] #上次的輸入
        uy_cf_past = self.__dict__[f"u_y{self.cf}_past"]
        
        #===========================離散版狀態矩陣==================================#
        #x, Ly
        Ax = np.array([
            [1,      0.00247],
            [0.8832, 1]
            ])
        Bx = np.vstack(( 0, 0.01 ))
        
        Kx = np.array([[ 290.3274, 15.0198 ]])*0.5
        Lx = np.array([
            [0.1390, 0.0025],
            [0.8832, 0.2803]
            ])
        #------------------------------------------------------------------------#
        #y, Lx
        Ay = np.array([
            [1,      -0.00247],
            [-0.8832, 1]
            ])
        By = np.vstack(( 0,0.01 ))
        
        Ky = np.array([[ -177.0596, 9.6014 ]])*0.15
        Ly = np.array([
            [ 0.1288, -0.0026],
            [-0.8832,  0.1480]
            ])
        
        #===========================離散版補償器：全都用過去的值==================================#
        oberr_xLy_com_in_cf_past = xLy_com_in_cf_past - ob_xLy_com_in_cf_past #計算估測誤差
        oberr_yLx_com_in_cf_past = yLx_com_in_cf_past - ob_yLx_com_in_cf_past 
        
        # xe1=Ax*e0+B*u0+L*err0 (1是現在,0是過去)
        ob_xLy_com_in_cf = Ax @ ob_xLy_com_in_cf_past + Bx @ ux_cf_past + Lx @ oberr_xLy_com_in_cf_past
        ob_yLx_com_in_cf = Ay @ ob_yLx_com_in_cf_past + By @ uy_cf_past + Ly @ oberr_yLx_com_in_cf_past
        
        #===========================現在的輸入==================================#
        ux_cf = -Kx @ (ob_xLy_com_in_cf - self.ref_xLy_com_in_cf)
        uy_cf = -Ky @ (ob_yLx_com_in_cf - self.ref_yLx_com_in_cf)
        
        #===========================在切換瞬間強迫角動量連續,把扭矩切成0來避免腳沒踩穩==================================#

        if self.cf != self.cf_past: 
            xLy_com_in_cf[1,0] = deepcopy(xLy_com_in_sw_past[1,0]) #擺動腳的過去是支撐腳
            yLx_com_in_cf[1,0] = deepcopy(yLx_com_in_sw_past[1,0]) 
            
            ob_xLy_com_in_cf[1,0] = deepcopy(ob_xLy_com_in_sw_past[1,0])
            ob_yLx_com_in_cf[1,0] = deepcopy(ob_yLx_com_in_sw_past[1,0])

            ux_cf = 0
            uy_cf = 0

        #===========================現在的力矩==================================#
        torq5_cf = -ux_cf
        torq6_cf = -uy_cf
        
        #===========================更新舊值==================================#
        self.__dict__[f"ux_{self.cf}_past"] = ux_cf
        self.__dict__[f"uy_{self.cf}_past"] = uy_cf
        
        self.__dict__[f"xLy_com_in_{self.cf}_past"] = xLy_com_in_cf
        self.__dict__[f"yLx_com_in_{self.cf}_past"] = yLx_com_in_cf
        
        self.__dict__[f"ob_xLy_com_in_{self.cf}_past"] = ob_xLy_com_in_cf
        self.__dict__[f"ob_yLx_com_in_{self.cf}_past"] = ob_yLx_com_in_cf
        
        




        # self.effort_publisher.publish(Float64MultiArray(data=torque))
        tl_data= np.array([[torque[4,0]],[torque[5,0]]])
        self.torque_L_publisher.publish(Float64MultiArray(data=tl_data))


        if stance == 1:
            alip_x_data = np.array([[ref_x_L[0,0]],[ref_x_L[1,0]],[self.ob_x_L[0,0]],[self.ob_x_L[1,0]]])
            alip_y_data = np.array([[ref_y_L[0,0]],[ref_y_L[1,0]],[self.ob_y_L[0,0]],[self.ob_y_L[1,0]]])
            # alip_x_data = np.array([[self.ref_x_L[0,0]],[self.ref_x_L[1,0]],[self.mea_x_L[0,0]],[self.mea_x_L[1,0]]])
            # alip_y_data = np.array([[self.ref_y_L[0,0]],[self.ref_y_L[1,0]],[self.mea_y_L[0,0]],[self.mea_y_L[1,0]]])
            self.alip_x_publisher.publish(Float64MultiArray(data=alip_x_data))
            self.alip_y_publisher.publish(Float64MultiArray(data=alip_y_data))
            

        return torque
    

        
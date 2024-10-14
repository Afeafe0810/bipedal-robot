'''
命名規則：
    點的位置：
        p/x/y/z_點的位置_in_frame名
    座標的姿態角：
        a/ax/ay/az_你的座標frame_in_frame名
    參考命令：
        ref_跟上面一樣
    向量：
        p/x/y/z_點a2點b_in_frame名
    座標的旋轉矩陣：
        rotat_座標a2座標b
'''
'''
更改紀錄：
    L_ref_wf=ref_p_lf_in_wf
    R_ref_wf=ref_p_rf_in_wf
    stance == 0 ->右單支撐
    stance == 1 ->左單支撐
    stance == 2 ->雙支撐
'''
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


def tp(nparray) -> np.ndarray:
    '''統一 1D或2D-array的所有轉置方法'''
    if len(nparray.shape)==1:
        return np.array([nparray]).T
    else:
        return nparray.T

def zyx_analy_2_gmtry_w(ay,az):
    return np.array([[cos(ay)*cos(az), -sin(az),0],
                     [cos(ay)*sin(az), cos(az), 0],
                     [-sin(ay), 0, 1]])

def xyz_rotation(axis,theta):
    if axis == 'x':
        return np.array([[1, 0,          0],
                         [0, cos(theta), -sin(theta)],
                         [0, sin(theta), cos(theta)]])
    if axis == 'y':
        return np.array([[cos(theta), 0, sin(theta)],
                         [0,          1, 0],
                         [-sin(theta),0, cos(theta)]])
    if axis == 'z':
        return np.array([[cos(theta), -sin(theta),0],
                         [sin(theta), cos(theta) ,0],
                         [0,          0,          1]])

class Allcoodinate():
    '''儲存所有座標軸的位置、速度、旋轉矩陣相對其他frame'''
    def __init__(self) -> None:
        pass
    
    def get_rpa_in_pf(self, config, ptname, linkname):
        ''' 儲存位態、旋轉 in pf '''
        pt_in_pf = config.get_transform_frame_to_world(linkname) #骨盆對pink_wf齊次矩陣
        p_pt_in_pf = tp(pt_in_pf.translation) 
        r_pt2pf = pt_in_pf.rotation
        a_pt_in_pf = Rotation.from_matrix(r_pt2pf).as_euler('zyx', degrees=False) [::-1] #把旋轉矩陣換成歐拉角zyx，並轉成ax,ay,az
        pa_pt_in_pf=tp( [ *p_pt_in_pf, *a_pt_in_pf] )
        
        self.__dict__[f'pa_{ptname}_in_pf'] = pa_pt_in_pf
        self.__dict__[f'r_{ptname}_2pf'] = r_pt2pf
    
    def all_rpa_in_pf(self,config):
        ptnames=['pel', 'lf', 'rf']
        linknames=["pelvis_link", "l_foot", "r_foot"]
        for ptname, linkname in zip(ptnames,linknames):
            self.get_pa_in_pf(config, ptname, linkname)
    
    def get_p_in_wf(self, ptname):
        '''
        利用訂閱推導出的的pel_in_wf，做相對轉換
        先算出pf下的向量，再轉到pel，再轉到wf
        '''
        r_pf2pel = np.identity(3) #直走不旋轉
        r_pel2wf = self.r_pel2wf
        p_pel_in_pf = self.pa_pel_in_pf[0:3,0]
        p_pel_in_wf = self.p_pel_in_wf
        
        p_pt_in_pf = self.__dict__[f'pa_{ptname}_in_pf'][0:3,0]
        r_pt2pf = self.__dict__[f'r_{ptname}_2pf']
        
        self.p_pt_in_wf = r_pel2wf @ r_pf2pel @(p_pt_in_pf - p_pel_in_pf) + p_pel_in_wf
        
    def all_p_in_wf(self):
        ptnames=['com', 'lf', 'rf']
        for ptname in ptnames:
            self.get_p_in_wf(ptname)
    
class UpperLevelController(Node):
    timer_period = 0.01 # seconds 跟joint state update rate&取樣次數有關
    DS_timeLength = 2 #雙支撐的時間總長
    
    def __init__(self):
        #定義self的property
        self.publisher = {}; self.subscriber = {}; #存放create的publisher和subscriber
        self.jntSub = { #存放joint_state_callback訂閱的各關節位置，diff2velocity的各關節速度
            'p' : np.zeros(12,1),
            'p_past' : np.zeros(12,1),
            'p_pp' : np.zeros(12,1),
            'v' : np.zeros(12,1),
            'v_past' : np.zeros(12,1),
            'v_pp' : np.zeros(12,1)
            }
        self.jntFilt = { #filter後的self.jntSub
            # 'p' : np.zeros(12,1),
            # 'p_past' : np.zeros(12,1),
            # 'p_pp' : np.zeros(12,1),
            'v' : np.zeros(12,1),
            'v_past' : np.zeros(12,1),
            'v_pp' : np.zeros(12,1)
            }
        self.pt = Allcoodinate()
        self.contact = {'lf':True, 'rf':True}
        self.DS_time = 0
        self.contact_t = 0.0
        
        def publisher_create(self):
            self.publisher['position'] = self.create_publisher(Float64MultiArray , '/position_controller/commands', 10)
            self.publisher['velocity'] = self.create_publisher(Float64MultiArray , '/velocity_controller/commands', 10)
            self.publisher['effort'] = self.create_publisher(Float64MultiArray , '/effort_controllers/commands', 10)
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
              
        def subscriber_create(self):
            callcount=0 #每5次run一次 main_callback，做decimate(down sampling)降低振盪
            sub={}#存放訂閱的資料，每5次輸出給self
            
            def base_in_wf_callback(msg):
                ''' 訂閱base_in_wf的位置和旋轉矩陣'''
                base = msg.pose.pose.position
                quaters_base = msg.pose.pose.orientation ##四元數法
                quaters_base = Rotation.from_quat([quaters_base.x, quaters_base.y, quaters_base.z, quaters_base.w])
                if callcount == 5:
                    sub['pt.p_base_in_wf'] = tp( np.array( [ base.x,base.y,base.z ] ) )
                    sub['pt.r_base2wf'] = quaters_base.as_matrix()
                            
            def contact_callback(msg):
                '''有接觸到才會有訊息'''
                if callcount == 5:
                    if msg.header.frame_id == 'l_foot_1':
                        sub['contact']['lf'] = bool(msg.states)
                    elif msg.header.frame_id == 'r_foot_1':
                        sub['contact']['rf'] = bool(msg.states)
            
            def state_callback(msg):
                sub['state'] = msg.data[0]
                    
            def joint_states_callback(msg):
                '''把訂閱到的關節位置、差分與飽和限制算出速度，並轉成我們想要的順序'''
                nonlocal callcount
                
                p, p_p, p_pp = 'p', 'p_past', 'p_pp'
                v, v_p, v_pp = 'v', 'v_past', 'v_pp'
                
                def diff2velocity(jntSub):
                    '''差分出速度，加上飽和限制在[-0.75, 0.75]'''                                 
                    jntSub[v] = (jntSub[p] - jntSub[p_p])/self.__class__.timer_period
                    for i in range(len(jntSub[v])):
                        if jntSub[v][i]>= 0.75:
                            jntSub[v][i] = 0.75
                        elif jntSub[v][i]<= -0.75:
                            jntSub[v][i] = -0.75
                    
                def joint_velocity_filter(jntSub, jntFilt):
                    # self.jv = 1.1580*self.jv_p - 0.4112*self.jv_pp + 0.1453*self.jv_sub_p + 0.1078*self.jv_sub_pp #10Hz
                    # self.jv = 0.5186*self.jv_p - 0.1691*self.jv_pp + 0.4215*self.jv_sub_p + 0.229*self.jv_sub_pp #20Hz
                    jntFilt[v] = 0.0063*jntFilt[v_p] - 0.0001383*jntFilt[v_pp] + 1.014*jntSub[v_p] - 0.008067*jntSub[v_pp] #100Hz
                    jntFilt[v_pp], jntFilt[v_p], jntSub[v_pp], jntSub[v_p] = jntFilt[v_p], jntFilt[v], jntSub[v_p], jntSub[v]
                    
                # def joint_position_filter(jntSub, jntFilt):
                #     '''濾了也不會用到'''
                #     jntFilt[p] = 1.1580*jntFilt[p_p] - 0.4112*jntFilt[p_pp]+ 0.1453*jntSub[p_p] + 0.1078*jntSub[p_pp] #10Hz
                #     jntFilt[p_pp], jntFilt[p_p], jntSub[p_pp], jntSub[p_p] = jntFilt[p_p], jntFilt[p], jntSub[p_p], jntSub[p]            
                                          
                original_order = ['L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 'L_Ankle_Pitch', 'L_Ankle_Roll', 'R_Hip_Roll',
                                  'R_Hip_Yaw', 'R_Knee_Pitch', 'R_Hip_Pitch', 'R_Ankle_Pitch', 'L_Hip_Roll', 'R_Ankle_Roll']
                desired_order = ['L_Hip_Roll', 'L_Hip_Yaw', 'L_Hip_Pitch', 'L_Knee_Pitch', 'L_Ankle_Pitch', 'L_Ankle_Roll',
                                 'R_Hip_Roll', 'R_Hip_Yaw', 'R_Hip_Pitch', 'R_Knee_Pitch', 'R_Ankle_Pitch', 'R_Ankle_Roll']
                
                if len(msg.position) == 12: # 將順序轉成我們想要的
                    position_order_dict = {joint: value for joint, value in zip(original_order, np.array(msg.position))}
                    sub['jntSub'][p] = tp( np.array( [ position_order_dict[joint] for joint in desired_order ] ))
                    
                callcount += 1
                if callcount == 5:
                    for key in sub.keys:#把sub的資料全部深複製成self的property，使得跑main_callback的時候不會中途被改變
                        self.__dict__[key] = deepcopy(sub[key])
                    diff2velocity(self.jntSub)
                    joint_velocity_filter(self.jntSub, self.jntFilt)
                    #joint_position_filter(self.jntSub, self.jntFilt)
                    self.main_callback()
                    callcount = 0
                       
            self.subscriber['base'] = self.create_subscription(Odometry, '/odom', base_in_wf_callback, 10) #base_state_subscribe
            self.subscriber['l_foot_contact'] = self.create_subscription(ContactsState, '/l_foot/bumper_demo', contact_callback, 10) #l_foot_contact_state_subscribe
            self.subscriber['r_foot_contact'] = self.create_subscription(ContactsState, '/r_foot/bumper_demo', contact_callback, 10) #r_foot_contact_state_subscribe
            self.subscriber['state'] = self.create_subscription(Float64MultiArray, 'state_topic', state_callback, 10)
            self.subscriber['joint_states'] = self.create_subscription(JointState, '/joint_states', joint_states_callback, 10) #joint_state_subscribe
                   
        def maybe_can_ignore(self):
            #position PI
            self.Le_dot_past = 0.0
            self.Le_past = 0.0
            self.Re_dot_past = 0.0
            self.Re_past = 0.0
        
        def tasks_init():
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
                  
        super().__init__('upper_level_controllers') #創建了一個叫做'upper_level_controllers'的節點
        publisher_create(self)
        subscriber_create(self)
        self.robot = self.load_URDF("/home/ldsc/ros2_ws/src/bipedal_floating_description/urdf/bipedal_floating.pin.urdf")
        
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
        self.tasks = self.tasks_init()
        #===========================================================
        

        self.tt = 0
        self.P_Y_ref = 0.0

        self.stance = 2
        self.stance_past = 2
        self.RSS_time = 0.0
        self.LSS_time = 0.0
        self.RSS_count = 0
        self.RDT = 1
        self.LDT = 1

        self.delay = 0

        #data_in_pf 

        #ALIP
        #time
        self.ALIP_count = 0
        self.alip_t = 0.0
        #--velocity
        self.CX_past_L = 0.0
        self.CX_dot_L = 0.0
        self.CY_past_L = 0.0
        self.CY_dot_L = 0.0
        self.CX_past_R = 0.0
        self.CX_dot_R = 0.0
        self.CY_past_R = 0.0
        self.CY_dot_R = 0.0
        #--velocity filter
        self.Vx_L = 0.0
        self.Vx_past_L = 0.0
        self.CX_dot_past_L = 0.0
        self.Vy_L = 0.0
        self.Vy_past_L = 0.0
        self.CY_dot_past_L = 0.0
        self.Vx_R = 0.0
        self.Vx_past_R = 0.0
        self.CX_dot_past_R = 0.0
        self.Vy_R = 0.0
        self.Vy_past_R = 0.0
        self.CY_dot_past_R = 0.0
        #--measurement
        self.mea_x_L = np.zeros((2,1))
        self.mea_x_past_L = np.zeros((2,1))
        self.mea_y_L = np.zeros((2,1))
        self.mea_y_past_L = np.zeros((2,1))
        self.mea_x_R = np.zeros((2,1))
        self.mea_x_past_R = np.zeros((2,1))
        self.mea_y_R = np.zeros((2,1))
        self.mea_y_past_R = np.zeros((2,1))
        #--compensator
        self.ob_x_L = np.zeros((2,1))
        self.ob_x_past_L = np.zeros((2,1))
        self.ob_y_L = np.zeros((2,1))
        self.ob_y_past_L = np.zeros((2,1))
        self.ob_x_R = np.zeros((2,1))
        self.ob_x_past_R = np.zeros((2,1))
        self.ob_y_R = np.zeros((2,1))
        self.ob_y_past_R = np.zeros((2,1))
        #--torque
        self.ap_L = 0.0
        self.ap_past_L = 0.0
        self.ar_L = 0.0
        self.ar_past_L = 0.0
        self.ap_R = 0.0
        self.ap_past_R = 0.0
        self.ar_R = 0.0
        self.ar_past_R = 0.0

        #--ref    
        self.ref_x_L = np.zeros((2,1))
        self.ref_y_L = np.zeros((2,1))
        self.ref_x_R = np.zeros((2,1))
        self.ref_y_R = np.zeros((2,1))

        #touch for am tracking check
        self.touch = 0
 
        # Initialize the service client
        self.attach_link_client = self.create_client(AttachLink, '/ATTACHLINK')
        self.detach_link_client = self.create_client(DetachLink, '/DETACHLINK')

    def ALIP_trajRef_planning(self):#todo 記得改mea的地方
        #理想機器人狀態
        m = 9    #機器人下肢總重
        H = 0.45 #理想質心高度
        W = 0.2 #兩腳底間距
        g = 9.81 #重力
        l = sqrt(g/H)
        T = 0.5 #支撐間隔時長
        
        mea0_xLy_com_in_cf = self.pt.__dict__[f"mea_xLy_com_in_{self.cf}"]
        mea0_yLx_com_in_cf = self.pt.__dict__[f"mea_yLx_com_in_{self.cf}"]
        p_cf_in_wf = self.pt.__dict__[f"p_{self.cf}_in_wf"]
        r_cf2wf = self.pt.__dict__[f"r_{self.cf}2wf"]
        r_wf2cf = r_cf2wf.T
        
        def ALIP_MAT(axis,t):
            """理想ALIP動態矩陣"""
            if axis == 'x':
                return np.array([[cosh(l*t),(sinh(l*t)/(m*H*l))], 
                                 [m*H*l*sinh(l*t),cosh(l*t)]])
            elif axis == 'y':
                return np.array([[cosh(l*t),-(sinh(l*t)/(m*H*l))],
                                 [-m*H*l*sinh(l*t),cosh(l*t)]])
         
        def getNextStep_xy_sw2com_in_cf(T):
            '''得到下一步擺動腳的落地點'''
            _, Ly_com_in_cf_T = ( ALIP_MAT('x',T) @ mea0_xLy_com_in_cf[0:2,0] ).T #下步換腳瞬間的角動量
            _, Lx_com_in_cf_T= ( ALIP_MAT('y',T) @ mea0_yLx_com_in_cf[0:2,0] ).T
            
            
            des_vx_com_in_cf_2T = 0.15 #下下步換腳瞬間的理想Vx
            des_Ly_com_in_cf_2T = m*des_vx_com_in_cf_2T*H #下下步換腳瞬間的理想Ly
            des_Lx_com_in_cf_2T = (0.5*m*H*W)*(l*sinh(l*T))/(1+cosh(l*T)) #下下步換腳瞬間的理想Lx
            if self.sw == 'rf': #下步支撐腳是rf的話要負的
                des_Lx_com_in_cf_2T = - des_Lx_com_in_cf_2T
                
            x_sw2com_in_cf_T = (des_Ly_com_in_cf_2T - cosh(l*T) * Ly_com_in_cf_T) / (m*H*l*sinh(l*T)) #下步換腳瞬間的位置
            y_sw2com_in_cf_T = (des_Lx_com_in_cf_2T - cosh(l*T) * Lx_com_in_cf_T) /-(m*H*l*sinh(l*T))
            return np.array([[x_sw2com_in_cf_T], [y_sw2com_in_cf_T]])
                       
        def get_com_trajpt_in_cf(t):
            '''得出到換腳前的com_in_cf的軌跡點'''
            ref_x_com_in_cf, _ = ( ALIP_MAT('x',t) @ mea0_xLy_com_in_cf[0:2,0] ).T
            ref_y_com_in_cf, _ = ( ALIP_MAT('y',t) @ mea0_yLx_com_in_cf[0:2,0] ).T
            ref_p_com_in_cf = np.array([[ref_x_com_in_cf, ref_y_com_in_cf, H ]]).T
            return ref_p_com_in_cf

        def get_com2sw_trajpt_in_cf(t, p0_sw2com_in_cf, xy_sw2com_in_cf_T):
            '''給初始點和下一點, 用弦波連成換腳前的sw_in_cf的軌跡點'''
            tn = t/T
            zCL = 0.02 #踏步高度
            ref_p_sw2com_in_cf = np.vstack((
                 0.5*( (1+cos(pi*tn))*p0_sw2com_in_cf[0:2] + (1-cos(pi*tn))*xy_sw2com_in_cf_T ),
                 4*zCL*(tn-0.5)**2 + (H-zCL)
            ))
            return - ref_p_sw2com_in_cf #scom2sw
        
        if self.state != self.state_past or self.cf != self.cf_past: #切換的瞬間，時間設成0，拿取初始值
            self.contact_t = 0.0 #接觸時間歸零
            
            p0_sw2com_in_cf = r_wf2cf @ (self.pt.p_com_in_wf - self.pt.p_sw_in_wf) 
            xy_sw2com_in_cf_T = getNextStep_xy_sw2com_in_cf(T)

        ref_p_com_in_cf = get_com_trajpt_in_cf(self.contact_t) #得到cf下的軌跡點
        ref_p_com2sw_in_cf = get_com2sw_trajpt_in_cf(self.contact_t, p0_sw2com_in_cf, xy_sw2com_in_cf_T)
        ref_p_sw_in_wf = r_cf2wf @ ref_p_com2sw_in_cf + self.pt.p_com_in_wf #sw參考軌跡
        
        ref_p_com_in_wf = r_cf2wf @ ref_p_com_in_cf + p_cf_in_wf #com參考軌跡
        p_com2pel_in_wf = self.pt.p_pel_in_wf - self.pt.p_com_in_wf #com和pel的相對位置
        ref_p_pel_in_wf = p_com2pel_in_wf + ref_p_com_in_wf #pel參考軌跡
        ref_p_pel_in_wf[2] = 0.55
        
        self.__dict__[f"ref_pa_{self.cf}_in_wf"] = np.vstack(( p_cf_in_wf, 0, 0, 0 ))
        self.__dict__[f"ref_pa_{self.sw}_in_wf"] = np.vstack(( ref_p_sw_in_wf, 0, 0, 0 ))
        self.ref_pa_pel_in_wf = np.vstack(( ref_p_pel_in_wf, 0, 0, 0 ))
        
    def outerloop(self):
        def calculateErr2endVeocity():
            '''計算error並經過Pcontrol，再轉成幾何角速度'''
            #控制相對於骨盆的形狀in wf
            ref_pa_pel2lf_in_wf = self.ref_pa_lf_in_wf - self.ref_pa_pel_in_wf
            ref_pa_pel2rf_in_wf = self.ref_pa_rf_in_wf - self.ref_pa_pel_in_wf
            pa_pel2lf_in_wf = self.pt.pa_lf_in_wf - self.pt.pa_pel_in_wf
            pa_pel2rf_in_wf = self.pt.pa_rf_in_wf - self.pt.pa_pel_in_wf

            err_pa_pel2lf_in_wf = ref_pa_pel2lf_in_wf - pa_pel2lf_in_wf
            err_pa_pel2rf_in_wf = ref_pa_pel2rf_in_wf - pa_pel2rf_in_wf

            kp=20 #p control
            derr_pa_pel2lf_in_wf = kp*err_pa_pel2lf_in_wf
            derr_pa_pel2rf_in_wf = kp*err_pa_pel2rf_in_wf
            
            w_lf_in_wf = zyx_analy_2_gmtry_w(*ref_pa_pel2lf_in_wf[2:]) @ derr_pa_pel2lf_in_wf[-3:-1] #analytical轉成Geometry
            w_rf_in_wf = zyx_analy_2_gmtry_w(*ref_pa_pel2rf_in_wf[2:]) @ derr_pa_pel2rf_in_wf[-3:-1]

            vw_lf_in_wf = np.vstack(( derr_pa_pel2lf_in_wf[0:3], w_lf_in_wf )) 
            vw_rf_in_wf = np.vstack(( derr_pa_pel2rf_in_wf[0:3], w_rf_in_wf ))

            return vw_lf_in_wf, vw_rf_in_wf
        
        def velocity_cmd(self,vw_lf_in_wf, vw_rf_in_wf):
            v =  copy.deepcopy(jv_f) #joint_velocity
            #獲取支撐狀態
            # print(stance)
            if self.state == 30: #ALIP mode
                if self.cf == 'rf':
                    #(右支撐腳腳踝動態排除)
                    R2_41 = np.reshape(vw_rf_in_wf[2:,0],(4,1)) #R2 z to wz
                    VR56 =  np.reshape(v[10:,0],(2,1)) #右腳腳踝速度
                    #計算右膝關節以上速度
                    R2_41_cal = R2_41 - self.JR_sp42@VR56
                    #彙整右腳速度
                    rw_41_d = np.dot(np.linalg.pinv(self.JR_sp44),R2_41_cal)
                    rw_21_d = np.zeros((2,1))
                    Rw_d = np.vstack((rw_41_d,rw_21_d))

                    #(左擺動腳腳踝動態排除)
                    #拿左腳 誤差及腳踝速度
                    L2_41 = np.array([[L2[0,0]],[L2[1,0]],[L2[2,0]],[L2[5,0]]]) #x y z yaw
                    VL56 =  np.reshape(v[4:6,0],(2,1)) #左腳腳踝速度
                    #計算左膝關節以上速度
                    L2_41_cal = L2_41 - self.JL_sw42@VL56
                    #彙整左腳速度
                    lw_41_d = np.dot(np.linalg.pinv(self.JL_sw44),L2_41_cal)
                    lw_21_d = np.zeros((2,1))
                    Lw_d = np.vstack((lw_41_d,lw_21_d))

                    # Lw_d = np.dot(np.linalg.pinv(self.JLL),L2) 
                    
                elif self.cf =='lf':
                    #(左支撐腳腳踝動態排除)
                    #拿左腳 誤差及腳踝速度
                    L2_41 = np.reshape(L2[2:,0],(4,1)) #L2 z to wz
                    VL56 =  np.reshape(v[4:6,0],(2,1)) #左腳腳踝速度
                    #計算左膝關節以上速度
                    L2_41_cal = L2_41 - self.JL_sp42@VL56
                    #彙整左腳速度
                    lw_41_d = np.dot(np.linalg.pinv(self.JL_sp44),L2_41_cal)
                    lw_21_d = np.zeros((2,1))
                    Lw_d = np.vstack((lw_41_d,lw_21_d))

                    #(右擺動腳腳踝動態排除)
                    #拿右腳 誤差及腳踝速度
                    R2_41 = np.array([[R2[0,0]],[R2[1,0]],[R2[2,0]],[R2[5,0]]]) #x y z yaw
                    VR56 =  np.reshape(v[10:,0],(2,1)) #右腳腳踝速度
                    #計算右膝關節以上速度
                    R2_41_cal = R2_41 - self.JR_sw42@VR56
                    #彙整右腳速度
                    rw_41_d = np.dot(np.linalg.pinv(self.JR_sw44),R2_41_cal)
                    rw_21_d = np.zeros((2,1))
                    Rw_d = np.vstack((rw_41_d,rw_21_d))
                    # Rw_d = np.dot(np.linalg.pinv(self.JRR),R2) 
            else:
                Lw_d = np.dot(np.linalg.pinv(self.JLL),L2) 
                Rw_d = np.dot(np.linalg.pinv(self.JRR),R2) 
            
            return Lw_d,Rw_d
        
        vw_lf_in_wf, vw_rf_in_wf = calculateErr2endVeocity()
        VL,VR = velocity_cmd(Le_2,Re_2,jv_f,stance,state)
    
    def innerloop(self):
        pass

    def ref_cmd(self):
        '''在0.5DDT內移到左邊0.06'''
        pyLth = 0.06
        if self.state == 0:
            P_Y_ref = 0.0
        elif self.state == 1:
            if self.DS_time > 0.0 and self.DS_time <= 0.5*self.DDT:
                P_Y_ref = pyLth*( self.DS_time/(0.5*self.DDT) )
            else:
                P_Y_ref = pyLth

        self.ref_pa_pel_in_wf = np.vstack(( 0.0, P_Y_ref, 0.55, 0.0, 0.0, 0.0 ))
        self.ref_pa_lf_in_wf = np.vstack(( 0.0, 0.1, 0.0, 0.0, 0.0, 0.0 ))
        self.ref_pa_rf_in_wf = np.vstack(( 0.0, -0.1, 0.0, 0.0, 0.0, 0.0 ))
        
    def main_callback(self):

        def com_position_in_pf(jp):
            ''' 回傳(對左腳質點位置，對右腳的，對骨盆的) p.s. 不管是哪個模型，原點都在兩隻腳(相距0.2m)中間'''
            #get com position   
            jp_l, jp_r = jp[:6],jp[6:] #分別取出左右腳的由骨盆往下的joint_position
            
            #右腳為支撐腳的模型
            jp_from_rf = np.vstack(( -jp_r[::-1], jp_l )) #從右腳掌到左腳掌的順序，由於jp_r從右腳對骨盆變成骨盆對右腳，所以要負號
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

            #floating com #todo 可以確認看看不同模型建立出來的質心會不會不一樣
            joint_angle = np.vstack((jp_l,jp_r))
            pin.centerOfMass(self.bipedal_floating_model, self.bipedal_floating_data, jp)
            p_com_in_pf = np.reshape(self.bipedal_floating_data.com[0],(3,1))

            return p_lf2com_in_pf, p_rf2com_in_pf, p_com_in_pf
        
        def pelvis_in_wf(self):
            '''base_in_wf_callback訂閱到的base_in_wf的位態求出pel_in_wf的位態'''
            p_pel_in_base = tp(np.array([0.0, 0.0, 0.598]))
            self.pt.p_pel_in_wf = self.pt.r_base2wf @ p_pel_in_base + self.pt.p_base_in_wf
            self.pt.r_pel2wf = deepcopy(self.pt.r_base2wf) #因為兩者是平移建立的
        
        def stance_change(self, pa_lf2pel_in_pf, pa_rf2pel_in_pf,contact_t): #todo還是看不太懂，要記得改
            '''
            state0利用雙腳左右距離來判斷是哪隻腳支撐/雙支撐(骨盆距某腳0.06內為支撐腳，其他狀態是雙支撐)
            state1...剩下的都看不太懂
            '''
            
            #==============================================#todo
            self.sw= ( {'lf','rf'}-{self.cf} ).pop()
            #==============================================
            
            
            
            if self.state == 0: #判斷單雙支撐
                if abs(pa_lf2pel_in_pf[1,0])<=0.06 or abs(pa_rf2pel_in_pf[1,0])<=0.06: #當其中一隻腳距骨盆0.06內
                    self.stance = 1 #單支撐
                else:
                    self.stance = 2 #雙支撐
                    
            elif self.state == 1: #運動學控制調到ALIP的初始位置
                if self.DS_time <= self.__class__.DS_timeLength:
                    self.stance = 2 #開始雙支撐
                    self.DS_time += self.__class__.timer_period #更新時間
                    print("DS",self.DS_time)
                else:
                    self.DS_time = 10.1
                    stance = 1
                    self.RSS_time = 0.01

            if state == 30:
                #踩到地面才切換支撐腳
                if abs(contact_t-0.5)<=0.005:#(T)
                    if self.stance == 1:
                        stance = 0
                        # if self.P_R_wf[2,0] <= 0.01:
                        #     stance = 0
                        # else:
                        #     stance = 1
                    elif self.stance == 0:
                        stance = 1
                        # if self.P_L_wf[2,0] <= 0.01:
                        #     stance = 1
                        # else:
                        #     stance = 0
                else:
                    self.stance = stance
            self.stance = stance
            return stance 
        
        config = pink.Configuration(self.robot.model, self.robot.data, self.jntSub['p'])
        self.viz.display(config.q)

        self.pt.all_rpa_in_pf(config)
        pa_lf2pel_in_pf = self.pt.pa_pel_in_pf - self.pt.pa_lf_in_pf #從pink拿骨盆座標下相對於左右腳掌的骨盆位置
        pa_rf2pel_in_pf = self.pt.pa_pel_in_pf - self.pt.pa_rf_in_pf
        
        p_lf2com_in_pf, p_rf2com_in_pf, self.p_com_in_pf = com_position_in_pf(self.jntSub['p']) #也得到com_in_pf

        pelvis_in_wf(self) #利用base求得pel_in_wf
        self.pt.all_p_in_wf() #利用座標轉換求得_in_wf
        
        #===========================================================
        #這邊算相對的矩陣
        self.rotation_matrix(joint_position) #todo還沒改
        #這邊算wf下各軸姿態
        self.relative_axis() #todo還沒改

        #學長我想問一下這個地方，我們不是從subscriber拿到左右腳接觸的訊息了嗎，為什麼還要自己判斷
        l_contact,r_contact = self.contact_collect()

        if self.P_L_wf[2,0] <= 0.01:##\\\\接觸的判斷是z方向在不在0.01以內
            l_contact == 1
        else:
            l_contact == 0
        if self.P_R_wf[2,0] <= 0.01:
            r_contact == 1
        else:
            r_contact == 0
            
        #怎麼切支撐狀態要改!!!!!
        stance,stance_past = self.stance_change(state,px_in_lf,px_in_rf,self.stance,self.ALIP_count)
        #===========================================================
        
        if self.state == 30: #ALIP模式
            self.ALIP_trajRef_planning() #得到參考命令
            self.contact_t += self.__class__.timer_period
        else:
            self.ref_cmd()
            l_leg_gravity,r_leg_gravity,kl,kr = self.gravity_compemsate(joint_position,stance,px_in_lf,px_in_rf,l_contact,r_contact,state)
        
        

    
    def jnt_rotat_mat(self,jp):#todo 要再修改
        self.pt
        
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

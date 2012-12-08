#!/usr/bin/env python
import roslib
roslib.load_manifest('FlyODE')
import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import *

import pygame
from pygame.locals import *
import ode
import numpy as np

from fly_plot_lib import flymath

window_size = (800,600)
fps = 500
dt = 1.0/fps
loopFlag = True
clk = pygame.time.Clock()

config = {  'body_mass': 1, # mg
            'head_mass': 1e-2, 
            'antenna_mass': 1e-3,
            'arista_mass': 1e-3,
            'body_radius': 1, # mm
            'head_radius': 1,
            'antenna_radius': 1e-1,
            'arista_length': 1,
            'arista_radius': 1e-1,
            'head_body_hinge_F': 1000,
            'antenna_head_hinge_F': 1000,
            'arista_antenna_hinge_F': 1000,
         }

def coord(x,y,field_of_view=100,window_size=(640,480)): # field of view in m
    "Convert world coordinates to pixel coordinates."
    fov_x = field_of_view
    fov_y = field_of_view/float(window_size[0])*float(window_size[1])
    wrapped_coord_x = np.remainder(x+fov_x/2.,fov_x)
    wrapped_coord_y = np.remainder(y+fov_y/2.,fov_y)
    return int(wrapped_coord_x/fov_x*window_size[0]), int(window_size[1]-wrapped_coord_y/fov_y*window_size[1])

def draw_canvas(srf):
    srf.fill((255,255,255))
    #pygame.draw.line(srf, (0,0,0), coord(0,0,window_size=window_size), coord(0.1,0,window_size=window_size), 1)
    #pygame.draw.line(srf, (0,0,0), coord(0,0,window_size=window_size), coord(0,0.1,window_size=window_size), 1)
    



class FlyModel(object):
    def __init__(self, world, dt):
        self.mass = config['body_mass']
        self.dt = dt
    
        # Create Fly Head
        mass_head = config['head_mass']
        radius_head = config['head_radius']
        volume_head = 4/3.*np.pi*radius_head**3
        density_head = mass_head/volume_head
        fly_head = ode.Body(world)
        M_head = ode.Mass()
        M_head.setSphere(density_head, radius_head)
        fly_head.setMass(M_head)
        fly_head.setPosition((0,0,0))
        self.head = fly_head

        # Create Fly Body
        mass_body = self.mass
        self.radius_body = config['body_radius']
        volume_body = 4/3.*np.pi*self.radius_body**3
        density_body = mass_body/volume_body
        fly_body = ode.Body(world)
        M_body = ode.Mass()
        M_body.setSphere(density_body, self.radius_body)
        fly_body.setMass(M_body)
        y_pos_start = fly_head.getPosition()[1] - radius_head - self.radius_body
        fly_body.setPosition((0,y_pos_start,0))
        self.body = fly_body
        
        # Attach Head and Body - locked hinge motor
        self.joint_head_body = ode.HingeJoint(world)
        self.joint_head_body.attach(self.head, self.body)
        self.joint_head_body.setAnchor((0,fly_head.getPosition()[1] - radius_head,0))
        self.joint_head_body.setAxis((0,0,1))
        #self.joint_head_body.setParam(ode.ParamVel, 0)
        self.joint_head_body.setParam(ode.ParamFMax, config['head_body_hinge_F'])
        
        ### Set Up Antenna ###
        
        ## Antenna (1st segment)
        mass_antenna = config['antenna_mass']
        radius_antenna = config['antenna_radius']
        volume_antenna = 4/3.*np.pi*radius_antenna**3
        density_antenna = mass_antenna/volume_antenna
        M_antenna = ode.Mass()
        M_antenna.setSphere(density_antenna, radius_antenna)
        
        ## Arista - cylinder
        mass_arista = config['arista_mass']
        length_arista = config['arista_length']
        self.arista_length = length_arista
        radius_arista = config['arista_radius']
        volume_arista = 2*np.pi*radius_arista**2*length_arista
        density_arista = mass_arista/volume_arista
        M_arista = ode.Mass()
        M_arista.setSphere(density_arista, radius_arista)
        
        # Left Side
        antenna_angle = 30*np.pi/180. + self.get_body_orientation()
        # Antenna
        fly_antenna_l = ode.Body(world)
        fly_antenna_l.setMass(M_antenna)
        x_pos_start = fly_head.getPosition()[1] + radius_head*np.cos(antenna_angle)
        y_pos_start = fly_head.getPosition()[1] + radius_head*np.sin(antenna_angle)
        fly_antenna_l.setPosition((x_pos_start,y_pos_start,0))
        self.antenna_l = fly_antenna_l
        # attach to head
        self.joint_antenna_l_head = ode.HingeJoint(world)
        self.joint_antenna_l_head.attach(self.antenna_l, self.head)
        self.joint_antenna_l_head.setAnchor(self.antenna_l.getPosition())
        self.joint_antenna_l_head.setAxis((0,0,1))
        #self.joint_antenna_l_head.setParam(ode.ParamVel, 0)
        self.joint_antenna_l_head.setParam(ode.ParamFMax, config['antenna_head_hinge_F'])
        self.joint_antenna_l_head.setParam(ode.ParamLoStop, -0.01)
        self.joint_antenna_l_head.setParam(ode.ParamHiStop, np.pi/5.5)
        # Arista
        fly_arista_l = ode.Body(world)
        fly_arista_l.setMass(M_head)
        x_pos_start = self.antenna_l.getPosition()[0]+length_arista*np.cos(antenna_angle)
        y_pos_start = self.antenna_l.getPosition()[1]+length_arista*np.sin(antenna_angle)
        fly_arista_l.setPosition((x_pos_start,y_pos_start,0))
        quat = list(fly_arista_l.getQuaternion())
        # self.body.getQuaternion()[3]*2+np.pi/2.
        quat[3] = (antenna_angle-np.pi/2.)/2.
        fly_arista_l.setQuaternion(quat)
        self.arista_l = fly_arista_l
        # attach to antenna
        self.joint_arista_antenna_l = ode.HingeJoint(world)
        self.joint_arista_antenna_l.attach(self.arista_l, self.antenna_l)
        self.joint_arista_antenna_l.setAnchor(self.antenna_l.getPosition())
        self.joint_arista_antenna_l.setAxis((0,0,1))
        #self.joint_arista_antenna_l.setParam(ode.ParamVel, 0)
        self.joint_arista_antenna_l.setParam(ode.ParamFMax, config['arista_antenna_hinge_F'])
        self.joint_arista_antenna_l.setFeedback(True)
        
        if 1:
            # Right Side
            antenna_angle = -30*np.pi/180. + self.get_body_orientation()
            # Antenna
            fly_antenna_r = ode.Body(world)
            fly_antenna_r.setMass(M_antenna)
            x_pos_start = fly_head.getPosition()[1] + radius_head*np.cos(antenna_angle)
            y_pos_start = fly_head.getPosition()[1] + radius_head*np.sin(antenna_angle)
            fly_antenna_r.setPosition((x_pos_start,y_pos_start,0))
            self.antenna_r = fly_antenna_r
            # attach to head
            self.joint_antenna_r_head = ode.HingeJoint(world)
            self.joint_antenna_r_head.attach(self.antenna_r, self.head)
            self.joint_antenna_r_head.setAnchor(self.antenna_r.getPosition())
            self.joint_antenna_r_head.setAxis((0,0,1))
            #self.joint_antenna_r_head.setParam(ode.ParamVel, 0)
            self.joint_antenna_r_head.setParam(ode.ParamFMax, config['antenna_head_hinge_F'])
            self.joint_antenna_r_head.setParam(ode.ParamLoStop, -np.pi/5.5)
            self.joint_antenna_r_head.setParam(ode.ParamHiStop, .01)
            # Arista
            fly_arista_r = ode.Body(world)
            fly_arista_r.setMass(M_head)
            x_pos_start = self.antenna_r.getPosition()[0]+length_arista*np.cos(antenna_angle)
            y_pos_start = self.antenna_r.getPosition()[1]+length_arista*np.sin(antenna_angle)
            fly_arista_r.setPosition((x_pos_start,y_pos_start,0))
            quat = list(fly_arista_r.getQuaternion())
            quat[3] = (antenna_angle-np.pi/2.)/2.
            fly_arista_r.setQuaternion(quat)
            self.arista_r = fly_arista_r
            # attach to antenna
            self.joint_arista_antenna_r = ode.HingeJoint(world)
            self.joint_arista_antenna_r.attach(self.arista_r, self.antenna_r)
            self.joint_arista_antenna_r.setAnchor(self.antenna_r.getPosition())
            self.joint_arista_antenna_r.setAxis((0,0,1))
            #self.joint_arista_antenna_r.setParam(ode.ParamVel, 0)
            self.joint_arista_antenna_r.setParam(ode.ParamFMax, config['arista_antenna_hinge_F'])
            self.joint_arista_antenna_r.setFeedback(True)
            
        self.slipangle = 0
        self.rand_slip_offset = np.pi/6.
        self.antenna_difference_history = []
        self.visual_slip_hover_history = []
        
        # set up rosnode
        rospy.init_node("fly_ode_simulation")
        
        # set up joystick
        self.joy = None
        rospy.Subscriber("joy", Joy, self.save_joy)
        self.arista_l_pub = rospy.Publisher('arista_l', Float32)
        self.arista_r_pub = rospy.Publisher('arista_r', Float32)
        
        self.arista_l_sensor_lowpass = 0
        self.arista_l_torque_lowpass = 0
        
        self.arista_l_sensor_history = []
        
    def get_body_orientation(self):
        # 2D
        sign = np.sign(self.body.getQuaternion()[3])
        if sign == 0:
            sign = 1
        woundup = (np.arccos(self.body.getQuaternion()[0])*2)*sign+np.pi/2.
        return flymath.fix_angular_rollover(woundup)
        
    def get_arista_orientation_l(self):
        sign = np.sign(self.arista_l.getQuaternion()[3])
        if sign == 0:
            sign = 1
        woundup = (np.arccos(self.arista_l.getQuaternion()[0])*2)*sign+np.pi/2.
        return flymath.fix_angular_rollover(woundup) 
        
    def get_arista_orientation_r(self):
        sign = np.sign(self.arista_r.getQuaternion()[3])
        if sign == 0:
            sign = 1
        woundup = (np.arccos(self.arista_r.getQuaternion()[0])*2)*sign+np.pi/2.
        return flymath.fix_angular_rollover(woundup) 
    
    def save_joy(self, Joy):
        self.joy = Joy
        
    def apply_joy_forces(self):
        if self.joy is None:
            return 
        
        gain_thrust = 2000
        gain_yaw = 100
        
        thrust = gain_thrust*self.joy.axes[1]
        yaw = gain_yaw*self.joy.axes[2]
        
        antenna_left_vel = -10*(self.joy.axes[12])
        antenna_right_vel = 10*(self.joy.axes[13])
        
        ori = self.get_body_orientation()
        force = (np.cos(ori)*thrust, np.sin(ori)*thrust, 0)
        
        rand = self.rand_slip_offset #0#(np.random.random()*2 -1)*np.pi/2.
        self.slipangle = 0#np.pi/2.*0.3#np.abs(rand) #self.rand_slip_offset
        
        if self.joy.buttons[13] > 0:
            antenna_difference = fly.joint_arista_antenna_l.getFeedback()[3][2] - -1*fly.joint_arista_antenna_r.getFeedback()[3][2]
            print antenna_difference
        else:
            antenna_difference = 0
            
        self.antenna_difference_history.append(antenna_difference)
        
        history_length = 20
        if len(self.antenna_difference_history) > history_length:
            antenna_difference_integral = np.sum(self.antenna_difference_history[-1*history_length])
        else:
            antenna_difference_integral = 0
            
        #print self.joint_arista_antenna_l.getFeedback()[3][2] - -1*self.joint_arista_antenna_r.getFeedback()[3][2], self.slipangle, antenna_difference_integral
        #print self.slipangle + -0.8*antenna_difference_integral
        
        if len(self.antenna_difference_history) > 2:
            #control = -0.1*antenna_difference_integral-0.05*antenna_difference -0.0*(self.body.getAngularVel()[2])#+ self.joy.axes[0]*100
            control = -0.05*antenna_difference_integral-0.1*antenna_difference -0.0*(self.body.getAngularVel()[2])#+ self.joy.axes[0]*100
        else:
            control = 0
            
        ###
        vel_ori = np.arctan2(self.body.getLinearVel()[1],self.body.getLinearVel()[0])
        body_ori = self.get_body_orientation()
        #print flymath.fix_angular_rollover(vel_ori-body_ori)        
        ###
        
        #print (self.body.getLinearVel()[1])
            
        self.body.addRelForce((thrust*np.sin(self.slipangle+control),thrust*np.abs(np.cos(self.slipangle+control)),0))
        self.body.addRelTorque((0,0,yaw))
        self.joint_antenna_l_head.setParam(ode.ParamVel, antenna_left_vel)
        self.joint_antenna_r_head.setParam(ode.ParamVel, antenna_right_vel)
        
        if self.joy.buttons[12]:
            self.hover()
            
    def cast(self, sign=-1):
        self.body.addRelTorque((0,0,sign*(self.body.getLinearVel()[1])))
        
    def hover(self):
        #self.body.addRelTorque((0,0,1*(self.body.getLinearVel()[0])))
        
        if 1:
            # calculate orientation of velocity
            vel_ori = np.arctan2(self.body.getLinearVel()[1],self.body.getLinearVel()[0])
            body_ori = self.get_body_orientation()
            body_ori_dot = self.body.getAngularVel()[2]
            gain = 300
            damping = 50
            self.visual_slip_hover_history.append(flymath.fix_angular_rollover(vel_ori-body_ori))
            if len(self.visual_slip_hover_history) > 10:
                visual_slip_integral = np.sum(self.visual_slip_hover_history[-10:])
            else:
                visual_slip_integral = 0
                
            
            control = -0*visual_slip_integral + -80*self.visual_slip_hover_history[-1]- 10*body_ori_dot
            
            self.body.addRelTorque((0,0,control))
            
            body_vec = np.array([np.cos(self.get_body_orientation()), np.sin(self.get_body_orientation()), 0])
            vel_comp_in_direc_of_travel = np.dot(body_vec, np.array(self.body.getLinearVel()))
            #self.body.addRelForce((0,-50*(vel_comp_in_direc_of_travel-10),0))
        
        
    def drive_antenna_to_zero(self):
        self.arista_l_sensor_history.append(self.arista_l.getAngularVel()[2])
        
        if len(self.arista_l_sensor_history) > 20:
            arista_integral = np.sum(self.arista_l_sensor_history[-5:-1]) / 4
        else:
            arista_integral = 0
        
        if np.abs(self.body.getAngularVel()[2]) > 1e-1: 
            self.joint_antenna_l_head.setParam(ode.ParamVel, -2*self.arista_l_sensor_history[-1])
        else:
            antenna_left_vel = -5*(self.joy.axes[12])
            self.joint_antenna_l_head.setParam(ode.ParamVel, antenna_left_vel)
            
        body_ori = self.get_body_orientation()
        arista_ori = self.get_arista_orientation_l()
        body_ori_dot = self.body.getAngularVel()[2]
        
        torque_dot = self.joint_arista_antenna_l.getFeedback()[3][2] - self.arista_l_torque_lowpass
        self.arista_l_torque_lowpass += 0.8*(self.joint_arista_antenna_l.getFeedback()[3][2] - self.arista_l_torque_lowpass)
        
        v_est = torque_dot / (np.cos(body_ori)*body_ori_dot*np.cos(arista_ori+body_ori) + np.sin(body_ori)*body_ori_dot*np.sin(arista_ori+body_ori)) 
        
        #print self.arista_l.getAngularVel()[2] - self.body.getAngularVel()[2]

        #if np.abs(body_ori_dot) > 1e-3 and np.abs(self.arista_l.getAngularVel()[2]) < 1e-3:
        #    print v_est, self.arista_l.getAngularVel()[2]
         
        print self.joint_antenna_l_head.getParam(ode.ParamVel)
                
    def apply_aero_body(self, wind=(0,0,0)):
        
        # get airspeed, groundspeed = airspeed + wind; airspeed = groundspeed - wind
        groundspeed = self.body.getLinearVel()
        airspeed = np.array(groundspeed) - np.array(wind)
        
        airspeed_norm = np.linalg.norm(airspeed)
        if airspeed_norm > 0:
            airspeed_dir = airspeed / airspeed_norm
        else:
            airspeed_dir = np.zeros(3)
            
            
        weight = 9.81*self.mass
        force_drag = -0.8*weight/1.*airspeed_norm*airspeed_dir
        self.body.addForce(force_drag.tolist())
        
        self.body.addTorque((0,0,-10*self.body.getAngularVel()[2]))
        
    def apply_aero_arista(self, wind=(0,0,0)):
    
        # dot product of airspeed and orientation
    
        # LEFT
        groundspeed = self.arista_l.getLinearVel()
        airspeed = np.array(groundspeed) - np.array(wind)
        ori_vec = np.array([np.cos(self.get_arista_orientation_l()), np.sin(self.get_arista_orientation_l()), 0])
        weight = 9.81*config['arista_mass']
        force_drag = -0.8*weight/1.*np.cross(airspeed, ori_vec)[2]
        self.arista_l.addRelForce((force_drag,0,0))
        #self.arista_l.addRelForce((0,0,0))
        
        # RIGHT
        groundspeed = self.arista_r.getLinearVel()
        airspeed = np.array(groundspeed) - np.array(wind)
        ori_vec = np.array([np.cos(self.get_arista_orientation_r()), np.sin(self.get_arista_orientation_r()), 0])
        weight = 9.81*config['arista_mass']
        force_drag = -0.8*weight/1.*np.cross(airspeed, ori_vec)[2]
        self.arista_r.addRelForce((force_drag,0,0))
        
    def draw(self, srf, window_size):
        
        head_x, head_y, head_z = self.head.getPosition()
        body_x, body_y, body_z = self.body.getPosition()
        antenna_l_x, antenna_l_y, antenna_l_z = self.antenna_l.getPosition()
        
        head_color = (200,0,55)
        body_color = (55,0,200)
        antenna_color = (0,0,0)
        pygame.draw.circle(srf, head_color, coord(head_x, head_y,window_size=window_size), 10, 0)
        pygame.draw.circle(srf, body_color, coord(body_x, body_y,window_size=window_size), 10, 0)
        pygame.draw.circle(srf, antenna_color, coord(antenna_l_x, antenna_l_y,window_size=window_size), 3, 0)
        if 1:
            antenna_r_x, antenna_r_y, antenna_r_z = self.antenna_r.getPosition()
            pygame.draw.circle(srf, antenna_color, coord(antenna_r_x, antenna_r_y,window_size=window_size), 3, 0)
        
        # left arista
        tip = self.arista_l.getRelPointPos( (0,self.arista_length/2.,0) )
        base = self.arista_l.getRelPointPos( (0,-1*self.arista_length/2.,0) )
        pygame.draw.line(srf, antenna_color, coord(base[0],base[1],window_size=window_size), coord(tip[0],tip[1],window_size=window_size), 1)

        # right arista
        if 1:
            tip = self.arista_r.getRelPointPos( (0,self.arista_length/2.,0) )
            base = self.arista_r.getRelPointPos( (0,-1*self.arista_length/2.,0) )
            pygame.draw.line(srf, antenna_color, coord(base[0],base[1],window_size=window_size), coord(tip[0],tip[1],window_size=window_size), 1)
            #pygame.draw.circle(srf, antenna_color, coord(self.arista_r.getPosition()[0],self.arista_r.getPosition()[1],window_size=window_size), 2, 0)

# Initialize pygame
pygame.init()

# Open a display
srf = pygame.display.set_mode(window_size)

# Create a world object
world = ode.World()
world.setGravity((0,0,0))

fly = FlyModel(world, dt)


# Simulation loop...



while loopFlag:
    events = pygame.event.get()
    for e in events:
        if e.type==QUIT:
            loopFlag=False
        if e.type==KEYDOWN:
            loopFlag=False

    # Clear the screen
    draw_canvas(srf)
    
    # Draw fly
    fly.draw(srf, window_size)
    
    fly.apply_joy_forces()
    #
    
    fly.apply_aero_body(wind=(-10,-10,0))
    fly.apply_aero_arista(wind=(-10,-10,0))
    
    torque_left = fly.joint_arista_antenna_l.getFeedback()[3][2]
    alpha = 0.9
    fly.arista_l_sensor_lowpass = fly.arista_l_sensor_lowpass + alpha*(torque_left-fly.arista_l_sensor_lowpass)
    
    if 0:
        if fly.joy is not None:
            if fly.joy.buttons[13]:
                fly.drive_antenna_to_zero()
                #fly.hover()
            
    #print (fly.arista_r.getQuaternion()[3]*2+np.pi/2.)*180/np.pi, (fly.arista_l.getQuaternion()[3]*2+np.pi/2.)*180/np.pi
    #print fly.get_body_orientation()*180/np.pi
    #print fly.arista_r.getQuaternion(), -1*(np.arccos(fly.arista_r.getQuaternion()[3])*2 - np.pi/180.*270)*180/np.pi
    
    
    
    if np.abs(fly.joint_arista_antenna_l.getFeedback()[3][2]) < 1e-6:
        fly.arista_l_pub.publish(fly.joint_arista_antenna_l.getFeedback()[3][2])
    else:
        fly.arista_l_pub.publish(np.sign(fly.joint_arista_antenna_l.getFeedback()[3][2])*1e-6)

    if np.abs(fly.joint_arista_antenna_r.getFeedback()[3][2]) < 1e-6:
        fly.arista_r_pub.publish(fly.joint_arista_antenna_r.getFeedback()[3][2])
    else:
        fly.arista_r_pub.publish(np.sign(fly.joint_arista_antenna_r.getFeedback()[3][2])*1e-6)
            
    pygame.display.flip()

    # Next simulation step
    world.step(dt)

    # Try to keep the specified framerate    
    clk.tick(fps)

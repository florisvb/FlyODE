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

HAVE_ROS = True

window_size = (800,600)
fps = 500
dt = 1.0/fps
loopFlag = True
clk = pygame.time.Clock()

# UNITS: mm, mg, sec

config = {  'body_mass': 1, # mg
            'head_mass': 1e-2, 
            'antenna_mass': 1e-3,
            'arista_mass': 1e-3,
            'body_radius': 1, # mm
            'head_radius': 1,
            'antenna_radius': 1e-1,
            'arista_length': 1,
            'arista_radius': 1e-1,
            'head_body_hinge_F': 1000000,
            'antenna_head_hinge_F': 10000,
            'arista_antenna_hinge_F': 100000,
            'antenna_max_angle': np.pi/10.,
            'antenna_initial_angle': 30*np.pi/180.,
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
        
        # Attach Head and Body - hinge motor that is feedback driven to 0
        self.joint_head_body = ode.HingeJoint(world)
        self.joint_head_body.attach(self.head, self.body)
        self.joint_head_body.setAnchor((0,fly_head.getPosition()[1] - radius_head,0))
        self.joint_head_body.setAxis((0,0,1))
        self.joint_head_body.setParam(ode.ParamFMax, config['head_body_hinge_F'])
        
        ### Set Up Antenna ###
        
        ## Antenna parameters
        mass_antenna = config['antenna_mass']
        radius_antenna = config['antenna_radius']
        volume_antenna = 4/3.*np.pi*radius_antenna**3
        density_antenna = mass_antenna/volume_antenna
        M_antenna = ode.Mass()
        M_antenna.setSphere(density_antenna, radius_antenna)
        
        ## Arista parameters (cylinder shape)
        mass_arista = config['arista_mass']
        length_arista = config['arista_length']
        self.arista_length = length_arista
        radius_arista = config['arista_radius']
        volume_arista = 2*np.pi*radius_arista**2*length_arista
        density_arista = mass_arista/volume_arista
        M_arista = ode.Mass()
        M_arista.setSphere(density_arista, radius_arista)
        
        # Left Side
        antenna_angle = config['antenna_initial_angle'] + self.get_body_orientation()
        # Antenna
        fly_antenna_l = ode.Body(world)
        fly_antenna_l.setMass(M_antenna)
        x_pos_start = fly_head.getPosition()[1] + radius_head*np.cos(antenna_angle)
        y_pos_start = fly_head.getPosition()[1] + radius_head*np.sin(antenna_angle)
        fly_antenna_l.setPosition((x_pos_start,y_pos_start,0))
        self.antenna_l = fly_antenna_l
        # Attach Antenna to Head
        self.joint_antenna_l_head = ode.HingeJoint(world)
        self.joint_antenna_l_head.attach(self.antenna_l, self.head)
        self.joint_antenna_l_head.setAnchor(self.antenna_l.getPosition())
        self.joint_antenna_l_head.setAxis((0,0,1))
        self.joint_antenna_l_head.setParam(ode.ParamFMax, config['antenna_head_hinge_F'])
        self.joint_antenna_l_head.setParam(ode.ParamLoStop, -0.01)
        self.joint_antenna_l_head.setParam(ode.ParamHiStop, config['antenna_max_angle'])
        # Arista
        fly_arista_l = ode.Body(world)
        fly_arista_l.setMass(M_head)
        x_pos_start = self.antenna_l.getPosition()[0]+length_arista*np.cos(antenna_angle)
        y_pos_start = self.antenna_l.getPosition()[1]+length_arista*np.sin(antenna_angle)
        fly_arista_l.setPosition((x_pos_start,y_pos_start,0))
        quat = list(fly_arista_l.getQuaternion())
        quat[3] = (antenna_angle-np.pi/2.)/2.
        fly_arista_l.setQuaternion(quat)
        self.arista_l = fly_arista_l
        # Attach arista to antenna
        self.joint_arista_antenna_l = ode.HingeJoint(world)
        self.joint_arista_antenna_l.attach(self.arista_l, self.antenna_l)
        self.joint_arista_antenna_l.setAnchor(self.antenna_l.getPosition())
        self.joint_arista_antenna_l.setAxis((0,0,1))
        self.joint_arista_antenna_l.setParam(ode.ParamFMax, config['arista_antenna_hinge_F'])
        self.joint_arista_antenna_l.setFeedback(True)
        
        # Right Side
        antenna_angle = -1*config['antenna_initial_angle'] + self.get_body_orientation()
        # Antenna
        fly_antenna_r = ode.Body(world)
        fly_antenna_r.setMass(M_antenna)
        x_pos_start = fly_head.getPosition()[1] + radius_head*np.cos(antenna_angle)
        y_pos_start = fly_head.getPosition()[1] + radius_head*np.sin(antenna_angle)
        fly_antenna_r.setPosition((x_pos_start,y_pos_start,0))
        self.antenna_r = fly_antenna_r
        # Attach antenna to head
        self.joint_antenna_r_head = ode.HingeJoint(world)
        self.joint_antenna_r_head.attach(self.antenna_r, self.head)
        self.joint_antenna_r_head.setAnchor(self.antenna_r.getPosition())
        self.joint_antenna_r_head.setAxis((0,0,1))
        self.joint_antenna_r_head.setParam(ode.ParamFMax, config['antenna_head_hinge_F'])
        self.joint_antenna_r_head.setParam(ode.ParamLoStop, -1*config['antenna_max_angle'])
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
        # Attach arista to antenna
        self.joint_arista_antenna_r = ode.HingeJoint(world)
        self.joint_arista_antenna_r.attach(self.arista_r, self.antenna_r)
        self.joint_arista_antenna_r.setAnchor(self.antenna_r.getPosition())
        self.joint_arista_antenna_r.setAxis((0,0,1))
        self.joint_arista_antenna_r.setParam(ode.ParamFMax, config['arista_antenna_hinge_F'])
        self.joint_arista_antenna_r.setFeedback(True)
            
        self.slipangle = 0
        self.rand_slip_offset = np.pi/6.
        self.antenna_difference_history = []
        self.visual_slip_hover_history = []
        
        # set up lowpass filters
        self.antenna_difference_lowpassed = 0
        self.visual_slip_lowpassed = 0
        
        if HAVE_ROS:
            # set up rosnode
            rospy.init_node("fly_ode_simulation")
            
            # set up joystick
            self.joy = None
            rospy.Subscriber("joy", Joy, self.save_joy)
            self.arista_l_pub = rospy.Publisher('arista_l', Float32)
            self.arista_r_pub = rospy.Publisher('arista_r', Float32)
            
        ## set up controls for fly
        
        # joystick controls
        self.joystick_thrust = 0
        self.joystick_yaw = 0
        self.joystick_antenna_left_vel = 0
        self.joystick_antenna_right_vel = 0
        
        # automatic controls
        self.thrust = 0
        self.yaw = 0
        self.antenna_left_vel = 0
        self.antenna_right_vel = 0
        
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
        
        # Joystick gains
        gain_thrust = 2000
        gain_yaw = 100
        
        # Joystick controls
        self.joystick_thrust = gain_thrust*self.joy.axes[1]
        self.joystick_yaw = gain_yaw*self.joy.axes[2]
        
        self.joystick_antenna_left_vel = -10*(self.joy.axes[12])
        self.joystick_antenna_right_vel = 10*(self.joy.axes[13])
        
        
    def apply_forces(self, wind=(0,0,0)):
        if self.joy is None:
            return 
        
        thrust = self.joystick_thrust + self.thrust
        yaw = self.joystick_yaw + self.yaw
        antenna_left_vel = self.joystick_antenna_left_vel + self.antenna_left_vel
        antenna_right_vel = self.joystick_antenna_right_vel + self.antenna_right_vel
        
        ori = self.get_body_orientation()
        force = (np.cos(ori)*thrust, np.sin(ori)*thrust, 0)
        
        # Add random slip angle
        rand = self.rand_slip_offset #0#(np.random.random()*2 -1)*np.pi/2.
        self.slipangle = 0#np.pi/2.*0.3#np.abs(rand) #self.rand_slip_offset
        
        # Get airspeed (for gain scheduling)
        groundspeed = self.body.getLinearVel()
        airspeed = np.array(groundspeed) - np.array(wind)
        airspeed_mag = np.linalg.norm(airspeed)
        
        # Calculate antenna controls
        force_drag_l, force_drag_r = self.apply_aero_arista(wind=wind, apply_forces=False)
        antenna_difference = -1*(force_drag_l + force_drag_r)
        self.antenna_difference_lowpassed += 0.9*(antenna_difference - self.antenna_difference_lowpassed)
        control = -0.5*airspeed_mag*self.antenna_difference_lowpassed # gain scheduled proportional controller
        # set control limits
        if np.abs(control) > np.pi/2.:
            control = np.sign(control)*np.pi/2.
        
        # Apply forces
        self.body.addRelForce((thrust*np.sin(flymath.fix_angular_rollover(self.slipangle+control)),thrust*np.abs(np.cos(flymath.fix_angular_rollover(self.slipangle+control))),0))
        self.body.addRelTorque((0,0,yaw))
        self.joint_antenna_l_head.setParam(ode.ParamVel, antenna_left_vel)
        self.joint_antenna_r_head.setParam(ode.ParamVel, antenna_right_vel)
        
        # Visual controllers:
        if self.joy.buttons[12]:
            self.hover()
            
        ## Control locked hinges
        # control head
        self.joint_head_body.addTorque(-1000*self.joint_head_body.getAngle())
        
        # control arista
        self.joint_arista_antenna_l.addTorque(-100*self.joint_arista_antenna_l.getAngle())
        self.joint_arista_antenna_r.addTorque(-100*self.joint_arista_antenna_r.getAngle())
        
        # Apply aerodynamics
        self.apply_aero_body(wind=wind)
        self.apply_aero_arista(wind=wind)
        
            
    def hover(self):
        groundspeed = self.body.getLinearVel()
        airspeed = np.array(groundspeed) - np.array(wind)
        airspeed_mag = np.linalg.norm(airspeed)
    
        # calculate orientation of velocity
        vel_ori = np.arctan2(self.body.getLinearVel()[1],self.body.getLinearVel()[0])
        body_ori = self.get_body_orientation()
        body_ori_dot = self.body.getAngularVel()[2]
        gain = -10*airspeed_mag
        damping = 50
        
        visual_slip = flymath.fix_angular_rollover(vel_ori-body_ori)
        self.visual_slip_lowpassed += 0.9*(visual_slip - self.visual_slip_lowpassed)
            
        control = gain*self.visual_slip_lowpassed - damping*body_ori_dot
        
        self.body.addRelTorque((0,0,control))
        
                
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
        
    def apply_aero_arista(self, wind=(0,0,0), apply_forces=True):
    
        # dot product of airspeed and orientation
        groundspeed = self.body.getLinearVel()
        
        # LEFT
        #groundspeed = self.arista_l.getLinearVel()
        airspeed = np.array(groundspeed) - np.array(wind)
        ori_vec = np.array([np.cos(self.get_arista_orientation_l()), np.sin(self.get_arista_orientation_l()), 0])
        weight = 9.81*config['arista_mass']
        force_drag = -0.5*weight/1.*np.sign(np.cross(airspeed, ori_vec)[2])*np.abs(np.cross(airspeed, ori_vec)[2])
        force_drag_l = force_drag
        if apply_forces:
            self.arista_l.addRelForce((force_drag,0,0))
        
        # RIGHT
        #groundspeed = self.arista_r.getLinearVel()
        airspeed = np.array(groundspeed) - np.array(wind)
        ori_vec = np.array([np.cos(self.get_arista_orientation_r()), np.sin(self.get_arista_orientation_r()), 0])
        weight = 9.81*config['arista_mass']
        force_drag = -0.5*weight/1.*np.sign(np.cross(airspeed, ori_vec)[2])*np.abs(np.cross(airspeed, ori_vec)[2])
        force_drag_r = force_drag 
        if apply_forces:
            self.arista_r.addRelForce((force_drag,0,0))
        
        if not apply_forces:
            return force_drag_l, force_drag_r
        
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

# Create a world object, and the fly in it
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
    
    wind=(-20,-20,0)
    #wind = (0,0,0)
    fly.apply_forces(wind=wind)
    
    
            
    pygame.display.flip()

    # Next simulation step
    world.step(dt)

    # Try to keep the specified framerate    
    clk.tick(fps)

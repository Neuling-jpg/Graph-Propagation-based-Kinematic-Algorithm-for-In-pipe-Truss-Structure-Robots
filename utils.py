import numpy as np
from params import *

def uv2xyz(u,v, R=18., r=6):
    z = (1.5*R + r*np.cos(v))*np.cos(u)
    x = (1.5*R + r*np.cos(v))*np.sin(u)
    y = r*np.sin(v)
    return x,y,z

def cal_ini_xyz(theta):
    u1, v1 = theta, 0
    u2, v2 = theta+np.pi/48, 2*np.pi/3
    u3, v3 = theta-np.pi/24, -2*np.pi/3

    x1, y1, z1 = uv2xyz(u1, v1, R_, r_)
    x2, y2, z2 = uv2xyz(u2, v2, R_, r_)
    x3, y3, z3 = uv2xyz(u3, v3, R_, r_)
    return [x1,y1,z1], [x2,y2,z2], [x3,y3,z3]

def base_coords(u, z=0):
    x = 6*np.cos(u)
    y = 6*np.sin(u)
    return x,y,z

def obj_ee(pos, target):
    if target==None: return 0., [0., 0., 0.]
    assert type(target)==list, "Wrong target input."
    assert len(target)==3, "Wrong target dimension."
    q1 = np.array(pos)-np.array(target)
    q2 = np.sqrt(np.sum(np.square(q1)))
    grad = q1/q2
    pipe_alignment = q2
    return pipe_alignment, grad

def obj_straight_pipe(pos, target):
    r = r_
    q1 = (pos[0]**2 + pos[1]**2)**0.5
    q2 = q1 - r
    dl_dx = pos[0]/q1
    dl_dy = pos[1]/q1
    if q2 < 0:
        dl_dx, dl_dy = -dl_dx, -dl_dy
    pipe_alignment = np.abs(q2)
    return pipe_alignment, [dl_dx, dl_dy, 0.]

def obj_torus_pipe(pos, target):
    R, r = R_, r_
    q1 = (pos[0]**2+pos[2]**2)**0.5
    q2 = R-q1
    q3 = (q2**2+pos[1]**2)**0.5
    q4 = q3-r
    dl_dx = -pos[0]*q2/(q3*q1)
    dl_dz = -pos[2]*q2/(q3*q1)
    dl_dy = pos[1]/q3
    if q4 <0:
        dl_dx, dl_dy, dl_dz = -dl_dx, -dl_dy, -dl_dz
    pipe_alignment = np.abs(q4)
    return pipe_alignment, [dl_dx, dl_dy, dl_dz]
import time
import random
import numpy as np
import vpython as v
import argparse

from robot_graph import RobotStructGraph
from utils import base_coords
from params import *

def ee_path_follow(use_propagation=True, sol="fmin_bfgs", n_nodes=14, freeze_num=0, path_name="cmu", visualize=False):
    end_effector = ["N{}".format(n_nodes)]
    rg = RobotStructGraph(num_nodes=n_nodes, task="end_traj_follow", end_effector=end_effector)
    
    u1 = np.pi
    u2 = -np.pi/3
    u3 = np.pi/3

    x1, y1, z1 = base_coords(u1, 0)
    x2, y2, z2 = base_coords(u2, 0)
    x3, y3, z3 = base_coords(u3, 0)

    def robot_forward(w):
        try:
            for idn, n in enumerate(rg.node_list[3:]):
                n.ul1.w = w[3*idn]  
                n.ul2.w = w[3*idn+1]
                n.ul3.w = w[3*idn+2] 
        except:
            for idn, n in enumerate(rg.node_list[3:]):
                n.ul1.w = w[0][3*idn]  
                n.ul2.w = w[0][3*idn+1]
                n.ul3.w = w[0][3*idn+2] 
        target_point = [None]*(n_nodes-1)
        target_point.append(target)
        alignment = rg.forward(target_point)
        return alignment
    
    def robot_backward(w):
        grad_list = []
        rg.grads.reverse()
        rg.backward()
        for idn, n in enumerate(rg.node_list[3:]):
            grad_list.append(n.ul1.dl_dw)
            grad_list.append(n.ul2.dl_dw)
            grad_list.append(n.ul3.dl_dw)
        return np.array(grad_list)

    rg.input_vn([x1, y1, z1], [x2, y2, z2], [x3, y3, z3])
    for idn, n in enumerate(rg.node_list[3:]):
                alignment, grad = n.forward(None, False, rg.task)
    
    # random freeze
    links = []
    for n in rg.node_list:
        if n.ul1 is not None: links.append(n.ul1)
        if n.ul2 is not None: links.append(n.ul2)
        if n.ul3 is not None: links.append(n.ul3)
    random.shuffle(links)
    for l in links[:min(freeze_num, len(links))]:
        l.sr = 0.

    # c_cmu
    with open("reference_ee_path/{}.txt".format(path_name), "r") as f:
        target_path = []
        x = f.read()
        x = x.split("\n")[:-1]
        for xx in x:
            xx_list = xx.split(" ")
            xx_list = [float(_) for _ in xx_list]
            target_path.append(xx_list)
    
    err_list, time_list = [], []
    # initialization: reach to start point
    target = target_path[0]
    error, time, robot_nodes = rg.update(ff=robot_forward, fb=robot_backward, solver=sol, use_propagation=use_propagation)
    err_list.append(error)
    time_list.append(time)
    
    path_links, path_ee = [], []
    for iter in range(len(target_path)):
        target = target_path[iter]
        error, time, robot_nodes = rg.update(ff=robot_forward, fb=robot_backward, solver=sol, use_propagation=use_propagation)
        err_list.append(error)
        time_list.append(time)
        print("Reaching target point #{}/{}, runtime: {:3f}, objective func value: {}".format(iter, len(target_path), time, error))

        links_params = []
        for n in rg.node_list:
            if n.ul1 is not None:
                deltax = n.un1.x - n.x
                deltay = n.un1.y - n.y
                deltaz = n.un1.z - n.z
                links_params.append([(n.x, n.y, n.z), 
                                    (deltax, deltay, deltaz), 
                                    n.ul1.length, n.ul1.y, n.ul1.sr])
            if n.ul2 is not None:
                deltax = n.un2.x - n.x
                deltay = n.un2.y - n.y
                deltaz = n.un2.z - n.z
                links_params.append([(n.x, n.y, n.z), 
                                    (deltax, deltay, deltaz), 
                                    n.ul2.length, n.ul2.y, n.ul2.sr])
            if n.ul3 is not None:
                deltax = n.un3.x - n.x
                deltay = n.un3.y - n.y
                deltaz = n.un3.z - n.z
                links_params.append([(n.x, n.y, n.z), 
                                    (deltax, deltay, deltaz), 
                                    n.ul3.length, n.ul3.y, n.ul3.sr])
        path_links.append(links_params)
        path_ee.append(robot_nodes["N{}".format(n_nodes)])

    if visualize==False:
        return err_list, time
    
    # visualization
    # scene set up
    scene = v.canvas(title='End-point Placement', width=1280, height=720, up = v.vector(0,0,1),
                        center = v.vector(10,0,22), forward = v.vector(-1,0,-0.2), 
                        background=v.color.white)
    v.scene.lights.append(v.distant_light(direction=v.vector(1,0,1), color=v.color.gray(0.4)))
    v.scene.lights.append(v.distant_light(direction=v.vector(1,0,0), color=v.color.gray(0.4)))
    v.scene.lights.append(v.distant_light(direction=v.vector(0,0,-1), color=v.color.gray(0.4)))
    # base frame
    axis_x = v.arrow(pos=v.vector(0,0,0), axis=v.vector(5,0,0), shaftwidth=0.2, color=v.color.red)
    axis_y = v.arrow(pos=v.vector(0,0,0), axis=v.vector(0,5,0), shaftwidth=0.2, color=v.color.green)
    axis_z = v.arrow(pos=v.vector(0,0,0), axis=v.vector(0,0,5), shaftwidth=0.2, color=v.color.blue)
    # table
    table = v.cylinder(pos=v.vector(0,0,0), axis=v.vector(0,0,1), radius=6.5, length=0.1, opacity=0.5)
    links = []
    for id in range(len(path_links[0])):
        posx, posy, posz = path_links[0][id][0]
        axx, axy, axz = path_links[0][id][1]
        length = path_links[0][id][2]
        links.append(v.cylinder(pos=v.vector(posx, posy, posz), 
                axis=v.vector(axx, axy, axz), radius=0.3, color=v.vector(0.5,0.5,0.5), length = length))
    # ee path
    cx, cy, cz = path_ee[0]
    c = v.curve(pos=v.vector(cx, cy, cz), color= v.color.red, radius=0.2)
    c_cmu = v.curve(pos=v.vector(*target_path[0]), color= v.color.black, radius=0.1)
    for t in target_path:
        c_cmu.append(v.vector(*t))

    idx=0
    while True:
        v.rate(30)
        links_params_t = path_links[idx]
        for id in range(len(links_params_t)):
            posx, posy, posz = links_params_t[id][0]
            axx, axy, axz = links_params_t[id][1]
            length = links_params_t[id][2]
            links[id].pos = v.vector(posx, posy, posz)
            links[id].axis = v.vector(axx, axy, axz)
            links[id].length = length
            if links_params_t[id][4] == 0:
                links[id].color = v.vector(1, 1, 0.5)
                
        cx, cy, cz = path_ee[idx]
        c.append(v.vector(cx, cy, cz))
        idx += 1
        if idx == len(path_links):
            c.clear()
            idx=0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_propagation', type=bool, default=True, help='Use propagative computation for Jacobian.')
    parser.add_argument('--solver', type=str, default="fmin_bfgs", choices=["fmin_bfgs", "fmin_l_bfgs_b", "fmin_slsqp", "fmin_cg", "fmin_tnc"], help='Choose a solver.')
    parser.add_argument('--n_nodes', type=int, default=14, help='Num of nodes.')
    parser.add_argument('--freeze_num', type=int, default=0, help='Num of frozen edges.')
    parser.add_argument('--path_name', type=str, default="sin", choices=["sin", "circle", "polynomial", "cmu"], help='Name of the path.')
    parser.add_argument('--visualize', type=bool, default=True, help='Do visualization.')
    args = parser.parse_args()

    err_list, time_list = ee_path_follow(use_propagation=args.use_propagation, sol=args.solver, 
                                         n_nodes=args.n_nodes, freeze_num=args.freeze_num, 
                                         path_name=args.path_name, visualize=args.visualize)
    
import time
import random
import numpy as np
import vpython as v
import argparse

from robot_graph import RobotStructGraph
from utils import cal_ini_xyz
from params import *

n_nodes=7


def torus_pipe_crawl(use_propagation=True, sol="fmin_bfgs", freeze_nodes=0, num_steps=50, visualize=True):
    
    end_effector = ["N{}".format(x+1) for x in range(n_nodes)]
    rg = RobotStructGraph(num_nodes=n_nodes, task="torus_pipe_crawl", end_effector=end_effector)
    
    theta = np.pi/12
    p1,p2,p3 = cal_ini_xyz(theta)

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
        target_point = [None]*n_nodes
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

    # traj stack
    traj_links = []
    configs = []
    errs, times = [], []
    
    rg.input_vn(p1,p2,p3)
    # start freeze
    freeze_links = []
    for n in rg.node_list[3:]:
        if n.ul1 is not None:
            freeze_links.append(n.ul1)
        if n.ul2 is not None:
            freeze_links.append(n.ul2)
        if n.ul3 is not None:
            freeze_links.append(n.ul3)
    freeze_links = [freeze_links[5], *freeze_links[7:]]
    random.shuffle(freeze_links)
    for l in freeze_links[:freeze_nodes]:
        l.sr = 0.
    for idn, n in enumerate(rg.node_list[3:]):
        alignment, grad = n.forward(None, False, rg.task)
    err, time, robot_nodes = rg.update(ff=robot_forward, fb=robot_backward, solver=sol, use_propagation=use_propagation)
    configs.append(robot_nodes.copy())
    for iter in range(num_steps):
        theta += np.pi/72
        p1,p2,p3 = cal_ini_xyz(theta)
        rg.input_vn(p1,p2,p3)
        
        err, time, robot_nodes = rg.update(ff=robot_forward, fb=robot_backward, solver=sol, use_propagation=use_propagation)
        errs.append(err)
        times.append(time)
        configs.append(robot_nodes.copy())
        print("Climbing step #{}/{}, runtime: {:3f}, objective func value: {}".format(iter, num_steps, time, err))

    for iter in range(len(configs)):
        ro_x = np.array([coor[0] for coor in configs[iter].values()])
        ro_y = np.array([coor[1] for coor in configs[iter].values()])
        ro_z = np.array([coor[2] for coor in configs[iter].values()])

        links_params = []
        for idn in range(len(ro_x)):
            for skip in [1,2,3]:
                if idn < len(ro_x)-skip:
                    deltax = ro_x[idn+skip] - ro_x[idn]
                    deltay = ro_y[idn+skip] - ro_y[idn]
                    deltaz = ro_z[idn+skip] - ro_z[idn]
                    length = (deltax**2+deltay**2+deltaz**2)**0.5
                    # pos, axis, length
                    links_params.append(length)
        traj_links.append(links_params)
    x = np.std(np.array(traj_links), axis=0).tolist()
    freeze_link_id = []
    for i in range(len(x)):
        if x[i]<1e-7:
            freeze_link_id.append(i)
    
    if visualize==False:
        return errs, times
    
    traj_links=[]
    new_configs = []
    keys = list(configs[0].keys())[:-1]  # remove 'N7'
    for i in range(len(configs)-1):
        x = configs[i]
        new_configs.append(x.copy())
        for j in keys:
            x[j] = configs[i+1][j]
            new_configs.append(x.copy())
    new_configs.append(configs[-1])  # move one node at a time

    for iter in range(len(new_configs)):
        ro_x = np.array([coor[0] for coor in new_configs[iter].values()])
        ro_y = np.array([coor[1] for coor in new_configs[iter].values()])
        ro_z = np.array([coor[2] for coor in new_configs[iter].values()])

        links_params = []
        for idn in range(len(ro_x)):
            for skip in [1,2,3]:
                if idn < len(ro_x)-skip:
                    deltax = ro_x[idn+skip] - ro_x[idn]
                    deltay = ro_y[idn+skip] - ro_y[idn]
                    deltaz = ro_z[idn+skip] - ro_z[idn]
                    length = (deltax**2+deltay**2+deltaz**2)**0.5
                    # pos, axis, length
                    links_params.append([(ro_x[idn], ro_y[idn], ro_z[idn]), 
                                (deltax, deltay, deltaz), 
                                length])
        traj_links.append(links_params)
    
    # visualization
    # scene set up
    scene = v.canvas(title='Torus Pipe Crawl', width=1280, height=720, up = v.vector(0,0,1),
                        center = v.vector(0,0,0), forward = v.vector(0,-1,0), 
                        background=v.color.white)
    v.scene.lights.append(v.distant_light(direction=v.vector(0,1,0), color=v.color.gray(0.4)))

    pipe = v.ring(pos=v.vector(0,0,0),
                axis=v.vector(0,1,0),
                radius=R_, thickness=r_,
                opacity=0.2)
    
    links = []
    for id in range(len(traj_links[0])):
        posx, posy, posz = traj_links[0][id][0]
        axx, axy, axz = traj_links[0][id][1]
        length = traj_links[0][id][2]
        co = v.vector(0.1,0.1,0.1) if id in freeze_link_id else v.vector(0.8,0.8,0.8)
        links.append(v.cylinder(pos=v.vector(posx, posy, posz), 
                axis=v.vector(axx, axy, axz)
                , radius=0.3, color=co, length = length))
    idx=0
    while True:
        v.rate(30)
        links_params_t = traj_links[idx]
        for id in range(len(links_params_t)):
            posx, posy, posz = links_params_t[id][0]
            axx, axy, axz = links_params_t[id][1]
            length = links_params_t[id][2]
            links[id].pos = v.vector(posx, posy, posz)
            links[id].axis = v.vector(axx, axy, axz)
            links[id].length = length
        idx += 1
        if idx == len(traj_links):
            idx=0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_propagation', type=bool, default=True, help='Use propagative computation for Jacobian.')
    parser.add_argument('--solver', type=str, default="fmin_bfgs", choices=["fmin_bfgs", "fmin_l_bfgs_b", "fmin_slsqp", "fmin_cg", "fmin_tnc"], help='Choose a solver.')
    parser.add_argument('--freeze_num', type=int, default=0, help='Num of frozen edges.')
    parser.add_argument('--num_steps', type=int, default=100, help='Num of steps the robot crawl.')
    parser.add_argument('--visualize', type=bool, default=True, help='Do visualization.')
    args = parser.parse_args()

    errs, times = torus_pipe_crawl(use_propagation=args.use_propagation, sol=args.solver, freeze_nodes=args.freeze_num, num_steps=args.num_steps, visualize=args.visualize)
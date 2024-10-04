import numpy as np
import time
from scipy.optimize import fmin_bfgs, fmin_cg, fmin_l_bfgs_b, fmin_slsqp, fmin_tnc

from params import *
from utils import obj_ee, obj_straight_pipe, obj_torus_pipe

class Link:
    def __init__(self):
        self.w = 0.
        self.dl_dw = 0.
        
        self.sr = SR
        # self.minl = 7.9 + 0.2
        self.minl = MIN_L1
        self.midl = self.minl + 0.5*self.sr

        self.y = self.sigmoid(self.w)
        self.length = self.sr * self.y + self.minl
    
    def sigmoid(self, w):
        w = max(w, -100)
        w = min(w, 100)
        return 1. / (1 + np.exp(-w))
    
    def backward(self, dl_dr):
        self.dr_dy = self.sr
        self.dy_dw = self.y*(1-self.y)
        self.dl_dw = dl_dr*self.dr_dy*self.dy_dw
    
    def update(self, lr):
        self.w -= lr*self.dl_dw
        self.y = self.sigmoid(self.w)       
        self.length = self.sr * self.y + self.minl


class Node:
    def __init__(self, name="nomand"):
        self.dir_flag=0.
        self.name=name
        # x, y, z coordinates
        self.x = 0.
        self.y = 0.
        self.z = 0.
        # up stream links & nodes
        self.ul1, self.un1 = None, None
        self.ul2, self.un2 = None, None
        self.ul3, self.un3 = None, None

        # gradient
        self.dl_dx1, self.dl_dx2, self.dl_dx3 = 0., 0., 0.
        self.dl_dy1, self.dl_dy2, self.dl_dy3 = 0., 0., 0.
        self.dl_dz1, self.dl_dz2, self.dl_dz3 = 0., 0., 0.
    
    def input_vn(self,coor):
        x,y,z = coor
        self.x = x
        self.y = y
        self.z = z

    def forward(self, target, cal_alignment=False, task="end_traj_follow"):
        self.ul1.y = self.ul1.sigmoid(self.ul1.w)
        self.ul1.length = self.ul1.sr * self.ul1.y + self.ul1.minl
        self.ul2.y = self.ul2.sigmoid(self.ul2.w)
        self.ul2.length = self.ul2.sr * self.ul2.y + self.ul2.minl
        self.ul3.y = self.ul3.sigmoid(self.ul3.w)
        self.ul3.length = self.ul3.sr * self.ul3.y + self.ul3.minl
        # 
        self.a1 = self.ul1.length**2-(self.un1.x**2+self.un1.y**2+self.un1.z**2)
        self.a2 = self.ul2.length**2-(self.un2.x**2+self.un2.y**2+self.un2.z**2)
        self.a3 = self.ul3.length**2-(self.un3.x**2+self.un3.y**2+self.un3.z**2)
        # 
        self.x21 = self.un2.x-self.un1.x
        self.x31 = self.un3.x-self.un1.x
        self.y21 = self.un2.y-self.un1.y
        self.y31 = self.un3.y-self.un1.y
        self.z21 = self.un2.z-self.un1.z
        self.z31 = self.un3.z-self.un1.z
        self.a21 = -(self.a2-self.a1)/2.
        self.a31 = -(self.a3-self.a1)/2.
        # 
        self.d = self.x21*self.y31 - self.x31*self.y21
        if np.abs(self.d) < eps:
            self.d += eps
        self.b0 = (self.a21*self.y31 - self.a31*self.y21)/(self.d+eps)
        self.b1 = (self.z31*self.y21 - self.z21*self.y31)/(self.d+eps)
        self.c0 = (self.a31*self.x21 - self.a21*self.x31)/(self.d+eps)
        self.c1 = (self.x31*self.z21 - self.x21*self.z31)/(self.d+eps)
        # 
        self.e = self.b1**2 + self.c1**2 + 1
        self.f = self.b1*(self.b0-self.un1.x) + self.c1*(self.c0-self.un1.y)-self.un1.z
        self.g = (self.b0 - self.un1.x)**2 + (self.c0-self.un1.y)**2 + self.un1.z**2 - self.ul1.length**2
        # 
        self.m = self.f**2 - self.e*self.g
        self.m = max(self.m, 0.)

        self.n = self.m**0.5

        # positive or negative
        self.z = (-self.f+self.n)/self.e
        self.x = self.b0 + self.b1*self.z
        self.y = self.c0 + self.c1*self.z

        vec1 = np.array([self.x-self.un1.x, self.y-self.un1.y, self.z-self.un1.z])
        vec2 = np.array([self.un2.x-self.un1.x, self.un2.y-self.un1.y, self.un2.z-self.un1.z])
        vec3 = np.array([self.un3.x-self.un1.x, self.un3.y-self.un1.y, self.un3.z-self.un1.z])
        if np.cross(vec2, vec3)@vec1<0:
            self.dir_flag = -1.
            self.z = (-self.f-self.n)/self.e
            self.x = self.b0 + self.b1*self.z
            self.y = self.c0 + self.c1*self.z
        else:
            self.dir_flag = 1.

        if task=="end_traj_follow":
            func = obj_ee 
        elif task=="straight_pipe_crawl":
            func = obj_straight_pipe
        else:
            func = obj_torus_pipe
        if cal_alignment:
            alignment, grad = func([self.x, self.y, self.z], target)
        else:
            alignment, grad = 0, [0, 0, 0]
        # print(self.name, alignment)
        return alignment, grad

    def backward(self, dl_dx, dl_dy, dl_dz):
        # 
        self.da1_dr1, self.da1_dx1, self.da1_dy1, self.da1_dz1 = 2*self.ul1.length, -2*self.un1.x, -2*self.un1.y,  -2*self.un1.z
        self.da2_dr2, self.da2_dx2, self.da2_dy2, self.da2_dz2 = 2*self.ul2.length, -2*self.un2.x, -2*self.un2.y,  -2*self.un2.z
        self.da3_dr3, self.da3_dx3, self.da3_dy3, self.da3_dz3 = 2*self.ul3.length, -2*self.un3.x, -2*self.un3.y,  -2*self.un3.z
        # 
        self.dx21_dx2, self.dx21_dx1, self.dx31_dx3, self.dx31_dx1 = 1., -1., 1., -1.
        self.dy21_dy2, self.dy21_dy1, self.dy31_dy3, self.dy31_dy1 = 1., -1., 1., -1.
        self.dz21_dz2, self.dz21_dz1, self.dz31_dz3, self.dz31_dz1 = 1., -1., 1., -1.
        # 
        self.da21_da2, self.da21_da1 = -0.5, 0.5
        self.da31_da3, self.da31_da1 = -0.5, 0.5
        # 
        self.dd_dx21, self.dd_dy31, self.dd_dx31, self.dd_dy21 = self.y31, self.x21, -self.y21, -self.x31
        # 
        self.db0_da21, self.db0_dy31, self.db0_da31, self.db0_dy21 = self.y31/self.d, self.a21/self.d, -self.y21/self.d, -self.a31/self.d
        self.db0_dd = -(self.a21*self.y31 - self.a31*self.y21)/(self.d**2)

        self.db1_dy21, self.db1_dz31, self.db1_dy31, self.db1_dz21 = self.z31/self.d, self.y21/self.d, -self.z21/self.d, -self.y31/self.d
        self.db1_dd = -(self.z31*self.y21 - self.z21*self.y31)/(self.d**2)
        
        self.dc0_da31, self.dc0_dx21, self.dc0_da21, self.dc0_dx31 = self.x21/self.d, self.a31/self.d, -self.x31/self.d, -self.a21/self.d
        self.dc0_dd = -(self.a31*self.x21 - self.a21*self.x31)/(self.d**2)
        
        self.dc1_dx31, self.dc1_dz21, self.dc1_dx21, self.dc1_dz31 = self.z21/self.d, self.x31/self.d, -self.z31/self.d, -self.x21/self.d
        self.dc1_dd = -(self.x31*self.z21 - self.x21*self.z31)/(self.d**2)
        # 
        self.de_db1, self.de_dc1 = 2*self.b1, 2*self.c1
        self.df_db0, self.df_db1, self.df_dc0, self.df_dc1 = self.b1, self.b0-self.un1.x, self.c1, self.c0-self.un1.y
        self.dg_db0, self.dg_dc0 = 2*(self.b0-self.un1.x), 2*(self.c0-self.un1.y)
        # 
        self.dm_df, self.dm_de, self.dm_dg = 2*self.f, -self.g, -self.e
        self.dn_dm = 0.5*(self.m+1e-8)**(-0.5)
        # 
        assert self.dir_flag==1. or self.dir_flag==-1., "Wrong direction flag {}".format(self.dir_flag)
        self.dz_de = -(1./self.e**2+1e-8)*(-self.f+self.dir_flag*(self.f**2-self.e*self.g)**0.5) + self.dir_flag*(1./self.e+1e-8)*(-0.5*self.g/(self.f**2-self.e*self.g)**0.5+1e-8)
        self.dz_df = -1./(self.e + 1e-8) + self.dir_flag*self.f/(self.e*(self.f**2-self.e*self.g)**0.5+1e-8)
        self.dz_dg = self.dir_flag*(-0.5*(1./(self.f**2-self.e*self.g)**0.5 + 1e-8))
        # 
        self.dx_db0, self.dx_db1, self.dx_dz = 1, self.z, self.b1
        self.dy_dc0, self.dy_dc1, self.dy_dz = 1, self.z, self.c1
        
        # -------------------------------------------------------------------------------------------------------------------------------
        
        self.da21_dx1, self.da21_dx2 = self.da21_da1*self.da1_dx1, self.da21_da2*self.da2_dx2
        self.da21_dy1, self.da21_dy2 = self.da21_da1*self.da1_dy1, self.da21_da2*self.da2_dy2
        self.da21_dz1, self.da21_dz2 = self.da21_da1*self.da1_dz1, self.da21_da2*self.da2_dz2
        self.da21_dr1, self.da21_dr2 = self.da21_da1*self.da1_dr1, self.da21_da2*self.da2_dr2
        # 
        self.da31_dx1, self.da31_dx3 = self.da31_da1*self.da1_dx1, self.da31_da3*self.da3_dx3
        self.da31_dy1, self.da31_dy3 = self.da31_da1*self.da1_dy1, self.da31_da3*self.da3_dy3
        self.da31_dz1, self.da31_dz3 = self.da31_da1*self.da1_dz1, self.da31_da3*self.da3_dz3
        self.da31_dr1, self.da31_dr3 = self.da31_da1*self.da1_dr1, self.da31_da3*self.da3_dr3
        # 
        self.dd_dx1, self.dd_dx2, self.dd_dx3 = self.dd_dx21*self.dx21_dx1 + self.dd_dx31*self.dx31_dx1, self.dd_dx21*self.dx21_dx2, self.dd_dx31*self.dx31_dx3
        self.dd_dy1, self.dd_dy2, self.dd_dy3 = self.dd_dy31*self.dy31_dy1 + self.dd_dy21*self.dy21_dy1, self.dd_dy21*self.dy21_dy2, self.dd_dy31*self.dy31_dy3
        # 
        self.db0_dx1, self.db0_dx2, self.db0_dx3 = self.db0_da21*self.da21_dx1 + self.db0_da31*self.da31_dx1 + self.db0_dd*self.dd_dx1 \
                                                , self.db0_da21*self.da21_dx2 + self.db0_dd*self.dd_dx2, self.db0_da31*self.da31_dx3 + self.db0_dd*self.dd_dx3
        self.db0_dy1 = self.db0_da21*self.da21_dy1 + self.db0_dy31*self.dy31_dy1 + self.db0_da31*self.da31_dy1 + self.db0_dy21*self.dy21_dy1 + self.db0_dd*self.dd_dy1
        self.db0_dy2 = self.db0_da21*self.da21_dy2 + self.db0_dy21*self.dy21_dy2 + self.db0_dd*self.dd_dy2
        self.db0_dy3 = self.db0_dy31*self.dy31_dy3 + self.db0_da31*self.da31_dy3 + self.db0_dd*self.dd_dy3
        self.db0_dz1, self.db0_dz2, self.db0_dz3 = self.db0_da21*self.da21_dz1 + self.db0_da31*self.da31_dz1, self.db0_da21*self.da21_dz2, self.db0_da31*self.da31_dz3
        self.db0_dr1, self.db0_dr2, self.db0_dr3 = self.db0_da21*self.da21_dr1 + self.db0_da31*self.da31_dr1, self.db0_da21*self.da21_dr2, self.db0_da31*self.da31_dr3

        self.db1_dx1, self.db1_dx2, self.db1_dx3 = self.db1_dd*self.dd_dx1, self.db1_dd*self.dd_dx2, self.db1_dd*self.dd_dx3
        self.db1_dy1, self.db1_dy2, self.db1_dy3 = self.db1_dy21*self.dy21_dy1 + self.db1_dy31*self.dy31_dy1 + self.db1_dd*self.dd_dy1, \
                                                    self.db1_dy21*self.dy21_dy2 + self.db1_dd*self.dd_dy2, self.db1_dy31*self.dy31_dy3 + self.db1_dd*self.dd_dy3
        self.db1_dz1, self.db1_dz2, self.db1_dz3 = self.db1_dz31*self.dz31_dz1 + self.db1_dz21*self.dz21_dz1, \
                                                    self.db1_dz21*self.dz21_dz2, self.db1_dz31*self.dz31_dz3
        
        self.dc0_dx1 = self.dc0_da31*self.da31_dx1 + self.dc0_dx21*self.dx21_dx1 + self.dc0_da21*self.da21_dx1 + self.dc0_dx31*self.dx31_dx1 + self.dc0_dd*self.dd_dx1
        self.dc0_dx2 = self.dc0_dx21*self.dx21_dx2 + self.dc0_da21*self.da21_dx2 + self.dc0_dd*self.dd_dx2
        self.dc0_dx3 = self.dc0_da31*self.da31_dx3 + self.dc0_dx31*self.dx31_dx3 + self.dc0_dd*self.dd_dx3
        self.dc0_dy1 = self.dc0_da31*self.da31_dy1 + self.dc0_da21*self.da21_dy1 + self.dc0_dd*self.dd_dy1
        self.dc0_dy2 = self.dc0_da21*self.da21_dy2 + self.dc0_dd*self.dd_dy2
        self.dc0_dy3 = self.dc0_da31*self.da31_dy3 + self.dc0_dd*self.dd_dy3
        self.dc0_dz1, self.dc0_dz2, self.dc0_dz3 = self.dc0_da31*self.da31_dz1 + self.dc0_da21*self.da21_dz1, self.dc0_da21*self.da21_dz2, self.dc0_da31*self.da31_dz3
        self.dc0_dr1, self.dc0_dr2, self.dc0_dr3 = self.dc0_da31*self.da31_dr1 + self.dc0_da21*self.da21_dr1, self.dc0_da21*self.da21_dr2, self.dc0_da31*self.da31_dr3

        self.dc1_dx1, self.dc1_dx2, self.dc1_dx3 = self.dc1_dx31*self.dx31_dx1 + self.dc1_dx21*self.dx21_dx1 + self.dc1_dd*self.dd_dx1, \
                                                    self.dc1_dx21*self.dx21_dx2 + self.dc1_dd*self.dd_dx2, self.dc1_dx31*self.dx31_dx3 + self.dc1_dd*self.dd_dx3
        self.dc1_dy1, self.dc1_dy2, self.dc1_dy3 = self.dc1_dd*self.dd_dy1, self.dc1_dd*self.dd_dy2, self.dc1_dd*self.dd_dy3
        self.dc1_dz1, self.dc1_dz2, self.dc1_dz3 = self.dc1_dz21*self.dz21_dz1 + self.dc1_dz31*self.dz31_dz1, \
                                                    self.dc1_dz21*self.dz21_dz2, self.dc1_dz31*self.dz31_dz3
        #
        self.de_dx1, self.de_dx2, self.de_dx3 = self.de_db1*self.db1_dx1 + self.de_dc1*self.dc1_dx1, self.de_db1*self.db1_dx2 + self.de_dc1*self.dc1_dx2, \
                                                self.de_db1*self.db1_dx3 + self.de_dc1*self.dc1_dx3
        self.de_dy1, self.de_dy2, self.de_dy3 = self.de_db1*self.db1_dy1 + self.de_dc1*self.dc1_dy1, self.de_db1*self.db1_dy2 + self.de_dc1*self.dc1_dy2, \
                                                self.de_db1*self.db1_dy3 + self.de_dc1*self.dc1_dy3
        self.de_dz1, self.de_dz2, self.de_dz3 = self.de_db1*self.db1_dz1 + self.de_dc1*self.dc1_dz1, self.de_db1*self.db1_dz2 + self.de_dc1*self.dc1_dz2, \
                                                self.de_db1*self.db1_dz3 + self.de_dc1*self.dc1_dz3
        
        self.df_dx1 = self.df_db1*self.db1_dx1 + self.df_db0*self.db0_dx1 + (-self.b1) + self.df_dc1*self.dc1_dx1 + self.df_dc0*self.dc0_dx1
        self.df_dx2 = self.df_db1*self.db1_dx2 + self.df_db0*self.db0_dx2 + self.df_dc1*self.dc1_dx2 + self.df_dc0*self.dc0_dx2
        self.df_dx3 = self.df_db1*self.db1_dx3 + self.df_db0*self.db0_dx3 + self.df_dc1*self.dc1_dx3 + self.df_dc0*self.dc0_dx3
        self.df_dy1 = self.df_db1*self.db1_dy1 + self.df_db0*self.db0_dy1 + self.df_dc1*self.dc1_dy1 + self.df_dc0*self.dc0_dy1 + (-self.c1)
        self.df_dy2 = self.df_db1*self.db1_dy2 + self.df_db0*self.db0_dy2 + self.df_dc1*self.dc1_dy2 + self.df_dc0*self.dc0_dy2
        self.df_dy3 = self.df_db1*self.db1_dy3 + self.df_db0*self.db0_dy3 + self.df_dc1*self.dc1_dy3 + self.df_dc0*self.dc0_dy3
        self.df_dz1 = self.df_db1*self.db1_dz1 + self.df_db0*self.db0_dz1 + self.df_dc1*self.dc1_dz1 + self.df_dc0*self.dc0_dz1 + (-1)
        self.df_dz2 = self.df_db1*self.db1_dz2 + self.df_db0*self.db0_dz2 + self.df_dc1*self.dc1_dz2 + self.df_dc0*self.dc0_dz2
        self.df_dz3 = self.df_db1*self.db1_dz3 + self.df_db0*self.db0_dz3 + self.df_dc1*self.dc1_dz3 + self.df_dc0*self.dc0_dz3
        self.df_dr1, self.df_dr2, self.df_dr3 = self.df_db0*self.db0_dr1 + self.df_dc0*self.dc0_dr1, self.df_db0*self.db0_dr2 + self.df_dc0*self.dc0_dr2, \
                                                self.df_db0*self.db0_dr3 + self.df_dc0*self.dc0_dr3
        
        self.dg_dx1, self.dg_dx2, self.dg_dx3 = self.dg_db0*self.db0_dx1 + (-2*(self.b0-self.un1.x)) + self.dg_dc0*self.dc0_dx1, \
                                                self.dg_db0*self.db0_dx2 + self.dg_dc0*self.dc0_dx2, self.dg_db0*self.db0_dx3 + self.dg_dc0*self.dc0_dx3
        self.dg_dy1, self.dg_dy2, self.dg_dy3 = self.dg_db0*self.db0_dy1 + self.dg_dc0*self.dc0_dy1 + (-2*(self.c0-self.un1.y)), \
                                                self.dg_db0*self.db0_dy2 + self.dg_dc0*self.dc0_dy2, self.dg_db0*self.db0_dy3 + self.dg_dc0*self.dc0_dy3
        self.dg_dz1, self.dg_dz2, self.dg_dz3 = self.dg_db0*self.db0_dz1 + self.dg_dc0*self.dc0_dz1 + (2*self.un1.z), \
                                                self.dg_db0*self.db0_dz2 + self.dg_dc0*self.dc0_dz2, self.dg_db0*self.db0_dz3 + self.dg_dc0*self.dc0_dz3
        self.dg_dr1, self.dg_dr2, self.dg_dr3 = self.dg_db0*self.db0_dr1 + self.dg_dc0*self.dc0_dr1 + (-2*self.ul1.length), \
                                                self.dg_db0*self.db0_dr2 + self.dg_dc0*self.dc0_dr2, self.dg_db0*self.db0_dr3 + self.dg_dc0*self.dc0_dr3
        # 
        self.dz_dx1, self.dz_dx2, self.dz_dx3 = self.dz_de*self.de_dx1 + self.dz_df*self.df_dx1 + self.dz_dg*self.dg_dx1, \
                                                self.dz_de*self.de_dx2 + self.dz_df*self.df_dx2 + self.dz_dg*self.dg_dx2, \
                                                self.dz_de*self.de_dx3 + self.dz_df*self.df_dx3 + self.dz_dg*self.dg_dx3
        self.dz_dy1, self.dz_dy2, self.dz_dy3 = self.dz_de*self.de_dy1 + self.dz_df*self.df_dy1 + self.dz_dg*self.dg_dy1, \
                                                self.dz_de*self.de_dy2 + self.dz_df*self.df_dy2 + self.dz_dg*self.dg_dy2, \
                                                self.dz_de*self.de_dy3 + self.dz_df*self.df_dy3 + self.dz_dg*self.dg_dy3
        self.dz_dz1, self.dz_dz2, self.dz_dz3 = self.dz_de*self.de_dz1 + self.dz_df*self.df_dz1 + self.dz_dg*self.dg_dz1, \
                                                self.dz_de*self.de_dz2 + self.dz_df*self.df_dz2 + self.dz_dg*self.dg_dz2, \
                                                self.dz_de*self.de_dz3 + self.dz_df*self.df_dz3 + self.dz_dg*self.dg_dz3
        self.dz_dr1, self.dz_dr2, self.dz_dr3 = self.dz_df*self.df_dr1 + self.dz_dg*self.dg_dr1, \
                                                self.dz_df*self.df_dr2 + self.dz_dg*self.dg_dr2, \
                                                self.dz_df*self.df_dr3 + self.dz_dg*self.dg_dr3

        self.dy_dx1, self.dy_dx2, self.dy_dx3 = self.dy_dc0*self.dc0_dx1 + self.dy_dc1*self.dc1_dx1 + self.dy_dz*self.dz_dx1, \
                                                self.dy_dc0*self.dc0_dx2 + self.dy_dc1*self.dc1_dx2 + self.dy_dz*self.dz_dx2, \
                                                self.dy_dc0*self.dc0_dx3 + self.dy_dc1*self.dc1_dx3 + self.dy_dz*self.dz_dx3
        self.dy_dy1, self.dy_dy2, self.dy_dy3 = self.dy_dc0*self.dc0_dy1 + self.dy_dc1*self.dc1_dy1 + self.dy_dz*self.dz_dy1, \
                                                self.dy_dc0*self.dc0_dy2 + self.dy_dc1*self.dc1_dy2 + self.dy_dz*self.dz_dy2, \
                                                self.dy_dc0*self.dc0_dy3 + self.dy_dc1*self.dc1_dy3 + self.dy_dz*self.dz_dy3
        self.dy_dz1, self.dy_dz2, self.dy_dz3 = self.dy_dc0*self.dc0_dz1 + self.dy_dc1*self.dc1_dz1 + self.dy_dz*self.dz_dz1, \
                                                self.dy_dc0*self.dc0_dz2 + self.dy_dc1*self.dc1_dz2 + self.dy_dz*self.dz_dz2, \
                                                self.dy_dc0*self.dc0_dz3 + self.dy_dc1*self.dc1_dz3 + self.dy_dz*self.dz_dz3
        self.dy_dr1, self.dy_dr2, self.dy_dr3 = self.dy_dc0*self.dc0_dr1 + self.dy_dz*self.dz_dr1, \
                                                self.dy_dc0*self.dc0_dr2 + self.dy_dz*self.dz_dr2, \
                                                self.dy_dc0*self.dc0_dr3 + self.dy_dz*self.dz_dr3
        
        self.dx_dx1, self.dx_dx2, self.dx_dx3 = self.dx_db0*self.db0_dx1 + self.dx_db1*self.db1_dx1 + self.dx_dz*self.dz_dx1, \
                                                self.dx_db0*self.db0_dx2 + self.dx_db1*self.db1_dx2 + self.dx_dz*self.dz_dx2, \
                                                self.dx_db0*self.db0_dx3 + self.dx_db1*self.db1_dx3 + self.dx_dz*self.dz_dx3
        self.dx_dy1, self.dx_dy2, self.dx_dy3 = self.dx_db0*self.db0_dy1 + self.dx_db1*self.db1_dy1 + self.dx_dz*self.dz_dy1, \
                                                self.dx_db0*self.db0_dy2 + self.dx_db1*self.db1_dy2 + self.dx_dz*self.dz_dy2, \
                                                self.dx_db0*self.db0_dy3 + self.dx_db1*self.db1_dy3 + self.dx_dz*self.dz_dy3
        self.dx_dz1, self.dx_dz2, self.dx_dz3 = self.dx_db0*self.db0_dz1 + self.dx_db1*self.db1_dz1 + self.dx_dz*self.dz_dz1, \
                                                self.dx_db0*self.db0_dz2 + self.dx_db1*self.db1_dz2 + self.dx_dz*self.dz_dz2, \
                                                self.dx_db0*self.db0_dz3 + self.dx_db1*self.db1_dz3 + self.dx_dz*self.dz_dz3
        self.dx_dr1, self.dx_dr2, self.dx_dr3 = self.dx_db0*self.db0_dr1 + self.dx_dz*self.dz_dr1, \
                                                self.dx_db0*self.db0_dr2 + self.dx_dz*self.dz_dr2, \
                                                self.dx_db0*self.db0_dr3 + self.dx_dz*self.dz_dr3
        # 
        self.dl_dx1 = dl_dx*self.dx_dx1 + dl_dy*self.dy_dx1 + dl_dz*self.dz_dx1
        self.dl_dx2 = dl_dx*self.dx_dx2 + dl_dy*self.dy_dx2 + dl_dz*self.dz_dx2
        self.dl_dx3 = dl_dx*self.dx_dx3 + dl_dy*self.dy_dx3 + dl_dz*self.dz_dx3

        self.dl_dy1 = dl_dx*self.dx_dy1 + dl_dy*self.dy_dy1 + dl_dz*self.dz_dy1
        self.dl_dy2 = dl_dx*self.dx_dy2 + dl_dy*self.dy_dy2 + dl_dz*self.dz_dy2
        self.dl_dy3 = dl_dx*self.dx_dy3 + dl_dy*self.dy_dy3 + dl_dz*self.dz_dy3

        self.dl_dz1 = dl_dx*self.dx_dz1 + dl_dy*self.dy_dz1 + dl_dz*self.dz_dz1
        self.dl_dz2 = dl_dx*self.dx_dz2 + dl_dy*self.dy_dz2 + dl_dz*self.dz_dz2
        self.dl_dz3 = dl_dx*self.dx_dz3 + dl_dy*self.dy_dz3 + dl_dz*self.dz_dz3

        self.dl_dr1 = dl_dx*self.dx_dr1 + dl_dy*self.dy_dr1 + dl_dz*self.dz_dr1
        self.dl_dr2 = dl_dx*self.dx_dr2 + dl_dy*self.dy_dr2 + dl_dz*self.dz_dr2
        self.dl_dr3 = dl_dx*self.dx_dr3 + dl_dy*self.dy_dr3 + dl_dz*self.dz_dr3

        self.ul1.backward(self.dl_dr1)
        self.ul2.backward(self.dl_dr2)
        self.ul3.backward(self.dl_dr3)
    
    def update(self, lr):
        self.ul1.update(lr)
        self.ul2.update(lr)
        self.ul3.update(lr)

class RobotStructGraph:
    def __init__(self, num_nodes=7, task="end_traj_follow", end_effector=["N7"]):
        self.num_nodes = num_nodes
        self.task = task
        self.end_effector = end_effector
        # print(self.end_effector)
        # set node
        self.node_list = []
        for _ in range(3):
            self.node_list.append(Node("V{}".format(_+1)))
        for _ in range(self.num_nodes):
            self.node_list.append(Node("N{}".format(_+1)))
        for i, node in enumerate(self.node_list[3:]):
            node.un1, node.un2, node.un3 = self.node_list[i], self.node_list[i+1], self.node_list[i+2]
            node.ul1, node.ul2, node.ul3 = Link(), Link(), Link()
        
        if self.task=="end_traj_follow":
            pass
            for n in self.node_list[3:]:
                n.ul1.min_length = 8
                n.ul1.sr = 2
                n.ul2.min_length = 8
                n.ul2.sr = 2
                n.ul3.min_length = 8
                n.ul3.sr = 2
        else:
            "special link: self.N3.ul2, self.N5.ul1, self.N6.ul2"
            for n in self.node_list:
                if n.name=="N3" or n.name=="N6" or n.name=="N7":
                    n.ul2.minl = MIN_L2
                if n.name=="N5":
                    n.ul1.minl = MIN_L2

        self.rev_node_list = self.node_list.copy()
        self.rev_node_list.reverse()
        self.align, self.grads = [], []
    
    def input_vn(self, coor1, coor2, coor3):
        # initialize virtual nodes
        self.node_list[0].input_vn(coor1)
        self.node_list[1].input_vn(coor2)
        self.node_list[2].input_vn(coor3)
    
    def forward(self, target):
        self.align, self.grads = [], []
        for idn, n in enumerate(self.node_list[3:]):
            cal_alignment = True if n.name in self.end_effector else False
            alignment, grad = n.forward(target[idn], cal_alignment, self.task)
            self.align.append(alignment)
            self.grads.append(grad)
        if self.task=="end_traj_follow":
            return np.max(np.abs(np.array(self.align)))
        else:
            return np.mean(np.abs(np.array(self.align)))
    
    def backward(self):
        for idn, n in enumerate(self.rev_node_list[:-3]):
            # local distance error
            dl_dx, dl_dy, dl_dz = self.grads[idn]
            if idn-1>=0:
                dl_dx += self.rev_node_list[idn-1].dl_dx3
                dl_dy += self.rev_node_list[idn-1].dl_dy3
                dl_dz += self.rev_node_list[idn-1].dl_dz3
            if idn-2>=0:
                dl_dx += self.rev_node_list[idn-2].dl_dx2
                dl_dy += self.rev_node_list[idn-2].dl_dy2
                dl_dz += self.rev_node_list[idn-2].dl_dz2
            if idn-3>=0:
                dl_dx += self.rev_node_list[idn-3].dl_dx1
                dl_dy += self.rev_node_list[idn-3].dl_dy1
                dl_dz += self.rev_node_list[idn-3].dl_dz1
            n.backward(dl_dx, dl_dy, dl_dz)
            
    # def update(self, base_lr, max_iter=1000, 
    #            target=[-20., -20., 40.], tol=1e-2, early_break=True):
        
    #     g_nodes = {}
    #     alignments = {}
    #     it = 0
    #     alignment = 10
    #     while it < max_iter:
    #         it += 1
            
    #         max_align = self.forward(target)
    #         # print("alignment: ", align)
    #         # quit()
    #         if max_align <= tol and early_break:
    #             print("break!!!")
    #             break
            
    #         self.grads.reverse()
    #         self.backward()
            
    #         lr = base_lr# * (1/1+0.0001*np.sqrt(it))
    #         for idn, n in enumerate(self.node_list[3:]):
    #             n.update(lr)
    #     if self.task=="straight_pipe_crawl" or self.task=="torus_pipe_crawl":
    #         out_list = self.node_list[3:]
    #     else:
    #         out_list = self.node_list

    #     for idn, n in enumerate(out_list):
    #         g_nodes[n.name] = [n.x, n.y, n.z]
    #         # print(n.name, ": ", alignments[n.name], "--", n.ul1.length, n.ul2.length, n.ul3.length)
        
    #     print(self.align)
    #     return g_nodes, np.linalg.norm(np.array(self.align))
    
    def update(self, ff, fb, solver="fmin_bfgs", use_propagation=True):
        
        self.w_list = []
        g_nodes = {}
        
        for idn, n in enumerate(self.node_list[3:]):
            self.w_list.append(n.ul1.w)
            self.w_list.append(n.ul2.w)
            self.w_list.append(n.ul3.w)
        
        T1 = time.time()
        
        if use_propagation:
            if solver=="fmin_bfgs":
                x = fmin_bfgs(ff, self.w_list, fprime=fb, maxiter=10000, disp=0)
            elif solver=="fmin_l_bfgs_b":
                x = fmin_l_bfgs_b(ff, self.w_list, fprime=fb, maxiter=10000, disp=0)
                x = x[0]
            elif solver=="fmin_slsqp":
                x = fmin_slsqp(ff, self.w_list, fprime=fb, disp=0)
            elif solver=="fmin_cg":
                x = fmin_cg(ff, self.w_list, fprime=fb, maxiter=10000, disp=0, gtol=1e-8)
            elif solver=="fmin_tnc":
                x, _, __ = fmin_tnc(ff, self.w_list, fprime=fb, disp=0, pgtol=1e-8)
            else:
                raise ValueError
        else:
            if solver=="fmin_bfgs":
                x = fmin_bfgs(ff, self.w_list, maxiter=10000, disp=0)
            elif solver=="fmin_l_bfgs_b":
                x = fmin_l_bfgs_b(ff, self.w_list, maxiter=10000, disp=0, approx_grad=True)
                x = x[0]
            elif solver=="fmin_slsqp":
                x = fmin_slsqp(ff, self.w_list, disp=0)
            elif solver=="fmin_cg":
                x = fmin_cg(ff, self.w_list, maxiter=10000, disp=0)
            elif solver=="fmin_tnc":
                x, _, __ = fmin_tnc(ff, self.w_list, disp=0, approx_grad=True)
            else:
                raise ValueError

        T2 = time.time()
        if self.task=="straight_pipe_crawl" or self.task=="torus_pipe_crawl":
            out_list = self.node_list[3:]
        else:
            out_list = self.node_list

        for idn, n in enumerate(out_list):
            g_nodes[n.name] = [n.x, n.y, n.z]
        
        # print("{:20}time: {:.3f}".format(solver, T2-T1), "   err: {}".format(self.align))
        if self.task=="end_traj_follow":
            err = ff(x)
            return err, T2-T1, g_nodes
        else:
            return self.align, T2-T1, g_nodes

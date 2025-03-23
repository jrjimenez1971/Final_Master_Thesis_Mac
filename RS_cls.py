# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 16:02:52 2022

@author: jufen
"""

import numpy as np
import os
import time


class RandomSearch(object):
    """ Random search algorithm for wind farm layout optimization. Changing 
    turbine locations (x_i, y_i) for num_dim=2 or turbine locations and mooring
    line configurations (x_i, y_i, alpha_i) for num_dim=3.

    Referece of the algorithm:
        Feng, J. and Shen, W.Z., 2015. Solving the wind farm layout 
        optimization problem using random search algorithm. Renewable Energy, 
        78, pp.182-192. [https://doi.org/10.1016/j.renene.2015.01.005]
    """
    def __init__(self, fun_obj, 
                 lower_bounds, upper_bounds, fun_constr=None, num_dim=3,
                 max_move_step=None, save_path=None):
        self.fun_obj = fun_obj
        self.fun_constr = fun_constr
        self.lower_bounds = lower_bounds # [x_min, y_min] or [x_min, y_min, 
                                         # alpha_min] for num_dim=2 or 3 
        self.upper_bounds = upper_bounds
        self.num_dim=num_dim
        
        if max_move_step is None:
            max_move_step = np.sqrt((upper_bounds[0] - lower_bounds[0])**2 +
                                    (upper_bounds[1] - lower_bounds[1])**2)
        self.max_move_step = max_move_step
        if save_path is None:
            save_path = './'
        
        self.make_save_path(save_path)
        

    def make_save_path(self, save_path):
        self.save_path = save_path

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)


    def randomly_move_a_turbine(self, x_i, y_i, i_move):
        in_bounds = False
        
        while not in_bounds:
            move_dir = np.random.rand() * 2 * np.pi
            move_step = np.random.rand() * self.max_move_step
            new_x = x_i[i_move] + move_step * np.cos(move_dir)
            new_y = y_i[i_move] + move_step * np.sin(move_dir)
            
            in_bounds = (
                new_x <= self.upper_bounds[0] and new_x >= self.lower_bounds[0]
            and new_y <= self.upper_bounds[1] and new_y >= self.lower_bounds[1])
        x_i_new = x_i + 0
        y_i_new = y_i + 0
        x_i_new[i_move] = new_x
        y_i_new[i_move] = new_y
        return x_i_new,  y_i_new


    def randomly_change_a_configuration(self, alpha_i, i_move):
        right_prob = np.random.rand()
        if right_prob > 0.5:
            alpha_new = alpha_i[i_move] + (
                self.upper_bounds[2] - alpha_i[i_move]) * np.random.rand()
        else:
            alpha_new = alpha_i[i_move] - (
                alpha_i[i_move] - self.lower_bounds[2]) * np.random.rand()
                
        alpha_i_new = alpha_i + 0
        alpha_i_new[i_move] = alpha_new
        return alpha_i_new


    def generate_feasible_new_design(self, x, max_tries=1000,
                                     change_same_wt=True):
        i_try = 0

        if self.num_dim == 3:
            num_wt = len(x)//3
            x_i = x[:num_wt]
            y_i = x[num_wt:num_wt*2]
            alpha_i = x[num_wt*2:num_wt*3]
        
            vio_constr = True
            while i_try < max_tries and vio_constr:
                i_move = np.random.randint(num_wt)
                x_i_new, y_i_new = self.randomly_move_a_turbine(x_i, y_i, i_move)

                if change_same_wt:
                    i_change = i_move
                else:
                    i_change = np.random.randint(num_wt)
                alpha_i_new = self.randomly_change_a_configuration(alpha_i, i_change)
                
                x_new = np.hstack((x_i_new, y_i_new, alpha_i_new))
                if self.fun_constr is None:
                    vio_constr = False
                else:
                    vio_constr = self.fun_constr(x_new) > 0

        elif self.num_dim == 2:
            num_wt = len(x)//2
            x_i = x[:num_wt]
            y_i = x[num_wt:num_wt*2]
        
            vio_constr = True
            while i_try < max_tries and vio_constr:
                i_move = np.random.randint(num_wt)
                x_i_new, y_i_new = self.randomly_move_a_turbine(x_i, y_i, i_move)
                
                x_new = np.hstack((x_i_new, y_i_new))
                if self.fun_constr is None:
                    vio_constr = False
                else:
                    vio_constr = self.fun_constr(x_new) > 0
    
        if vio_constr:
            return x
        else:
            return x_new


    def rs_run(self, x0, num_iter=100, max_move_step=None, change_same_wt=True,
               disp_every_step=True):
        if max_move_step is not None:
            self.max_move_step = max_move_step
        start = time.time()
        x_current = x0 + 0
        i = 0
        obj_current = self.fun_obj(x_current)
        hist = [np.hstack((i, x_current, obj_current))]
    
        while i < num_iter:
            x_new = self.generate_feasible_new_design(x_current,
                                                      change_same_wt)
            obj_new = self.fun_obj(x_new)
            i = i + 1
            
            if obj_new < obj_current:
                x_current = x_new
                obj_current = obj_new
                hist.append(np.hstack((i, x_current, obj_current)))
                np.savetxt(os.path.join(
                    self.save_path, 'hist_rs_current.txt'), hist)
                if not disp_every_step:
                    print(f'{i} / {num_iter}, obj={obj_current};')
            if disp_every_step:        
                print(f'{i} / {num_iter}, obj={obj_current};')

        cpu_time = time.time() - start
        np.savetxt(os.path.join(
            self.save_path, 'hist_rs.txt'), hist)
        np.savetxt(os.path.join(
            self.save_path, 'best_solution_rs.txt'),
            np.hstack((x_current, obj_current)))
        
        with open(os.path.join(self.save_path, 'general_info.txt'), 'w') as f:
            f.write('max_iteration: {0}\n'.format(num_iter)  + 
                    'max_move_step: {0}\n'.format(self.max_move_step) +
                    'cpu_time: {0}\n'.format(cpu_time))

        return x_current, obj_current, hist
        
        
        
        
        
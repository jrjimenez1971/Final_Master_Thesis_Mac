import numpy as np
import os

class PSO(object):
    """
    Particle swarm optimization algorithm for continuous optimization problems.
    Assume constraints are included in the objective function using static 
    penalty terms.

    Reference of the algorithm and default parameters:
        Chowdhury, S., Zhang, J., Messac, A. and Castillo, L., 2012. 
        Unrestricted wind farm layout optimization (UWFLO): Investigating key 
        factors influencing the maximum power generation. Renewable Energy, 
        38(1), pp.16-30. [https://doi.org/10.1016/j.renene.2011.06.045]
    """
    
    def __init__(self, obj_fun, num_design_variables, 
                 lower_bounds, upper_bounds, 
                 pop_size=None, max_gen=10,
                 alpha=0.5, beta_l=1.4, beta_g=1.4,
                 save_path=None):
        self.obj_fun = obj_fun
        self.num_design_variables = num_design_variables
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        if pop_size is None:
            self.pop_size = 5 * self.num_design_variables
        else:
            self.pop_size = pop_size
        self.max_gen = max_gen
        self.i_gen = 0
            
        self.alpha = alpha
        self.beta_l = beta_l
        self.beta_g = beta_g
        self.hist = []
        
        if save_path is None:
            save_path = './'
        self.save_path = save_path
        
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        
        
    def initialize_pop(self, feedin_solutions=None):
        """ Initialize the population with random solutions. feedin_solutions 
        can pass in specified initial solutions (as a list of solutions).
        """
        self.current_pop = np.zeros((self.pop_size, self.num_design_variables+1))
        
        for i_variable in range(self.num_design_variables):
            self.current_pop[:, i_variable] = (
                    (self.lower_bounds[i_variable] +
                            np.random.rand(self.pop_size) * 
                            (self.upper_bounds[i_variable] - 
                             self.lower_bounds[i_variable])))
        
        if feedin_solutions is not None:
            for j in range(self.pop_size):
                self.current_pop[j, :-1] = feedin_solutions[j]
                
                
        for k in range(self.pop_size):
            self.current_pop[k, -1] = self.obj_fun(
                    self.current_pop[k, :-1])
        
        self.best_pop = (np.zeros_like(self.current_pop) +
                         self.current_pop)
        self.best_global = self.best_pop[np.argmin(
                self.best_pop[:, -1]), :] 
        self.v_current = np.zeros((self.pop_size,
                                   self.num_design_variables))
        np.savetxt(os.path.join(self.save_path, 'initial_pop.txt'),
                   self.current_pop)
        self.record()
   

    def record(self):
        self.hist.append([np.min(self.best_pop[:, -1]), 
                          np.max(self.best_pop[:, -1]),
                          np.mean(self.best_pop[:, -1]),
                          np.std(self.best_pop[:, -1]),
                          np.min(self.current_pop[:, -1]), 
                          np.max(self.current_pop[:, -1]),
                          np.mean(self.current_pop[:, -1]),
                          np.std(self.current_pop[:, -1]),
                          ])

    
    def save_current_best(self):
        i_best = np.argmin(self.current_pop[:, -1])
        np.savetxt(os.path.join(self.save_path, 'current_best_solution.txt'),
                   self.current_pop[i_best, :])
        np.savetxt(os.path.join(self.save_path, 'current_hist.txt'),
                   self.hist)
        with open(os.path.join(self.save_path, 'current_info.txt'), 'w') as f:
            f.write('gen: {0}\n'.format(self.i_gen)  + 
                    'pop_size: {0}\n'.format(self.pop_size) +
                    'alpha: {0}\n'.format(self.alpha) +
                    'beta_g: {0}\n'.format(self.beta_g) +
                    'beta_l: {0}\n'.format(self.beta_l))

        
    def evolve(self):
        self.v_current = (self.alpha*self.v_current + self.beta_l *
                          np.random.rand() * (self.best_pop[:, :-1] -
                          self.current_pop[:, :-1]) + self.beta_g *           
                          np.random.rand() * 
                          (self.best_global[np.newaxis, :-1] -
                           self.current_pop[:, :-1]))
        self.current_pop[:, :-1] = (self.current_pop[:, :-1] +
                        self.v_current)
        
        for k in range(self.pop_size):
            self.current_pop[k, :-1] = np.where(
                    self.current_pop[k, :-1] < self.lower_bounds,
                    self.lower_bounds, self.current_pop[k, :-1])
            self.current_pop[k, :-1] = np.where(
                    self.current_pop[k, :-1] > self.upper_bounds,
                    self.upper_bounds, self.current_pop[k, :-1])
            
            self.current_pop[k, -1] = self.obj_fun(
                    self.current_pop[k, :-1])
            
            if self.current_pop[k, -1] < self.best_pop[k, -1]:
                self.best_pop[k, :] = self.current_pop[k, :]
        
        self.best_global = self.best_pop[
                np.argmin(self.best_pop[:, -1]), :]
        self.record()
        self.save_current_best()
        self.i_gen += 1
  
    
    def pso_run(self, feedin_solutions=None):
        
        self.initialize_pop(feedin_solutions)
        while self.i_gen < self.max_gen:
            self.evolve()
            print('Gen {0}/{1}, '.format(self.i_gen, self.max_gen) + 
                  'Global best obj = {0:.6f}, '.format(
                          self.best_global[-1]) +
                  'Current best obj = {0:.6f}, '.format(
                          min(self.current_pop[:, -1])) +
                  'Current worst obj = {0:.6f}'.format(
                          max(self.current_pop[:, -1]))
                          )
        
        best_solution = self.best_global[:-1]
        best_obj = self.best_global[-1]

        return best_solution, best_obj, self.hist
        
 

if __name__ == '__main__': 
    
    def obj_fun(x):
        return (x[0]-2)**2 + (x[1]-1)**2
    
    num_design_variable = 2
    lower_bounds = np.array([-5, -5])
    upper_bounds = np.array([5, 5])
    
    PSO_opt = PSO(obj_fun, num_design_variable, 
                       lower_bounds, upper_bounds, max_gen=100)
        
    PSO_opt.pso_run()
    
    
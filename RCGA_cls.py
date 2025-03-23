
import numpy as np
import os
import time


class RealCodedGA(object):
    """ Real-coded genetic algorithm based on ranking selection, 
    direction-based crossover and dynbamic random mutation, as proposed in the 
    following paper is implemented here:
        
        Chuang, Yao-Chen, Chyi-Tsong Chen, and Chyi Hwang. "A simple and 
        efficient real-coded genetic algorithm for constrained optimization." 
        Applied Soft Computing 38 (2016): 87-105.
        
    Constraints can be included in obj_fun using static penalty function as
    described in the above paper. Lower and upper bounds for each design
    variables are required.
    """
    def __init__(self, obj_fun, num_design_variables, 
                 lower_bounds, upper_bounds, pop_size=10, max_gen=10,
                 p_remove=None, p_crossover=0.9, mutation_bound=0.5,
                 save_path=None):
        """ p_crossover controls the probability two selected parents being
        crossovered (equal to 1-r in the paper), and mutation_bound controls
        the largest mutation amplitute with regards to the distance between
        upper and lower bounds for each design variable.
        """
        self.obj_fun = obj_fun
        self.num_design_variables = num_design_variables
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.i_gen = 0
        
        if p_remove is None:
            self.p_remove = 1/self.num_design_variables
        else:
            self.p_remove = p_remove
            
        self.p_crossover = 0.9
        self.mutation_bound = 0.5
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
        self.current_pop = np.zeros(
                (self.pop_size, self.num_design_variables+1))
        
        for i_variable in range(self.num_design_variables):
            self.current_pop[:, i_variable] = (
                    self.lower_bounds[i_variable] +
                            np.random.rand(self.pop_size) * 
                            (self.upper_bounds[i_variable] - 
                             self.lower_bounds[i_variable]))
        
        if feedin_solutions is not None:
            for j in range(self.pop_size):
                self.current_pop[j, :-1] = feedin_solutions[j]
                
        for k in range(self.pop_size):
            self.current_pop[k, -1] = self.obj_fun(
                    self.current_pop[k, :-1])
            
        self.rank_pop()    
        self.record()
        np.savetxt(os.path.join(self.save_path, 'initial_pop.txt'),
                   self.current_pop)
        self.new_pop = np.zeros_like(self.current_pop) + self.current_pop
        

    def record(self):
        self.hist.append([np.min(self.current_pop[:, -1]),
                          np.max(self.current_pop[:, -1]),
                          np.mean(self.current_pop[:, -1]),
                          np.std(self.current_pop[:, -1])])
        

    def rank_pop(self):
        self.current_pop = self.current_pop[
                np.argsort(self.current_pop[:, -1]), :]
        
    
    def rank_selection(self):
        self.rank_pop()
        num_remove = int(self.p_remove * self.pop_size)
        if num_remove == 0:
            num_remove = 1
        self.current_pop[-num_remove:, :] = (
                self.current_pop[:num_remove, :])
        self.rank_pop()
        self.obj_max_current = np.max(self.current_pop[:, -1])
        self.obj_min_current = np.min(self.current_pop[:, -1])
        

    def crossover_and_mutation(self):
        
        M = self.pop_size // 2

        for i in range(M):
            solution_A = self.current_pop[i, :]
            solution_B = self.current_pop[i+M, :]
            
            if (np.random.rand() < self.p_crossover and solution_A[-1] != 
                solution_B[-1]):
                solution_A_new, solution_B_new = self.crossover(
                        solution_A, solution_B)
            else:
                solution_A_new, solution_B_new = self.mutation(
                        solution_A, solution_B)
                
            solution_A_new[-1] = self.obj_fun(solution_A_new[:-1])
            solution_B_new[-1] = self.obj_fun(solution_B_new[:-1])    
            self.new_pop[i, :] = solution_A_new
            self.new_pop[i+M, :] = solution_B_new
                
    
    def replacement(self):
        for i_pop in range(self.pop_size):
            if self.new_pop[i_pop, -1] < self.current_pop[i_pop, -1]:
                self.current_pop[i_pop, :] = self.new_pop[i_pop, :]
        
        
    def crossover(self, solution_A, solution_B):
        sc = np.abs(solution_A[-1] - solution_B[-1])/(self.obj_max_current - 
                   self.obj_min_current)
        
        D = np.where(np.random.rand(self.num_design_variables+1)>=0.5, 
                     solution_A - solution_B, 0)
        
        solution_A_new = solution_A + sc*D
        solution_B_new = solution_B + sc*D
 
        return (self.repair_bounds(solution_A_new), 
                self.repair_bounds(solution_B_new))
    

    def mutation(self, solution_A, solution_B):
        sm = 1-((1-self.i_gen)/self.max_gen)**2
        mutation_sizes = -self.mutation_bound + np.random.rand(
                self.num_design_variables)*2*self.mutation_bound
                
        solution_A_new = np.zeros_like(solution_A)
        solution_B_new = np.zeros_like(solution_B)
        
        solution_A_new[:-1] = solution_A[:-1] + sm*mutation_sizes*(
                self.upper_bounds - self.lower_bounds)
        solution_B_new[:-1] = solution_B[:-1] + sm*mutation_sizes*(
                self.upper_bounds - self.lower_bounds)
        
        return (self.repair_bounds(solution_A_new), 
                self.repair_bounds(solution_B_new))
        
    
    def repair_bounds(self, solution):
        solution[:-1] = np.where(solution[:-1] < self.lower_bounds, 
                self.lower_bounds, solution[:-1])
        solution[:-1] = np.where(solution[:-1] > self.upper_bounds, 
                self.upper_bounds, solution[:-1])
        return solution
    
    
    def save_current_best(self, cpu_time):
        i_best = np.argmin(self.current_pop[:, -1])
        np.savetxt(os.path.join(self.save_path, 'current_best_solution.txt'),
                   self.current_pop[i_best, :])
        np.savetxt(os.path.join(self.save_path,  'current_hist.txt'),
                   self.hist)
        with open(os.path.join(self.save_path, 'current_info.txt'), 'w') as f:
            f.write('gen: {0}\n'.format(self.i_gen)  + 
                    'pop_size: {0}\n'.format(self.pop_size) +
                    'p_crossover: {0}\n'.format(self.p_crossover) +
                    'p_remove: {0}\n'.format(self.p_remove) +
                    'mutation_bound: {0}\n'.format(self.mutation_bound) +
                    'cpu_time: {0}\n'.format(cpu_time) )

        
    def evolve(self):
        self.rank_selection()
        self.crossover_and_mutation()
        self.replacement()
        self.record()
        self.i_gen += 1
    

    def ga_run(self, feedin_solutions=None, max_gen=None):
        start = time.time()
        self.initialize_pop(feedin_solutions)
        if max_gen is None:
            max_gen = self.max_gen
    
        while self.i_gen < max_gen:
            self.evolve()
            cpu_time = time.time() - start
            self.save_current_best(cpu_time)
            print('Gen {0}/{1}, '.format(self.i_gen, max_gen) + 
                  'Current best obj = {0:.6f}, '.format(
                          min(self.current_pop[:, -1])) +
                  'worst obj = {0:.6f}'.format(max(self.current_pop[:, -1])))

        i_best = np.argmin(self.current_pop[:, -1])
        best_solution = self.current_pop[i_best, :-1]
        best_obj = self.current_pop[i_best, -1]
        
        return best_solution, best_obj, self.hist
    


if __name__ == '__main__': 
    
    def obj_fun(x):
        return (x[0]-2)**2 + (x[1]-1)**2
    
    num_design_variable = 2
    lower_bounds = np.array([-5, -5])
    upper_bounds = np.array([5, 5])
    
    RCGA_opt = RealCodedGA(obj_fun, num_design_variable, 
                       lower_bounds, upper_bounds, pop_size=50,
                       max_gen=50)
        
    RCGA_opt.ga_run()
    
    
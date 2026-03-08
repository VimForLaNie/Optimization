import numpy as np
from optimizer import Optimizer

class OriginalHS(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, c_r: float = 0.95, pa_r: float = 0.05, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c_r = self.validator.check_float("c_r", c_r, (0, 1.0))
        self.pa_r = self.validator.check_float("pa_r", pa_r, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "c_r", "pa_r"])
        self.sort_flag = False

    def initialize_variables(self):
        self.fw = 0.0001 * (self.problem.ub - self.problem.lb)
        self.fw_damp = 0.9995 
        self.dyn_fw = self.fw

    def evolve(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.generator.uniform(self.problem.lb, self.problem.ub)
            delta = self.dyn_fw * self.generator.normal(self.problem.lb, self.problem.ub)
            pos_new = np.where(self.generator.random(self.problem.n_dims) < self.c_r, self.g_best.solution, pos_new)
            x_new = pos_new + delta
            pos_new = np.where(self.generator.random(self.problem.n_dims) < self.pa_r, x_new, pos_new)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        self.dyn_fw = self.dyn_fw * self.fw_damp
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_new, self.pop_size, minmax=self.problem.minmax)

if __name__ == "__main__" :
    import random
    import numpy as np
    from optimizer import *
    random.seed(42)
    N_CITIES = 10
    cities = [ (random.uniform(0, 100), random.uniform(0, 100)) for _ in range(N_CITIES) ]

    import math

    def distance(a, b):
        return math.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )

    dist_matrix = [ [0]*N_CITIES for _ in range(N_CITIES) ]
    for i in range(N_CITIES):
        for j in range(N_CITIES):
            dist_matrix[i][j] = distance(cities[i], cities[j])

    def evaluate(npindividual) :
        total_distance = 0
        individual = np.argsort(npindividual).tolist()
        # print(individual)
        if len(individual) != N_CITIES :
            raise ValueError("Invalid individual length")
        for i in range(len(individual) - 1) :
            total_distance += dist_matrix[individual[i]][ individual[i + 1]]
        total_distance += dist_matrix[individual[-1]][individual[0]]
        return total_distance

    def tsp_cost(solution) :
        perm = np.argsort(solution)
        # perm = solution
        return evaluate(perm)

    problem_dict = {
        "obj_func" : tsp_cost,
        "bounds": IntegerVar(lb=[0] * N_CITIES, ub=[1]*N_CITIES,),
        "minmax" : "min",
    }
    optimizer = OriginalHS(epoch=200, pop_size=100)
    optimizer.solve(problem_dict)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
    print(evaluate(np.argsort(optimizer.g_best.solution)))
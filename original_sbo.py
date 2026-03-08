import random
import numpy as np
from optimizer import *

class OriginalSBO(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, alpha: float = 0.94, p_m: float = 0.05, psw: float = 0.02, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.alpha = self.validator.check_float("alpha", alpha, [0.5, 3.0])
        self.p_m = self.validator.check_float("p_m", p_m, (0, 1.0))
        self.psw = self.validator.check_float("psw", psw, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "alpha", "p_m", "psw"])
        self.sort_flag = False

    def evolve(self, epoch):
        self.sigma = self.psw * (self.problem.ub - self.problem.lb)
        fit_list = np.array([agent.target.fitness for agent in self.pop])
        pop_new = []
        for idx in range(0, self.pop_size):
            rdx = self.get_index_roulette_wheel_selection(fit_list)
            lamda = self.alpha * self.generator.uniform()
            pos_new = self.pop[idx].solution + lamda * ((self.pop[rdx].solution + self.g_best.solution) / 2 - self.pop[idx].solution)
            temp = self.pop[idx].solution + self.generator.normal(0, 1, self.problem.n_dims) * self.sigma
            pos_new = np.where(self.generator.random(self.problem.n_dims) < self.p_m, temp, pos_new)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

if __name__ == "__main__" :
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
    optimizer = OriginalSBO(epoch=200, pop_size=100)
    optimizer.solve(problem_dict)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
    print(evaluate(np.argsort(optimizer.g_best.solution)))
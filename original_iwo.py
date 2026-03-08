import random
import numpy as np
from optimizer import *

class OriginalIWO(Optimizer):
    def __init__(self, epoch:int = 10000, pop_size: int = 100, seed_min = 2, seed_max = 10,
                 exponent = 2, sigma_start = 1.0, sigma_end = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch",epoch, [1, 10000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5,10000])
        self.seed_min = self.validator.check_int("seed_min", seed_min, [1,3])
        self.seed_max = self.validator.check_int("seed_max", seed_max, [4,int(self.pop_size/2)])
        self.exponent = self.validator.check_int("exponent", exponent, [2, 4])
        self.sigma_start = self.validator.check_float("sigma_start", sigma_start, [0.5, 5.0])
        self.sigma_end = self.validator.check_float("sigma_end", sigma_end, (0, 0.5))
        self.set_parameters(["epoch", "pop_size", "seed_min", "seed_max", "exponent", "sigma_start", "sigma_end"])
        self.sort_flag = True

    def evolve(self, epoch:int) :
        sigma = (1. - epoch/self.epoch) ** self.exponent * (self.sigma_start - self.sigma_end) + self.sigma_end
        pop, list_best, list_worst = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        best, worst = list_best[0], list_worst[0]
        pop_new = []
        for idx in range(0, self.pop_size):
            temp = best.target.fitness - worst.target.fitness
            if temp < 1e-18:
                ratio = self.generator.random()
            else:
                ratio = (pop[idx].target.fitness - worst.target.fitness) / temp
            s = int(np.ceil(self.seed_min + (self.seed_max - self.seed_min) * ratio))
            if s > int(np.sqrt(self.pop_size)):
                s = int(np.sqrt(self.pop_size))
            pop_local = []
            for jdx in range(s):
                pos_new = pop[idx].solution + sigma * self.generator.normal(0, 1, self.problem.n_dims)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_local.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_local[-1].target = self.get_target(pos_new)
            if self.mode in self.AVAILABLE_MODES:
                pop_local = self.update_target_for_population(pop_local)
            pop_new += pop_local
        self.pop = self.get_sorted_and_trimmed_population(pop_new, self.pop_size, self.problem.minmax)

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
    optimizer = OriginalIWO(epoch=100, pop_size=100)
    optimizer.solve(problem_dict)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
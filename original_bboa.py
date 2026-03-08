import random
import numpy as np
from optimizer import *

class OriginalBBOA(Optimizer) :
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.mu = (self.pop_size + 1 - np.array(range(1, self.pop_size + 1))) / (self.pop_size + 1)
        self.mr = 1 - self.mu

    def evolve(self, epoch):
        pp = epoch / self.epoch

        ## Pedal marking behaviour
        pop_new = []
        for idx in range(0, self.pop_size):
            if pp <= 1/3:           
                # print("case 1")
                pos_new = self.pop[idx].solution + (-pp * self.generator.random(self.problem.n_dims) * self.pop[idx].solution)
            elif 1/3 < pp <= 2/3:
                # print("case 2")
                qq = pp * self.generator.random(self.problem.n_dims)
                pos_new = self.pop[idx].solution + (qq * (self.g_best.solution - self.generator.integers(1, 3) * self.g_worst.solution))
            else:
                # print("case 3")
                ww = 2 * pp * np.pi * self.generator.random(self.problem.n_dims)
                pos_new = self.pop[idx].solution + (ww*self.g_best.solution - np.abs(self.pop[idx].solution)) - (ww*self.g_worst.solution - np.abs(self.pop[idx].solution))
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        pop_new = []
        for idx in range(0, self.pop_size):
            kk = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            if self.compare_target(self.pop[idx].target, self.pop[kk].target, self.problem.minmax):
                pos_new = self.pop[idx].solution + self.generator.random() * (self.pop[idx].solution - self.pop[kk].solution)
            else:
                pos_new = self.pop[idx].solution + self.generator.random() * (self.pop[kk].solution - self.pop[idx].solution)
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
    N_CITIES = 1000
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
        # perm = np.argsort(solution)
        perm = solution
        return evaluate(perm)

    problem_dict = {
        "obj_func" : tsp_cost,
        "bounds": IntegerVar(lb=[0] * N_CITIES, ub=[1]*N_CITIES,),
        "minmax" : "min",
    }
    optimizer = BBOA(epoch=100, pop_size=100)
    optimizer.solve(problem_dict)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
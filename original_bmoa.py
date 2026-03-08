from optimizer import *
import random
import numpy as np

class OriginalBMOA(Optimizer) :
    def __init__(self, epoch=10000, pop_size=100, pl=5, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pl = self.validator.check_int("pl", pl, [1, self.pop_size-1])
        self.set_parameters(["epoch", "pop_size", "pl"])
        self.sort_flag = True

    def evolve(self, epoch):
        k1 = self.generator.permutation(self.pop_size)
        k2 = self.generator.permutation(self.pop_size)
        temp = np.abs(k1 - k2)
        pop_new = []
        for idx in range(0, self.pop_size):
            if temp[idx] <= self.pl:
                p = self.generator.uniform(0, 1)
                pos_new = p * self.pop[k1[idx]].solution + (1 - p) * self.pop[k2[idx]].solution
            else:
                pos_new = self.generator.uniform(0, 1) * self.pop[k2[idx]].solution
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_new)

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
    optimizer = BMOA(epoch=100, pop_size=100)
    optimizer.solve(problem_dict)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
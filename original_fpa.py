import numpy as np
from optimizer import Optimizer

class OriginalFPA(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_s: float = 0.8, levy_multiplier: float = 0.1, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_s = self.validator.check_float("p_s", p_s, (0, 1.0))
        self.levy_multiplier = self.validator.check_float("levy_multiplier", levy_multiplier, (-10000, 10000))
        self.set_parameters(["epoch", "pop_size", "p_s", "levy_multiplier"])
        self.sort_flag = False

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        random_pos = self.problem.generate_solution()
        return np.where(condition, solution, random_pos)

    def evolve(self, epoch):
        pop = []
        for idx in range(0, self.pop_size):
            if self.generator.uniform() < self.p_s:
                levy = self.get_levy_flight_step(multiplier=self.levy_multiplier, size=self.problem.n_dims, case=-1)
                pos_new = self.pop[idx].solution + 1.0 / np.sqrt(epoch) * levy * (self.pop[idx].solution - self.g_best.solution)
            else:
                id1, id2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.pop[idx].solution + self.generator.uniform() * (self.pop[id1].solution - self.pop[id2].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)

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
    optimizer = OriginalFPA(epoch=200, pop_size=100)
    optimizer.solve(problem_dict)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
    print(evaluate(np.argsort(optimizer.g_best.solution)))
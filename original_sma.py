import random
import numpy as np
from optimizer import *

class OriginalSMA(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_t: float = 0.03, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_t = self.validator.check_float("p_t", p_t, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "p_t"])
        self.sort_flag = True

    def initialize_variables(self):
        self.weights = np.zeros((self.pop_size, self.problem.n_dims))

    def evolve(self, epoch):
        ss = self.g_best.target.fitness - self.pop[-1].target.fitness + self.EPSILON
        for idx in range(0, self.pop_size):
            if idx <= int(self.pop_size / 2):
                self.weights[idx] = 1 + self.generator.uniform(0, 1, self.problem.n_dims) * \
                                    np.log10((self.g_best.target.fitness - self.pop[idx].target.fitness) / ss + 1)
            else:
                self.weights[idx] = 1 - self.generator.uniform(0, 1, self.problem.n_dims) * \
                                    np.log10((self.g_best.target.fitness - self.pop[idx].target.fitness) / ss + 1)
        a = np.arctanh(1 - epoch / self.epoch)
        b = 1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            if self.generator.uniform() < self.p_t:
                pos_new = self.problem.generate_solution()
            else:
                p = np.tanh(np.abs(self.pop[idx].target.fitness - self.g_best.target.fitness))
                vb = self.generator.uniform(-a, a, self.problem.n_dims)
                vc = self.generator.uniform(-b, b, self.problem.n_dims)
                id_a, id_b = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_1 = self.g_best.solution + vb * (self.weights[idx] * self.pop[id_a].solution - self.pop[id_b].solution)
                pos_2 = vc * self.pop[idx].solution
                condition = self.generator.random(self.problem.n_dims) < p
                pos_new = np.where(condition, pos_1, pos_2)
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
    optimizer = OriginalSMA(epoch=200, pop_size=100)
    optimizer.solve(problem_dict)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
    print(evaluate(np.argsort(optimizer.g_best.solution)))
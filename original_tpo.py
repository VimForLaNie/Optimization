import numpy as np
from optimizer import Optimizer

class OriginalTPO(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, alpha: float = 0.3, beta: float = 50.0, theta: float = 0.9, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.alpha = self.validator.check_float("alpha", alpha, [-10.0, 10.])
        self.beta = self.validator.check_float("beta", beta, [-100., 100])
        self.theta = self.validator.check_float("theta", theta, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "alpha", "beta", "theta"])
        self.sort_flag = False

    def initialize_variables(self):
        self.n_leafs = int(np.sqrt(self.pop_size) + 1)  # Number of leafs
        self._theta = self.theta
        self.roots = self.generator.uniform(0, 1, (self.n_leafs, self.problem.n_dims))

    def initialization(self):
        self.pop_total = []
        self.pop = []                       # The best leaf in each branches
        for idx in range(self.pop_size):
            leafs = self.generate_population(self.n_leafs)
            best = self.get_best_agent(leafs, self.problem.minmax)
            self.pop.append(best)
            self.pop_total.append(leafs)

    def evolve(self, epoch):
        for idx in range(0, self.pop_size):
            pos_list = np.array([agent.solution for agent in self.pop_total[idx]])
            carbon_gain = self._theta * self.g_best.solution - pos_list
            roots_old = np.copy(self.roots)
            self.roots += self.alpha * carbon_gain * self.generator.uniform(-0.5, 0.5, (self.n_leafs, self.problem.n_dims))
            nutrient_value = self._theta * (self.roots - roots_old)
            pos_list_new = self.g_best.solution + self.beta * nutrient_value
            pop_new = []
            for jdx in range(0, self.n_leafs):
                pos_new = self.correct_solution(pos_list_new[jdx])
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop_total[idx][jdx] = self.get_better_agent(agent, self.pop_total[idx][jdx], self.problem.minmax)
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)
                self.pop_total[idx] = self.greedy_selection_population(pop_new, self.pop_total[idx], self.problem.minmax)
        self._theta = self._theta * self.theta
        for idx in range(0, self.pop_size):
            best = self.get_best_agent(self.pop_total[idx], self.problem.minmax)
            self.pop[idx] = best

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
    optimizer = OriginalTPO(epoch=200, pop_size=100)
    optimizer.solve(problem_dict)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
    print(evaluate(np.argsort(optimizer.g_best.solution)))
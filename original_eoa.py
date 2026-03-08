from optimizer import *
import random
import numpy as np

class OriginalEOA(Optimizer):

    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_c: float = 0.9, p_m: float = 0.01, n_best: int = 2,
                 alpha: float = 0.98, beta: float = 0.9, gama: float = 0.9, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_c = self.validator.check_float("p_c", p_c, (0, 1.0))
        self.p_m = self.validator.check_float("p_m", p_m, (0, 1.0))
        self.n_best = self.validator.check_int("n_best", n_best, [2, int(self.pop_size / 2)])
        self.alpha = self.validator.check_float("alpha", alpha, (0, 1.0))
        self.beta = self.validator.check_float("beta", beta, (0, 1.0))
        self.gama = self.validator.check_float("gama", gama, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "p_c", "p_m", "n_best", "alpha", "beta", "gama"])
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_beta = self.beta

    def evolve(self, epoch):
        pop_elites, _, _ = self.get_special_agents(self.pop, n_best=1, minmax=self.problem.minmax)
        pop = []
        for idx in range(0, self.pop_size):
            x_t1 = self.problem.lb + self.problem.ub - self.alpha * self.pop[idx].solution
            if idx >= self.n_best:
                idx = int(self.pop_size * 0.2)
                if self.generator.uniform() < 0.5:
                    idx1, idx2 = self.generator.choice(range(0, idx), 2, replace=False)
                else:
                    idx1, idx2 = self.generator.choice(range(idx, self.pop_size), 2, replace=False)
                r = self.generator.uniform()
                x_child = r * self.pop[idx2].solution + (1 - r) * self.pop[idx1].solution
            else:
                r1 = self.generator.integers(0, self.pop_size)
                x_child = self.pop[r1].solution
            x_t1 = self.dyn_beta * x_t1 + (1.0 - self.dyn_beta) * x_child
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
        self.dyn_beta = self.gama * self.beta
        self.pop = self.get_sorted_and_trimmed_population(self.pop, self.pop_size, self.problem.minmax)

        pos_list = np.array([agent.solution for agent in self.pop])
        x_mean = np.mean(pos_list, axis=0)
        # Cauchy Mutation :O
        cauchy_w = self.g_best.solution.copy()
        pop_new = []
        for idx in range(self.n_best, self.pop_size):  # Don't allow the elites to be mutated
            condition = self.generator.random(self.problem.n_dims) < self.p_m
            cauchy_w = np.where(condition, x_mean, cauchy_w)
            x_t1 = (cauchy_w + self.g_best.solution) / 2
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop[self.n_best:] = self.greedy_selection_population(pop_new, self.pop[self.n_best:], self.problem.minmax)

        self.pop, _, _ = self.get_special_agents(self.pop, minmax=self.problem.minmax)
        for idx in range(0, self.n_best):
            self.pop[self.pop_size - idx - 1] = pop_elites[idx].copy()

        new_set = set()
        for idx, agent in enumerate(self.pop):
            if tuple(agent.solution.tolist()) in new_set:
                self.pop[idx] = self.generate_agent()
            else:
                new_set.add(tuple(agent.solution.tolist()))

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
        perm = np.argsort(solution)
        # perm = solution
        return evaluate(perm)

    problem_dict = {
        "obj_func" : tsp_cost,
        "bounds": IntegerVar(lb=[0] * N_CITIES, ub=[1]*N_CITIES,),
        "minmax" : "min",
    }
    optimizer = OriginalEOA(epoch=100, pop_size=100)
    optimizer.solve(problem_dict,seed=44)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
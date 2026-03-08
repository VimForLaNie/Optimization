import random
import numpy as np
from optimizer import *
    
class OriginalBBO(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_m: float = 0.01, n_elites: int = 2, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_m = self.validator.check_float("p_m", p_m, (0., 1.0))
        self.n_elites = self.validator.check_int("n_elites", n_elites, [2, int(self.pop_size / 2)])
        self.set_parameters(["epoch", "pop_size", "p_m", "n_elites"])
        self.sort_flag = False
        self.mu = (self.pop_size + 1 - np.array(range(1, self.pop_size + 1))) / (self.pop_size + 1)
        self.mr = 1 - self.mu

    def evolve(self, epoch: int) -> None:
        _, pop_elites, _ = self.get_special_agents(self.pop, n_best=self.n_elites, minmax=self.problem.minmax)
        pop = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution.copy()
            for j in range(self.problem.n_dims):
                if self.generator.random() < self.mr[idx]:  
                    random_number = self.generator.random() * np.sum(self.mu)
                    select = self.mu[0]
                    select_index = 0
                    while (random_number > select) and (select_index < self.pop_size - 1):
                        select_index += 1
                        select += self.mu[select_index]
                    pos_new[j] = self.pop[select_index].solution[j]
            noise = self.generator.uniform(self.problem.lb, self.problem.ub)
            condition = self.generator.random(self.problem.n_dims) < self.p_m
            pos_new = np.where(condition, noise, pos_new)
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_empty_agent(pos_new)
            pop.append(agent_new)
            if self.mode not in self.AVAILABLE_MODES:
                agent_new.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent_new, minmax=self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_elites, self.pop_size, self.problem.minmax)

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        return solution
    
    def correct_solution(self, solution: np.ndarray) -> np.ndarray:
        solution = self.amend_solution(solution)
        return self.problem.correct_solution(solution)
    
    
if __name__ == "__main__":
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

    # shuffle_list = [i for i in range(N_CITIES)]
    # random.shuffle(shuffle_list)
    # shuffle_list = [0,9,3,4,1,6,5,8,7,2]
    # res = evaluate(shuffle_list)
    # print(res)

    def tsp_cost(solution) :
        # perm = np.argsort(solution)
        perm = solution
        return evaluate(perm)

    problem_dict = {
        "obj_func" : tsp_cost,
        "bounds": IntegerVar(lb=[0] * N_CITIES, ub=[1]*N_CITIES,),
        "minmax" : "min",
    }
    
    optimizer = BBO(epoch=200, pop_size=100)
    optimizer.solve(problem_dict, seed=45)

    print(optimizer.g_best)
    print(np.argsort(optimizer.g_best))
    print(optimizer.g_best.target.fitness)


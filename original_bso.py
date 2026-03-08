import numpy as np
from optimizer import Optimizer

class OriginalBSO(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, m_clusters: int = 5,
                 p1: float = 0.25, p2: float = 0.5, p3: float = 0.75, p4: float = 0.5, **kwargs: object) -> None:

        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.m_clusters = self.validator.check_int("m_clusters", m_clusters, [2, int(self.pop_size/5)])
        self.p1 = self.validator.check_float("p1", p1, (0, 1.0))
        self.p2 = self.validator.check_float("p2", p2, (0, 1.0))
        self.p3 = self.validator.check_float("p3", p3, (0, 1.0))
        self.p4 = self.validator.check_float("p4", p4, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "m_clusters", "p1", "p2", "p3", "p4"])
        self.sort_flag = False
        self.m_solution = int(self.pop_size / self.m_clusters)
        self.pop_group, self.centers = None, None

    def find_cluster__(self, pop_group):
        centers = []
        for idx in range(0, self.m_clusters):
            local_best = self.get_best_agent(pop_group[idx], self.problem.minmax)
            centers.append(local_best.copy())
        return centers

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.pop_group = self.generate_group_population(self.pop, self.m_clusters, self.m_solution)
        self.centers = self.find_cluster__(self.pop_group)

    def evolve(self, epoch):
        epsilon = 1. - 1. * epoch / self.epoch
        if self.generator.uniform() < self.p1:
            idx = self.generator.integers(0, self.m_clusters)
            self.centers[idx] = self.generate_agent()
        pop_group = self.pop_group
        for idx in range(0, self.pop_size):
            cluster_id = int(idx / self.m_solution)
            location_id = int(idx % self.m_solution)

            if self.generator.uniform() < self.p2:
                if self.generator.uniform() < self.p3:
                    pos_new = self.centers[cluster_id].solution + epsilon * self.generator.normal(0, 1, self.problem.n_dims)
                else:
                    levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.001, size=self.problem.n_dims, case=-1)
                    pos_new = self.pop_group[cluster_id][location_id].solution + levy_step
            else:
                id1, id2 = self.generator.choice(range(0, self.m_clusters), 2, replace=False)
                if self.generator.uniform() < self.p4:
                    pos_new = 0.5 * (self.centers[id1].solution + self.centers[id2].solution) + epsilon * self.generator.normal(0, 1, self.problem.n_dims)
                else:
                    rand_id1 = self.generator.integers(0, self.m_solution)
                    rand_id2 = self.generator.integers(0, self.m_solution)
                    pos_new = 0.5 * (self.pop_group[id1][rand_id1].solution + self.pop_group[id2][rand_id2].solution) + \
                              epsilon * self.generator.normal(0, 1, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_group[cluster_id][location_id] = agent
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop_group[cluster_id][location_id] = self.get_better_agent(agent, self.pop_group[cluster_id][location_id], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            for idx in range(0, self.m_clusters):
                pop_group[idx] = self.update_target_for_population(pop_group[idx])
                pop_group[idx] = self.greedy_selection_population(self.pop_group[idx], pop_group[idx], self.problem.minmax)

        self.centers = self.find_cluster__(pop_group)
        self.pop = []
        for idx in range(0, self.m_clusters):
            self.pop += pop_group[idx]

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
    optimizer = OriginalBSO(epoch=200, pop_size=100)
    optimizer.solve(problem_dict)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
    print(evaluate(np.argsort(optimizer.g_best.solution)))
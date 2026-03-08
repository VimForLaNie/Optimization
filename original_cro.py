import numpy as np
from optimizer import Optimizer

class OriginalCRO(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, po: float = 0.4, Fb: float = 0.9, Fa: float = 0.1, Fd: float = 0.1,
                 Pd: float = 0.5, GCR: float = 0.1, gamma_min: float = 0.02, gamma_max: float = 0.2, n_trials: int = 3, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.po = self.validator.check_float("po", po, (0, 1.0))
        self.Fb = self.validator.check_float("Fb", Fb, (0, 1.0))
        self.Fa = self.validator.check_float("Fa", Fa, (0, 1.0))
        self.Fd = self.validator.check_float("Fd", Fd, (0, 1.0))
        self.Pd = self.validator.check_float("Pd", Pd, (0, 1.0))
        self.GCR = self.validator.check_float("GCR", GCR, (0, 1.0))
        self.gamma_min = self.validator.check_float("gamma_min", gamma_min, (0, 0.15))
        self.gamma_max = self.validator.check_float("gamma_max", gamma_max, (0.15, 1.0))
        self.n_trials = self.validator.check_int("n_trials", n_trials, [2, int(self.pop_size / 2)])
        self.set_parameters(["epoch", "pop_size", "po", "Fb", "Fa", "Fd", "Pd", "GCR", "gamma_min", "gamma_max", "n_trials"])
        self.sort_flag = False

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.reef = np.array([])
        self.occupied_position = []
        self.G1 = self.gamma_max
        self.alpha = 10 * self.Pd / self.epoch
        self.gama = 10 * (self.gamma_max - self.gamma_min) / self.epoch
        self.num_occupied = int(self.pop_size / (1 + self.po))
        self.dyn_Pd = 0
        self.occupied_list = np.zeros(self.pop_size)
        self.occupied_idx_list = self.generator.choice(list(range(self.pop_size)), self.num_occupied, replace=False)
        self.occupied_list[self.occupied_idx_list] = 1

    def gaussian_mutation__(self, position):
        random_pos = position + self.G1 * (self.problem.ub - self.problem.lb) * self.generator.normal(0, 1, self.problem.n_dims)
        condition =self.generator.random(self.problem.n_dims) < self.GCR
        pos_new = np.where(condition, random_pos, position)
        return self.correct_solution(pos_new)

    def multi_point_cross__(self, pos1, pos2):
        p1, p2 = self.generator.choice(list(range(len(pos1))), 2, replace=False)
        start, end = min(p1, p2), max(p1, p2)
        pos_new = np.concatenate((pos1[:start], pos2[start:end], pos1[end:]), axis=0)
        return self.correct_solution(pos_new)

    def larvae_setting__(self, larvae):
        for larva in larvae:
            for idx in range(self.n_trials):
                pdx = self.generator.integers(0, self.pop_size - 1)
                if self.occupied_list[pdx] == 0:
                    self.pop[pdx] = larva
                    self.occupied_idx_list = np.append(self.occupied_idx_list, pdx)
                    self.occupied_list[pdx] = 1
                    break
                else:
                    if self.compare_target(larva.target, self.pop[pdx].target, self.problem.minmax):
                        self.pop[pdx] = larva
                        break

    def sort_occupied_reef__(self):
        def reef_fitness(idx):
            return self.pop[idx].target.fitness
        return sorted(self.occupied_idx_list, key=reef_fitness)

    def broadcast_spawning_brooding__(self):
        larvae = []
        selected_corals = self.generator.choice(self.occupied_idx_list, int(len(self.occupied_idx_list) * self.Fb), replace=False)
        for idx in self.occupied_idx_list:
            if idx not in selected_corals:
                pos_new = self.gaussian_mutation__(self.pop[idx].solution)
                agent = self.generate_empty_agent(pos_new)
                larvae.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    larvae[-1].target = self.get_target(pos_new)
        while len(selected_corals) >= 2:
            id1, id2 = self.generator.choice(range(len(selected_corals)), 2, replace=False)
            pos_new = self.multi_point_cross__(self.pop[selected_corals[id1]].solution, self.pop[selected_corals[id2]].solution)
            agent = self.generate_empty_agent(pos_new)
            larvae.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                larvae[-1].target = self.get_target(pos_new)
            selected_corals = np.delete(selected_corals, [id1, id2])
        return self.update_target_for_population(larvae)

    def evolve(self, epoch):
        larvae = self.broadcast_spawning_brooding__()
        self.larvae_setting__(larvae)
        num_duplicate = int(len(self.occupied_idx_list) * self.Fa)
        pop_best = [self.pop[idx] for idx in self.occupied_idx_list]
        pop_best = self.get_sorted_and_trimmed_population(pop_best, num_duplicate, self.problem.minmax)
        self.larvae_setting__(pop_best)
        if self.generator.random() < self.dyn_Pd:
            num__depredation__ = int(len(self.occupied_idx_list) * self.Fd)
            idx_list_sorted = self.sort_occupied_reef__()
            selected_depredator = idx_list_sorted[-num__depredation__:]
            self.occupied_idx_list = np.setdiff1d(self.occupied_idx_list, selected_depredator)
            for idx in selected_depredator:
                self.occupied_list[idx] = 0
        if self.dyn_Pd <= self.Pd:
            self.dyn_Pd += self.alpha
        if self.G1 >= self.gamma_min:
            self.G1 -= self.gama

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
    optimizer = OriginalCRO(epoch=200, pop_size=100)
    optimizer.solve(problem_dict)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
    print(evaluate(np.argsort(optimizer.g_best.solution)))
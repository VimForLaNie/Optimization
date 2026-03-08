import numpy as np
from optimizer import Optimizer

class OriginalWHO(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, n_explore_step=3, n_exploit_step=3, eta=0.15, p_hi=0.9,
                 local_alpha=0.9, local_beta=0.3, global_alpha=0.2, global_beta=0.8, delta_w=2.0, delta_c=2.0, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_explore_step = self.validator.check_int("n_explore_step", n_explore_step, [2, 10])
        self.n_exploit_step = self.validator.check_int("n_exploit_step", n_exploit_step, [2, 10])
        self.eta = self.validator.check_float("eta", eta, (0, 1.0))
        self.p_hi = self.validator.check_float("p_hi", p_hi, (0, 1.0))
        self.local_alpha = self.validator.check_float("local_alpha", local_alpha, (0, 3.0))
        self.local_beta = self.validator.check_float("local_beta", local_beta, (0, 3.0))
        self.global_alpha = self.validator.check_float("global_alpha", global_alpha, (0, 3.0))
        self.global_beta = self.validator.check_float("global_beta", global_beta, (0, 3.0))
        self.delta_w = self.validator.check_float("delta_w", delta_w, (0.5, 5.0))
        self.delta_c = self.validator.check_float("delta_c", delta_c, (0.5, 5.0))
        self.set_parameters(["epoch", "pop_size", "n_explore_step", "n_exploit_step",
                             "eta", "p_hi", "local_alpha", "local_beta", "global_alpha", "global_beta", "delta_w", "delta_c"])
        self.sort_flag = False

    def evolve(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            local_list = []
            for j in range(0, self.n_explore_step):
                temp = self.pop[idx].solution + self.eta * self.generator.uniform() * self.generator.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.correct_solution(temp)
                agent = self.generate_empty_agent(pos_new)
                local_list.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    local_list[-1].target = self.get_target(pos_new)
            local_list = self.update_target_for_population(local_list)
            best_local = self.get_best_agent(local_list, self.problem.minmax)
            temp = self.local_alpha * best_local.solution + self.local_beta * (self.pop[idx].solution - best_local.solution)
            pos_new = self.correct_solution(temp)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        for idx in range(0, self.pop_size):
            idr = self.generator.choice(range(0, self.pop_size))
            if self.compare_target(self.pop[idr].target, self.pop[idx].target, self.problem.minmax) and self.generator.random() < self.p_hi:
                temp = self.global_alpha * self.pop[idx].solution + self.global_beta * self.pop[idr].solution
                pos_new = self.correct_solution(temp)
                tar_new = self.get_target(pos_new)
                if self.compare_target(tar_new, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx].update(solution=pos_new, target=tar_new)

        _, best, worst = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        g_best, g_worst = best[0], worst[0]
        pop_child = []
        for idx in range(0, self.pop_size):
            dist_to_worst = np.linalg.norm(self.pop[idx].solution - g_worst.solution)
            dist_to_best = np.linalg.norm(self.pop[idx].solution - g_best.solution)
            if dist_to_worst < self.delta_w:
                temp = self.pop[idx].solution + self.generator.uniform() * (self.problem.ub - self.problem.lb) * \
                       self.generator.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.correct_solution(temp)
                agent = self.generate_empty_agent(pos_new)
                pop_child.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
            if 1.0 < dist_to_best and dist_to_best < self.delta_c:
                temp = g_best.solution + self.eta * self.generator.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.correct_solution(temp)
                agent = self.generate_empty_agent(pos_new)
                pop_child.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
            for jdx in range(0, self.n_exploit_step):
                temp = g_best.solution + 0.1 * self.generator.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.correct_solution(temp)
                agent = self.generate_empty_agent(temp)
                pop_child.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            pop_child = self.get_sorted_and_trimmed_population(pop_child, self.pop_size, self.problem.minmax)
            self.pop = self.greedy_selection_population(self.pop, pop_child, self.problem.minmax)

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
    optimizer = OriginalWHO(epoch=200, pop_size=100)
    optimizer.solve(problem_dict)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
    print(evaluate(np.argsort(optimizer.g_best.solution)))
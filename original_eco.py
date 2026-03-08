import numpy as np
import random
from optimizer import Optimizer

class OriginalECO(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        percent = epoch / self.epoch
        if epoch % 2 == 0:
            sorted_pop = self.get_sorted_and_trimmed_population(self.pop, self.pop_size, self.problem.minmax)
            pass_ratio = 0.5
            pass_count = int(self.pop_size * pass_ratio)
            if pass_count < 1:
                pass_count = 1
            if pass_count > self.pop_size:
                pass_count = self.pop_size
            failed = sorted_pop[pass_count:]
            passed = sorted_pop[:pass_count]
            replaced = []
            for agent in failed:
                tmp = self.problem.ub - agent.solution + self.problem.lb
                pos_new = self.correct_solution(tmp)
                pos_new = self.generate_agent(pos_new)
                replaced.append(pos_new)
            self.pop = passed + replaced
            self.generator.shuffle(self.pop)
            
        pop_new = []
        for idx in range(0, self.pop_size):
            if self.generator.random() < 1 - percent:
                pop_new.append(self.pop[idx].copy())
                continue
            range_size = self.pop_size
            p = np.zeros(self.problem.n_dims, dtype=float)
            cnt = 0
            choice = self.generator.choice([0, 1])
            for i in range(0, range_size):
                it = i + 1
                cnt += 1
                if choice :
                    p += self.pop[(idx - it) % self.pop_size].solution * (1 - it / range_size)
                else :
                    p += self.pop[(idx + it) % self.pop_size].solution * (1 - it / range_size)
            if cnt != 0 :
                p = p / cnt
            pos_new = self.pop[idx].solution.copy()
            for i in range(self.problem.n_dims):
                if self.generator.random() < 0.5:
                    pos_new[i] = p[i]
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_agent(pos_new)
            pop_new.append(agent_new)
        self.pop = pop_new
        # print(self.pop)

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
    optimizer = OriginalECO(epoch=200, pop_size=100)
    optimizer.solve(problem_dict, seed=45)

    print(optimizer.g_best.solution)
    print(np.argsort(optimizer.g_best.solution))
    print(optimizer.g_best.target.fitness)
    print(evaluate(np.argsort(optimizer.g_best.solution)))
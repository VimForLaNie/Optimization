import os
import time
import numpy as np
import numbers
import numbers as nb
import random
from abc import ABC, abstractmethod

from typing import List,Tuple,Union,Dict

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

class BaseVar(ABC):
    SUPPORTED_ARRAY = [tuple, list, np.ndarray]

    def __init__(self, name="variable"):
        self.name = name
        self.n_vars = None
        self.lb, self.ub = None, None
        self._seed = None
        self.generator = np.random.default_rng()

    def set_n_vars(self, n_vars):
        if type(n_vars) is int and n_vars > 0:
            self.n_vars = n_vars
        else:
            raise ValueError(f"Invalid n_vars. It should be integer and > 0.")

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value
        self.generator = np.random.default_rng(self._seed)

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    @abstractmethod
    def correct(self, x):
        pass

    @abstractmethod
    def generate(self):
        pass

    @staticmethod
    def round(x):
        frac = x - np.floor(x)
        t1 = np.floor(x)
        t2 = np.ceil(x)
        return np.where(frac < 0.5, t1, t2)

class FloatVar(BaseVar):
    def __init__(self, lb=-10., ub=10., name="float"):
        super().__init__(name)
        self._set_bounds(lb, ub)

    def _set_bounds(self, lb, ub):
        if isinstance(lb, nb.Number) and isinstance(ub, nb.Number):
            self.lb, self.ub = np.array((lb, ), dtype=float), np.array((ub,), dtype=float)
            self.n_vars = 1
        elif type(lb) in self.SUPPORTED_ARRAY and type(ub) in self.SUPPORTED_ARRAY:
            if len(lb) == len(ub):
                self.lb, self.ub = np.array(lb, dtype=float), np.array(ub, dtype=float)
                self.n_vars = len(lb)
            else:
                raise ValueError(f"Invalid lb or ub. Length of lb should equal to length of ub.")
        else:
            raise TypeError(f"Invalid lb or ub. It should be one of following: {self.SUPPORTED_ARRAY}")

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return np.array(x, dtype=float)

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.uniform(self.lb, self.ub)

class IntegerVar(BaseVar):
    def __init__(self, lb=-10, ub=10, name="integer"):
        super().__init__(name)
        self.eps = 1e-4
        self._set_bounds(lb, ub)

    def _set_bounds(self, lb, ub):
        if isinstance(lb, nb.Number) and isinstance(ub, nb.Number):
            lb, ub = int(lb) - 0.5, int(ub) + 0.5 - self.eps
            self.lb, self.ub = np.array((lb, ), dtype=float), np.array((ub, ), dtype=float)
            self.n_vars = 1
        elif type(lb) in self.SUPPORTED_ARRAY and type(ub) in self.SUPPORTED_ARRAY:
            if len(lb) == len(ub):
                self.lb, self.ub = np.array(lb, dtype=float) - 0.5, np.array(ub, dtype=float) + (0.5 - self.eps)
                self.n_vars = len(lb)
            else:
                raise ValueError(f"Invalid lb or ub. Length of lb should equal to length of ub.")
        else:
            raise TypeError(f"Invalid lb or ub. It should be one of following: {self.SUPPORTED_ARRAY}")

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        x = self.round(x)
        return np.array(x, dtype=int)

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.integers(self.lb+0.5, self.ub+0.5+self.eps)

class StringVar(BaseVar):
    def __init__(self, valid_sets=(("",),), name="string"):
        super().__init__(name)
        self.eps = 1e-4
        self._set_bounds(valid_sets)

    def _set_bounds(self, valid_sets):
        if type(valid_sets) in self.SUPPORTED_ARRAY:
            if type(valid_sets[0]) not in self.SUPPORTED_ARRAY:
                self.n_vars = 1
                self.valid_sets = (tuple(valid_sets),)
                le = LabelEncoder().fit(valid_sets)
                self.list_le = (le,)
                self.lb = np.array([0., ])
                self.ub = np.array([len(valid_sets) - self.eps, ])
            else:
                self.n_vars = len(valid_sets)
                if all(len(item) > 1 for item in valid_sets):
                    self.valid_sets = valid_sets
                    self.list_le = []
                    ub = []
                    for vl_set in valid_sets:
                        le = LabelEncoder().fit(vl_set)
                        self.list_le.append(le)
                        ub.append(len(vl_set) - self.eps)
                    self.lb = np.zeros(self.n_vars)
                    self.ub = np.array(ub)
                else:
                    raise ValueError(f"Invalid valid_sets. All variables need to have at least 2 values.")
        else:
            raise TypeError(f"Invalid valid_sets. It should be {self.SUPPORTED_ARRAY}.")

    def encode(self, x):
        return np.array([le.transform(val)[0] for (le, val) in zip(self.list_le, x)], dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return [le.inverse_transform(val)[0] for (le, val) in zip(self.list_le, x)]

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        return np.array(x, dtype=int)

    def generate(self):
        return [self.generator.choice(np.array(vl_set, dtype=str)) for vl_set in self.valid_sets]

class PermutationVar(BaseVar):
    def __init__(self, valid_set=(1, 2), name="permutation"):
        super().__init__(name)
        self.eps = 1e-4
        self._set_bounds(valid_set)

    def _set_bounds(self, valid_set):
        if type(valid_set) in self.SUPPORTED_ARRAY and len(valid_set) > 1:
            self.valid_set = np.array(valid_set)
            self.n_vars = len(valid_set)
            self.le = LabelEncoder().fit(valid_set)
            self.lb = np.zeros(self.n_vars)
            self.ub = (self.n_vars - self.eps) * np.ones(self.n_vars)
        else:
            raise TypeError(f"Invalid valid_set. It should be {self.SUPPORTED_ARRAY} and contains at least 2 variables")

    def encode(self, x):
        return np.array(self.le.transform(x), dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return self.le.inverse_transform(x)

    def correct(self, x):
        return np.argsort(x)

    def generate(self):
        return self.generator.permutation(self.valid_set)

class BinaryVar(BaseVar):
    def __init__(self, n_vars=1, name="binary"):
        super().__init__(name)
        self.set_n_vars(n_vars)
        self.eps = 1e-4
        self.lb = np.zeros(self.n_vars)
        self.ub = (2 - self.eps) * np.ones(self.n_vars)

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return np.array(x, dtype=int)

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.integers(0, 2, self.n_vars)

class BoolVar(BaseVar):
    def __init__(self, n_vars=1, name="boolean"):
        super().__init__(name)
        self.set_n_vars(n_vars)
        self.eps = 1e-4
        self.lb = np.zeros(self.n_vars)
        self.ub = (2 - self.eps) * np.ones(self.n_vars)

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        x = np.array(x, dtype=int)
        return x == 1

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.choice([True, False], self.n_vars, replace=True)

class MixedSetVar(StringVar) :
    def __init__(self, valid_sets=(("",),), name="mixed-set-var") :
        super().__init__(valid_sets, name)
        self.eps = 1e-4
        self._set_bounds(valid_sets)

    def generate(self):
        return [self.generator.choice(np.array(vl_set, dtype=object)) for vl_set in self.valid_sets]

class Target :
    SUPPORTED_ARRAY = [tuple, list, np.ndarray]

    def __init__(self, objectives: Union[list, tuple, np.ndarray, int ,float] = None,
                 weights: Union[list, tuple, np.ndarray] = None) -> None:
        self._objectives, self._weight, self._fitness = None,None,None
        self.set_objectives(objectives)
        self.set_weights(weights)
        self.calculate_fitness(self.weights)
        
    def copy(self) -> 'Target' :
        return Target(self._objectives,self._weight)

    @property
    def objectives(self):
        return self._objectives

    def set_objectives(self,objs):
        if objs is None :
            raise ValueError(f"Invalid objectives")
        else :
            if type(objs) not in self.SUPPORTED_ARRAY :
                if isinstance(objs, numbers.Number):
                    objs = [objs]
                else :
                    raise ValueError(f"Invalid Objective : not a number")
            objs = np.array(objs).flatten()
        self._objectives = objs
    
    @property
    def weights(self) :
        return self._weights
    
    def set_weights(self, weights):
        if weights is None :
            self._weights = len(self.objectives)
        else :
            if type(weights) not in self.SUPPORTED_ARRAY:
                if isinstance(weights, numbers.Number) :
                    weights = [weights, ] * len(self.objectives)
                else :
                    raise ValueError("weight error")
            weights = np.array(weights).flatten()
        self._weights = weights
    
    @property
    def fitness(self):
        return self._fitness

    def calculate_fitness(self, weights: Union[List, Tuple, np.ndarray]) -> None :
        if not (type(weights) in self.SUPPORTED_ARRAY and len(weights) == len(self.objectives)) :
            weights = len(self.objectives) * (1.,)
        self._fitness = np.dot(weights, self.objectives)
    
    def __str__(self) :
        return f"Objectives : {self.objectives}, Fitness: {self.fitness}"

class Agent:
    ID = 0

    def __init__(self, solution: np.ndarray = None, target: Target = None, **kwargs) -> None :
        self.solution = solution
        self.target = target
        self.set_kwargs(kwargs)
        self.id = self.increase()

    @classmethod
    def increase(cls) -> int :
        cls.ID += 1
        return cls.ID

    def set_kwargs(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def copy(self) -> 'Agent' :
        agent = Agent(self.solution, self.target.copy(), **self.kwargs)
        for attr, value in vars(self).items():
            if attr not in ['target', 'solution', 'id', 'kwargs'] :
                setattr(agent, attr, value)
        return agent

    def __init__(self, solution: np.ndarray = None, target: Target = None, **kwargs) -> None :
        self.solution = solution
        self.target = target
        self.set_kwargs(kwargs)
        self.kwargs = kwargs
        self.id = self.increase()

    def update_agent(self, solution: np.ndarray, target: Target) -> None :
        self.solution = solution
        self.target = target

    def update(self, **kwargs) -> None :
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def get_better_solution(self, compared_agent: 'Agent', minmax:str = "min")-> 'Agent' :
        if minmax == "min" :
            return self if self.target.fitness < compared_agent.target.fitness else compared_agent
        else :
            return self if self.target.fitness > compared_agent.target.fitness else compared_agent
    
    def is_better_than(self, compared_agent: 'Agent', minmax: str = "min") -> bool :
        pred = self.target.fitness < compared_agent.target.fitness
        return pred if minmax == "min" else (not pred)

    def is_duplicate(self, compared_agent: 'Agent') -> bool :
        return np.all(self.solution - compared_agent.solution) == 0 

    def compare_duplicate(self, compared_agent: 'Agent') -> bool:
        if np.all(self.solution - compared_agent.solution) == 0 :
            self.target = compared_agent.target
            return True
        return False


    def __repr__(self):
        return f"id : {self.id}, target: {self.target}, solution: {self.solution}"

class History :
    def __init__(self, **kwargs):
        self.list_global_best = []  # List of global best solution found so far in all previous generations
        self.list_current_best = []  # List of current best solution in each previous generations
        self.list_epoch_time = []  # List of runtime for each generation
        self.list_global_best_fit = []  # List of global best fitness found so far in all previous generations
        self.list_current_best_fit = []  # List of current best fitness in each previous generations
        self.list_population = []  # List of population in each generation
        self.list_diversity = []  # List of diversity of swarm in all generations
        self.list_exploitation = []  # List of exploitation percentages for all generations
        self.list_exploration = []  # List of exploration percentages for all generations
        self.list_global_worst = [] # List of global worst solution found so far in all previous generations
        self.list_current_worst = [] # List of current worst solution in each previous generations
        self.epoch, self.log_to, self.log_file = None, None, None
        self.__set_keyword_arguments(kwargs)
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(name=f"{__name__}.{__class__.__name__}",
            format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def store_initial_best_worst(self, best_agent: Agent, worst_agent: Agent) -> None:
        self.list_global_best = [best_agent.copy()]
        self.list_current_best = [best_agent.copy()]
        self.list_global_worst = [worst_agent.copy()]
        self.list_current_worst = [worst_agent.copy()]

    def get_global_repeated_times(self, epsilon: float) -> int:
        count = 0
        for idx in range(0, len(self.list_global_best) - 1):
            temp = np.abs(self.list_global_best[idx].target.fitness - self.list_global_best[idx + 1].target.fitness)
            if temp <= epsilon:
                count += 1
            else:
                count = 0
        return count

class Problem:
    SUPPORTED_VARS = (IntegerVar, FloatVar, PermutationVar, StringVar, BinaryVar, BoolVar, MixedSetVar)
    SUPPORTED_ARRAYS = (list, tuple, np.ndarray)

    def __init__(self, bounds: Union[List, Tuple, np.ndarray, BaseVar], minmax: str = "min", **kwargs) -> None:
        self._bounds, self.lb, self.ub = None, None, None
        self.minmax = minmax
        self.seed = None
        self.name, self.log_to, self.log_file = "P", "console", "history.txt"
        self.n_objs, self.obj_weights = 1, None
        self.n_dims, self.save_population = None, False
        self.__set_keyword_arguments(kwargs)
        self.set_bounds(bounds)
        self.__set_functions()
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(
            name=f"{__name__}.{__class__.__name__}",
            format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')


    @property
    def bounds(self):
        return self._bounds

    def set_bounds(self, bounds):
        # print(type(bounds))
        if isinstance(bounds, BaseVar):
            bounds.seed = self.seed
            self._bounds = [bounds, ]
        elif type(bounds) in self.SUPPORTED_ARRAYS:
            self._bounds = []
            for bound in bounds:
                if isinstance(bound, BaseVar):
                    bound.seed = self.seed
                else:
                    raise ValueError(f"Invalid bounds. All variables in bounds should be an instance of {self.SUPPORTED_VARS}")
                self._bounds.append(bound)
        else:
            raise TypeError(f"Invalid bounds. It should be type of {self.SUPPORTED_ARRAYS} or an instance of {self.SUPPORTED_VARS}")
        self.lb = np.concatenate([bound.lb for bound in self._bounds])
        self.ub = np.concatenate([bound.ub for bound in self._bounds])

    def set_seed(self, seed: int = None) -> None:
        self.seed = seed
        for idx in range(len(self._bounds)):
            self._bounds[idx].seed = seed

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __set_functions(self):
        tested_solution = self.generate_solution(encoded=True)
        self.n_dims = len(tested_solution)
        result = self.obj_func(tested_solution)
        if type(result) in self.SUPPORTED_ARRAYS:
            result = np.array(result).flatten()
            self.n_objs = len(result)
            if self.n_objs > 1:
                if type(self.obj_weights) in self.SUPPORTED_ARRAYS:
                    self.obj_weights = np.array(self.obj_weights).flatten()
                    if self.n_objs != len(self.obj_weights):
                        raise ValueError(f"{self.n_objs}-objective problem, but N weights = {len(self.obj_weights)}.")
                    self.msg = f"Solving {self.n_objs}-objective optimization problem with weights: {self.obj_weights}."
                else:
                    raise ValueError(f"Solving {self.n_objs}-objective optimization, need to set obj_weights list with length: {self.n_objs}")
            elif self.n_objs == 1:
                self.obj_weights = np.ones(1)
                self.msg = f"Solving single objective optimization problem."
            else:
                raise ValueError(f"obj_func needs to return a single value or a list of values")
        elif isinstance(result, numbers.Number):
            self.obj_weights = np.ones(1)
            self.msg = f"Solving single objective optimization problem."
        else:
            raise ValueError(f"obj_func needs to return a single value or a list of values")

    def obj_func(self, x: np.ndarray) -> Union[List, Tuple, np.ndarray, int, float]:
        raise NotImplementedError

    def get_name(self) -> str:
        return self.name

    def get_class_name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def encode_solution_with_bounds(x, bounds):
        x_new = []
        for idx, var in enumerate(bounds):
            x_new += list(var.encode(x[idx]))
        return np.array(x_new)

    @staticmethod
    def decode_solution_with_bounds(x, bounds):
        x_new, n_vars = {}, 0
        for idx, var in enumerate(bounds):
            temp = var.decode(x[n_vars:n_vars + var.n_vars])
            if var.n_vars == 1:
                x_new[var.name] = temp[0]
            else:
                x_new[var.name] = temp
            n_vars += var.n_vars
        return x_new

    @staticmethod
    def correct_solution_with_bounds(x: Union[List, Tuple, np.ndarray], bounds: List) -> np.ndarray:
        x_new, n_vars = [], 0
        for idx, var in enumerate(bounds):
            x_new += list(var.correct(x[n_vars:n_vars+var.n_vars]))
            n_vars += var.n_vars
        return np.array(x_new)

    @staticmethod
    def generate_solution_with_bounds(bounds: Union[List, Tuple, np.ndarray], encoded: bool = True) -> Union[List, np.ndarray]:
        x = [var.generate() for var in bounds]
        if encoded:
            return Problem.encode_solution_with_bounds(x, bounds)
        return x

    def encode_solution(self, x: Union[List, tuple, np.ndarray]) -> np.ndarray:
        return self.encode_solution_with_bounds(x, self.bounds)

    def decode_solution(self, x: np.ndarray) -> Dict:
        return self.decode_solution_with_bounds(x, self.bounds)

    def correct_solution(self, x: np.ndarray) -> np.ndarray:
        return self.correct_solution_with_bounds(x, self.bounds)

    def generate_solution(self, encoded: bool = True) -> Union[List, np.ndarray]:
        return self.generate_solution_with_bounds(self.bounds, encoded)

    def get_target(self, solution: np.ndarray) -> Target:
        objs = self.obj_func(solution)
        return Target(objectives=objs, weights=self.obj_weights)

class Optimizer:
    EPSILON = 10E-10
    SUPPORTED_MODES = ["process","thread","swarm","single"]
    AVAILABLE_MODES = ["process","thread","swarm"]
    PARALLEL_MODES = [ "process","thread"]
    SUPPORTED_ARRAYS = [list,tuple, np.ndarray]

    def __init__(self, **kwargs):
        super(Optimizer, self).__init__()
        self.epoch, self.pop_size = None,None
        self.mode, self.n_workers, self.name = None, None, None
        self.pop, self.g_best, self.g_worst = None, Agent(), None
        self.problem, self.logger, self.history = None, None, None
        self.__set_keyword_arguments(kwargs)
        self.validator = Validator(log_to="console", log_file=None)

        if self.name is None :
            self.name = self.__class__.__name__
        self.sort_flag = False
        self.nfe_counter = -1
        self.parameters, self.params_name_ordered = {},None
        self.is_parallelizable = True

    def __set_keyword_arguments(self, kwargs) :
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def set_parameters(self, parameters: Union[List,Tuple,Dict]) -> None :
        if type(parameters) in (list, tuple) :
            self.params_name_ordered = tuple(parameters)
            self.parameters = {}
            for name in parameters :
                self.parameters[name] = self.__dict__[name]

        if type(parameters) is dict :
            valid_para_names = set(self.parameters.keys())
            new_para_names = set(parameters.keys())
            if new_para_names.issubset(valid_para_names) :
                for key, value in parameters.items() :
                    setattr(self, key, value)
                    self.parameters[key] = value
            else :
                raise ValueError(f"Invalid input param : {new_para_names} for {self.get_name()} optimizer. | Valid = {valid_para_names}")
            
    def get_parameters(self) -> Dict :
        return self.parameters
    
    def get_attributes(self) -> Dict :
        return self.__dict__
    
    def get_name(self) -> str :
        return self.name
    
    def before_main_loop(self):
        pass

    def evolve(self, epoch: int) -> None :
        pass

    def generate_group_population(self, pop: List[Agent], n_groups: int, m_agents: int) -> List:
        """
        Generate a list of group population from pop

        Args:
            pop: The current population
            n_groups: The n groups
            m_agents: The m agents in each group

        Returns:
            A list of group population
        """
        pop_group = []
        for idx in range(0, n_groups):
            group = pop[idx * m_agents: (idx + 1) * m_agents]
            pop_group.append([agent.copy() for agent in group])
        return pop_group

    def get_levy_flight_step(self, beta: float = 1.0, multiplier: float = 0.001, 
                             size: Union[List, Tuple, np.ndarray] = None, case: int = 0) -> Union[float, List, np.ndarray]:
        """
        Get the Levy-flight step size

        Args:
            beta (float): Should be in range [0, 2].

                * 0-1: small range --> exploit
                * 1-2: large range --> explore

            multiplier (float): default = 0.001
            size (tuple, list): size of levy-flight steps, for example: (3, 2), 5, (4, )
            case (int): Should be one of these value [0, 1, -1].

                * 0: return multiplier * s * self.generator.uniform()
                * 1: return multiplier * s * self.generator.normal(0, 1)
                * -1: return multiplier * s

        Returns:
            float, list, np.ndarray: The step size of Levy-flight trajectory
        """
        # u and v are two random variables which follow self.generator.normal distribution
        # sigma_u : standard deviation of u
        from math import gamma
        sigma_u = np.power(gamma(1. + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2.) * beta * np.power(2., (beta - 1) / 2)), 1. / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        size = 1 if size is None else size
        u = self.generator.normal(0, sigma_u, size)
        v = self.generator.normal(0, sigma_v, size)
        s = u / np.power(np.abs(v) + self.EPSILON, 1 / beta)
        if case == 0:
            step = multiplier * s * self.generator.uniform()
        elif case == 1:
            step = multiplier * s * self.generator.normal(0, 1)
        else:
            step = multiplier * s
        return step[0] if size == 1 else step

    def generate_population(self, pop_size: int = None) -> List[Agent]:
        if pop_size is None:
            pop_size = self.pop_size
        pop = [self.generate_agent() for _ in range(0, pop_size)]
        return pop

    def get_index_roulette_wheel_selection(self, list_fitness: np.array):
        
        if isinstance(list_fitness, (list, tuple, np.ndarray)):
            list_fitness = np.array(list_fitness).ravel()
            
        if np.ptp(list_fitness) == 0:
            return int(self.generator.integers(0, len(list_fitness)))
        
        if np.any(list_fitness < 0):
            list_fitness = list_fitness - np.min(list_fitness)
            
        final_fitness = list_fitness
        if self.problem.minmax == "min":
            final_fitness = np.max(list_fitness) - list_fitness
            
        prob = final_fitness / np.sum(final_fitness)
        return int(self.generator.choice(range(0, len(list_fitness)), p=prob))
    
    @staticmethod
    def get_special_agents(pop: List[Agent] = None, n_best: int = 3, n_worst: int = 3,
                           minmax: str = "min") -> Tuple[List[Agent], Union[List[Agent], None], Union[List[Agent], None]]:
        pop = Optimizer.get_sorted_population(pop, minmax)
        if n_best is None:
            if n_worst is None:
                return pop, None, None
            else:
                return pop, None, [agent.copy() for agent in pop[::-1][:n_worst]]
        else:
            if n_worst is None:
                return pop, [agent.copy() for agent in pop[:n_best]], None
            else:
                return pop, [agent.copy() for agent in pop[:n_best]], [agent.copy() for agent in pop[::-1][:n_worst]]
    
    @staticmethod
    def get_sorted_population(pop: List[Agent], minmax: str = "min", return_index: bool = False) -> List[Agent]:
        print(pop)
        list_fits = [agent.target.fitness for agent in pop]
        indices = np.argsort(list_fits).tolist()
        if minmax == "max":
            indices = indices[::-1]
        pop_new = [pop[idx] for idx in indices]
        if return_index:
            return pop_new, indices
        else:
            return pop_new

    def initialize_variables(self):
        pass
    
    def update_target_for_population(self, pop: List[Agent] = None) -> List[Agent]:
        pos_list = [agent.solution for agent in pop]
        if self.mode == "thread":
            with parallel.ThreadPoolExecutor(self.n_workers) as executor:
                # Return result as original order, not the future object
                list_results = executor.map(partial(self.get_target, counted=False), pos_list)
                for idx, target in enumerate(list_results):
                    pop[idx].target = target
        elif self.mode == "process":
            with parallel.ProcessPoolExecutor(self.n_workers) as executor:
                # Return result as original order, not the future object
                list_results = executor.map(partial(self.get_target, counted=False), pos_list)
                for idx, target in enumerate(list_results):
                    pop[idx].target = target
        elif self.mode == "swarm":
            for idx, pos in enumerate(pos_list):
                pop[idx].target = self.get_target(pos, counted=False)
        else:
            return pop
        self.nfe_counter += len(pop)
        return pop
    
    def before_initialization(self, starting_solutions: Union[List, Tuple, np.ndarray] = None) -> None:
        if starting_solutions is None:
            pass
        elif type(starting_solutions) in self.SUPPORTED_ARRAYS and len(starting_solutions) == self.pop_size:
            if type(starting_solutions[0]) in self.SUPPORTED_ARRAYS and len(starting_solutions[0]) == self.problem.n_dims:
                if self.mode in self.AVAILABLE_MODES:
                    self.pop = [self.generate_empty_agent(solution) for solution in starting_solutions]
                    self.pop = self.update_target_for_population(self.pop)
                else:
                    self.pop = [self.generate_agent(solution) for solution in starting_solutions]
            else:
                raise ValueError("Invalid starting_solutions. It should be a list of positions or 2D matrix of positions only.")
        else:
            raise ValueError("Invalid starting_solutions. It should be a list/2D matrix of positions with same length as pop_size.")

    def initialization(self) -> None:
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

    def after_initialization(self) -> None:
        # The initial population is sorted or not depended on algorithm's strategy
        pop_temp, best, worst = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        self.g_best, self.g_worst = best[0], worst[0]
        if self.sort_flag: self.pop = pop_temp
        ## Store initial best and worst solutions
        self.history.store_initial_best_worst(self.g_best, self.g_worst)

    def before_main_loop(self):
        pass

    
    def check_mode_and_workers(self, mode, n_workers):
        self.mode = self.validator.check_str("mode", mode, self.SUPPORTED_MODES)
        if self.mode in self.PARALLEL_MODES:
            if not self.is_parallelizable:
                self.logger.warning(f"{self.get_name()} doesn't support parallelization. The default mode 'single' is activated.")
                self.mode = "single"
            elif n_workers is not None:
                if self.mode == "process":
                    self.n_workers = self.validator.check_int("n_workers", n_workers, [2, min(61, os.cpu_count() - 1)])
                if self.mode == "thread":
                    self.n_workers = self.validator.check_int("n_workers", n_workers, [2, min(32, os.cpu_count() + 4)])
                self.logger.info(f"The parallel mode '{self.mode}' is selected with {self.n_workers} workers.")
            else:
                self.logger.warning(f"The parallel mode: {self.mode} is selected. But n_workers is not set. The default n_workers = 4 is used.")
                self.n_workers = 4

    def check_termination(self, mode="start", termination=None, epoch=None):
        if mode == "start":
            self.termination = termination
            if termination is not None:
                if isinstance(termination, Termination):
                    self.termination = termination
                elif type(termination) == dict:
                    self.termination = Termination(log_to=self.problem.log_to, log_file=self.problem.log_file, **termination)
                else:
                    raise ValueError("Termination needs to be a dict or an instance of Termination class.")
                self.nfe_counter = 0
                self.termination.set_start_values(0, self.nfe_counter, time.perf_counter(), 0)
        else:
            finished = False
            if self.termination is not None:
                es = self.history.get_global_repeated_times(self.termination.epsilon)
                finished = self.termination.should_terminate(epoch, self.nfe_counter, time.perf_counter(), es)
                if finished:
                    self.logger.warning(self.termination.message)
            return finished

    def solve(self, problem: Union[Dict, Problem] = None, mode: str = 'single', n_workers: int = None,
              termination: Dict = None, starting_solutions: Union[List, np.ndarray, Tuple] = None,
              seed: int = None) -> Agent:
        self.check_problem(problem, seed)
        self.check_mode_and_workers(mode, n_workers)
        self.check_termination("start", termination, None)
        self.initialize_variables()

        self.before_initialization(starting_solutions)
        self.initialization()
        self.after_initialization()

        self.before_main_loop()

        # Check tqdm
        use_tqdm = self.problem.log_to == "console"
        loop = range(1, self.epoch + 1)
        if use_tqdm:
            desc = f"{self.__module__}.{self.__class__.__name__}"
            loop = tqdm(loop, desc=desc, unit="epoch")

        for epoch in loop:
            time_epoch = time.perf_counter()

            ## Evolve method will be called in child class
            self.evolve(epoch)

            # Update global best solution, the population is sorted or not depended on algorithm's strategy
            pop_temp, self.g_best = self.update_global_best_agent(self.pop)
            if self.sort_flag: self.pop = pop_temp

            time_epoch = time.perf_counter() - time_epoch
            self.track_optimize_step(self.pop, epoch, time_epoch)

            # update tqdm postfix 
            if use_tqdm:
                loop.set_postfix({
                    "c_best": f"{self.history.list_current_best[-1].target.fitness:.6f}",
                    "g_best": f"{self.g_best.target.fitness:.6f}"
                })

            if self.check_termination("end", None, epoch):
                break
        self.track_optimize_process()
        return self.g_best

    def check_problem(self, problem, seed) -> None:
        if isinstance(problem, Problem):
            problem.set_seed(seed)
            self.problem = problem
        elif type(problem) == dict:
            problem["seed"] = seed
            self.problem = Problem(**problem)
        else:
            raise ValueError("problem needs to be a dict or an instance of Problem class.")
        self.generator = np.random.default_rng(seed)
        self.rng = random.Random(seed)  # local RNG for random module
        self.logger = Logger(self.problem.log_to, log_file=self.problem.log_file).create_logger(name=f"{self.__module__}.{self.__class__.__name__}")
        self.logger.info(self)
        self.history = History(log_to=self.problem.log_to, log_file=self.problem.log_file)
        self.pop, self.g_best, self.g_worst = None, None, None
    
    def track_optimize_step(self, pop: List[Agent] = None, epoch: int = None, runtime: float = None) -> None:
        """
        Save some historical data and print out the detailed information of training process in each epoch

        Args:
            pop: the current population
            epoch: current iteration
            runtime: the runtime for current iteration
        """
        ## Save history data
        if self.problem.save_population:
            self.history.list_population.append(Optimizer.duplicate_pop(pop))
        self.history.list_epoch_time.append(runtime)
        self.history.list_global_best_fit.append(self.history.list_global_best[-1].target.fitness)
        self.history.list_current_best_fit.append(self.history.list_current_best[-1].target.fitness)
        # Save the exploration and exploitation data for later usage
        pos_matrix = np.array([agent.solution for agent in pop])
        div = np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0)
        self.history.list_diversity.append(np.mean(div, axis=0))
        ## Print epoch
        self.logger.info(f">>>Problem: {self.problem.name}, Epoch: {epoch}, Current best: {self.history.list_current_best[-1].target.fitness}, "
                         f"Global best: {self.history.list_global_best[-1].target.fitness}, Runtime: {runtime:.5f} seconds")

    def track_optimize_process(self) -> None:
        """
        Save some historical data after training process finished
        """
        self.history.epoch = len(self.history.list_diversity)
        div_max = np.max(self.history.list_diversity)
        self.history.list_exploration = 100 * (np.array(self.history.list_diversity) / div_max)
        self.history.list_exploitation = 100 - self.history.list_exploration
        self.history.list_global_best = self.history.list_global_best[1:]
        self.history.list_current_best = self.history.list_current_best[1:]
        self.history.list_global_worst = self.history.list_global_worst[1:]
        self.history.list_current_worst = self.history.list_current_worst[1:]

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        """
        Generate new agent with solution

        Args:
            solution (np.ndarray): The solution
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        return Agent(solution=solution)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        """
        Generate new agent with full information

        Args:
            solution (np.ndarray): The solution
        """
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        return agent

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        return solution

    def correct_solution(self, solution: np.ndarray) -> np.ndarray:
        solution = self.amend_solution(solution)
        return self.problem.correct_solution(solution)

    def get_target(self, solution: np.ndarray, counted: bool = True) -> Target:
        if counted:
            self.nfe_counter += 1
        return self.problem.get_target(solution)

    @staticmethod
    def compare_target(target_x: Target, target_y: Target, minmax: str = "min") -> bool:
        if minmax == "min":
            return True if target_x.fitness < target_y.fitness else False
        else:
            return False if target_x.fitness < target_y.fitness else True

    @staticmethod
    def compare_fitness(fitness_x: Union[float, int], fitness_y: Union[float, int], minmax: str = "min") -> bool:
        if minmax == "min":
            return True if fitness_x < fitness_y else False
        else:
            return False if fitness_x < fitness_y else True

    @staticmethod
    def duplicate_pop(pop: List[Agent]) -> List[Agent]:
        return [agent.copy() for agent in pop]

    @staticmethod
    def get_sorted_population(pop: List[Agent], minmax: str = "min", return_index: bool = False) -> List[Agent]:

        list_fits = [agent.target.fitness for agent in pop]
        indices = np.argsort(list_fits).tolist()
        if minmax == "max":
            indices = indices[::-1]
        pop_new = [pop[idx] for idx in indices]
        if return_index:
            return pop_new, indices
        else:
            return pop_new

    @staticmethod
    def get_best_agent(pop: List[Agent], minmax: str = "min") -> Agent:
        pop = Optimizer.get_sorted_population(pop, minmax)
        return pop[0].copy()

    @staticmethod
    def get_index_best(pop: List[Agent], minmax: str = "min") -> int:
        fit_list = np.array([agent.target.fitness for agent in pop])
        if minmax == "min":
            return np.argmin(fit_list)
        else:
            return np.argmax(fit_list)

    @staticmethod
    def get_worst_agent(pop: List[Agent], minmax: str = "min") -> Agent:
        pop = Optimizer.get_sorted_population(pop, minmax)
        return pop[-1].copy()

    @staticmethod
    def get_special_agents(pop: List[Agent] = None, n_best: int = 3, n_worst: int = 3,
                           minmax: str = "min") -> Tuple[List[Agent], Union[List[Agent], None], Union[List[Agent], None]]:
        pop = Optimizer.get_sorted_population(pop, minmax)
        if n_best is None:
            if n_worst is None:
                return pop, None, None
            else:
                return pop, None, [agent.copy() for agent in pop[::-1][:n_worst]]
        else:
            if n_worst is None:
                return pop, [agent.copy() for agent in pop[:n_best]], None
            else:
                return pop, [agent.copy() for agent in pop[:n_best]], [agent.copy() for agent in pop[::-1][:n_worst]]

    @staticmethod
    def get_special_fitness(pop: List[Agent] = None, minmax: str = "min") -> Tuple[Union[float, np.ndarray], float, float]:

        total_fitness = np.sum([agent.target.fitness for agent in pop])
        pop = Optimizer.get_sorted_population(pop, minmax)
        return total_fitness, pop[0].target.fitness, pop[-1].target.fitness

    @staticmethod
    def get_better_agent(agent_x: Agent, agent_y: Agent, minmax: str = "min", reverse: bool = False) -> Agent:

        minmax_dict = {"min": 0, "max": 1}
        idx = minmax_dict[minmax]
        if reverse:
            idx = 1 - idx
        if idx == 0:
            return agent_x.copy() if agent_x.target.fitness < agent_y.target.fitness else agent_y.copy()
        else:
            return agent_y.copy() if agent_x.target.fitness < agent_y.target.fitness else agent_x.copy()

    @staticmethod
    def greedy_selection_population(pop_old: List[Agent] = None, pop_new: List[Agent] = None, minmax: str = "min") -> List[Agent]:

        len_old, len_new = len(pop_old), len(pop_new)
        if len_old != len_new:
            raise ValueError("Greedy selection of two population with different length.")
        if minmax == "min":
            return [pop_new[idx] if pop_new[idx].target.fitness < pop_old[idx].target.fitness else pop_old[idx] for idx in range(len_old)]
        else:
            return [pop_new[idx] if pop_new[idx].target.fitness > pop_old[idx].target.fitness else pop_old[idx] for idx in range(len_old)]

    @staticmethod
    def get_sorted_and_trimmed_population(pop: List[Agent] = None, pop_size: int = None, minmax: str = "min") -> List[Agent]:

        pop = Optimizer.get_sorted_population(pop, minmax)
        return pop[:pop_size]

    def update_global_best_agent(self, pop: List[Agent], save: bool = True) -> Union[List, Tuple]:

        sorted_pop = self.get_sorted_population(pop, self.problem.minmax)
        c_best, c_worst = sorted_pop[0], sorted_pop[-1]
        if save:
            ## Save current best
            self.history.list_current_best.append(c_best)
            better = self.get_better_agent(c_best, self.history.list_global_best[-1], self.problem.minmax)
            self.history.list_global_best.append(better)
            ## Save current worst
            self.history.list_current_worst.append(c_worst)
            worse = self.get_better_agent(c_worst, self.history.list_global_worst[-1], self.problem.minmax, reverse=True)
            self.history.list_global_worst.append(worse)
            return sorted_pop, better
        else:
            ## Handle current best
            local_better = self.get_better_agent(c_best, self.history.list_current_best[-1], self.problem.minmax)
            self.history.list_current_best[-1] = local_better
            global_better = self.get_better_agent(c_best, self.history.list_global_best[-1], self.problem.minmax)
            self.history.list_global_best[-1] = global_better
            ## Handle current worst
            local_worst = self.get_better_agent(c_worst, self.history.list_current_worst[-1], self.problem.minmax, reverse=True)
            self.history.list_current_worst[-1] = local_worst
            global_worst = self.get_better_agent(c_worst, self.history.list_global_worst[-1], self.problem.minmax, reverse=True)
            self.history.list_global_worst[-1] = global_worst
            return sorted_pop, global_better

    def __str__(self):
        temp = ""
        for key in self.params_name_ordered :
            temp += f"{key} = {self.parameters[key]}, "
        temp = temp[:-2]
        return f"{self.__class__.__name__}({temp})"

class LabelEncoder:
    """
    Encode categorical features as integer labels.
    Especially, it can encode a list of mixed types include integer, float, and string. Better than scikit-learn module.
    """

    def __init__(self):
        self.unique_labels = None
        self.label_to_index = {}

    def set_y(self, y):
        if type(y) not in (list, tuple, np.ndarray):
            y = (y,)
        return y

    def fit(self, y):
        """
        Fit label encoder to a given set of labels.

        Parameters:
        -----------
        y : list, tuple
            Labels to encode.
        """
        def safe_key(val):
            # Chuyển None -> 0, số -> 1, chuỗi -> 2, object khác -> 3
            if val is None:
                return (0, '')
            elif isinstance(val, nb.Number):
                return (1, val)
            elif isinstance(val, str):
                return (2, val)
            else:
                return (3, str(val))

        # self.unique_labels = sorted(set(y), key=lambda x: (isinstance(x, (int, float)), x))
        self.unique_labels = sorted(set(y), key=safe_key)
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}
        return self

    def transform(self, y):
        """
        Transform labels to encoded integer labels.

        Parameters:
        -----------
        y : list, tuple
            Labels to encode.

        Returns:
        --------
        encoded_labels : list
            Encoded integer labels.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = self.set_y(y)
        return [self.label_to_index[label] for label in y]

    def fit_transform(self, y):
        y = self.set_y(y)
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = self.set_y(y)
        return [self.unique_labels[i] if i in self.label_to_index.values() else "unknown" for i in y]

import numbers
import numpy as np
from typing import Union, List, Tuple, Dict
import sys
sys.path.append("..")


#!/usr/bin/env python
# Created by "Thieu" at 21:32, 14/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import operator
from numbers import Number

SEQUENCE = (list, tuple, np.ndarray)
DIGIT = (int, np.integer)
REAL = (float, np.floating)


def is_in_bound(value, bound):
    ops = None
    if type(bound) is tuple:
        ops = operator.lt
    elif type(bound) is list:
        ops = operator.le
    if bound[0] == float("-inf") and bound[1] == float("inf"):
        return True
    elif bound[0] == float("-inf") and ops(value, bound[1]):
        return True
    elif ops(bound[0], value) and bound[1] == float("inf"):
        return True
    elif ops(bound[0], value) and ops(value, bound[1]):
        return True
    return False


def is_str_in_list(value: str, my_list: list):
    if type(value) == str and my_list is not None:
        return True if value in my_list else False
    return False


class Validator:
    def __init__(self, **kwargs):
        self.log_to, self.log_file = None, None
        self.__set_keyword_arguments(kwargs)
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(name=f"{__name__}.{__class__.__name__}",
            format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')
        self.logger.propagate = False

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def check_int(self, name:str, value: int, bound=None):
        if isinstance(value, Number):
            if bound is None:
                return int(value)
            elif is_in_bound(value, bound):
                return int(value)
        bound = "" if bound is None else f"and value should be in range: {bound}"
        raise ValueError(f"'{name}' is an integer {bound}.")

    def check_float(self, name: str, value: float, bound=None):
        if isinstance(value, Number):
            if bound is None:
                return float(value)
            elif is_in_bound(value, bound):
                return float(value)
        bound = "" if bound is None else f"and value should be in range: {bound}"
        raise ValueError(f"'{name}' is a float {bound}.")

    def check_str(self, name: str, value: str, bound=None):
        if type(value) is str:
            if bound is None or is_str_in_list(value, bound):
                return value
        bound = "" if bound is None else f"and value should be one of this: {bound}"
        raise ValueError(f"'{name}' is a string {bound}.")

    def check_bool(self, name: str, value: bool, bound=(True, False)):
        if type(value) is bool:
            if value in bound:
                return value
        bound = "" if bound is None else f"and value should be one of this: {bound}"
        raise ValueError(f"'{name}' is a boolean {bound}.")

    def check_tuple_int(self, name: str, values: tuple, bounds=None):
        if isinstance(values, SEQUENCE) and len(values) > 1:
            value_flag = [isinstance(item, DIGIT) for item in values]
            if np.all(value_flag):
                if bounds is not None and len(bounds) == len(values):
                    value_flag = [is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                    if np.all(value_flag):
                        return values
                else:
                    return values
        bounds = "" if bounds is None else f"and values should be in range: {bounds}"
        raise ValueError(f"'{name}' are integer {bounds}.")

    def check_tuple_float(self, name: str, values: tuple, bounds=None):
        if isinstance(values, SEQUENCE) and len(values) > 1:
            value_flag = [isinstance(item, Number) for item in values]
            if np.all(value_flag):
                if bounds is not None and len(bounds) == len(values):
                    value_flag = [is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                    if np.all(value_flag):
                        return values
                else:
                    return values
        bounds = "" if bounds is None else f"and values should be in range: {bounds}"
        raise ValueError(f"'{name}' are float {bounds}.")

    def check_list_tuple(self, name: str, value: any, data_type: str):
        if type(value) in (tuple, list) and len(value) >= 1:
            return list(value)
        raise ValueError(f"'{name}' should be a list or tuple of {data_type}, and length >= 1.")

    def check_is_instance(self, name: str, value: any, class_type: any):
        if isinstance(value, class_type):
            return value
        raise ValueError(f"'{name}' should be an instance of {class_type} class.")

    def check_is_int_and_float(self, name: str, value: any, bound_int=None, bound_float=None):
        if type(value) is int:
            if bound_int is None or is_in_bound(value, bound_int):
                return int(value)
        bound_int_str = "" if bound_int is None else f"and value in range: {bound_int}"
        if type(value) is float:
            if bound_float is None or is_in_bound(value, bound_float):
                return float(value)
        bound_float_str = "" if bound_float is None else f"and value in range: {bound_float}"
        raise ValueError(f"'{name}' can be int {bound_int_str}, or float {bound_float_str}.")

import logging
class Logger:
    def __init__(self, log_to="console", **kwargs):
        self.log_to = log_to
        self.log_file = None
        self.log_file_mode = "a"
        self.__set_keyword_arguments(kwargs)
        self.default_formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
        self.default_logfile = "optimizer.log"

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def create_logger(self, name=__name__, format_str=None):
        logger = logging.getLogger(name)
        if self.log_to == "console":
            logger.setLevel(logging.INFO)
            if format_str is None:
                formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
            else:
                formatter = logging.Formatter(format_str, datefmt="%Y/%m/%d %I:%M:%S %p")
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
        elif self.log_to == "file":
            logger.setLevel(logging.DEBUG)
            if format_str is None:
                formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
            else:
                formatter = logging.Formatter(format_str, datefmt="%Y/%m/%d %I:%M:%S %p")
            if self.log_file is None:
                self.log_file = self.default_logfile
            handler = logging.FileHandler(self.log_file, mode=self.log_file_mode)
            handler.setFormatter(formatter)
        else:
            logger.setLevel(logging.ERROR)
            if format_str is None:
                formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
            else:
                formatter = logging.Formatter(format_str, datefmt="%Y/%m/%d %I:%M:%S %p")
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(handler)
        return logger
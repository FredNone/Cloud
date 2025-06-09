# Hey, welcome.
import numpy as np
import time

class ACLHO_3_2:
    def __init__(self, environment, config):
        self.env = environment
        self.config = config
        self.population_size = config['population_size']
        self.max_iterations = config['max_iterations']
        self.num_tasks = environment.num_tasks
        self.num_robots = environment.num_robots
        self.history_pool = []
        self._cached_state = None
        self.temperature = 1.0  # 退火温度

    def _initialize_population(self):
        return np.random.randint(0, self.num_robots, size=(self.population_size, self.num_tasks))

    def _levy_flight(self, solution, alpha=0.01):
        levy = np.random.standard_cauchy(size=solution.shape)
        perturbed = solution + alpha * levy
        return np.clip(np.round(perturbed), 0, self.num_robots - 1).astype(int)

    def _interpolate_elites(self):
        if len(self.history_pool) < 2:
            return None
        a, b = np.random.choice(list(self.history_pool), 2, replace=False)
        lam = np.random.rand()
        return np.clip(np.round(lam * a + (1 - lam) * b), 0, self.num_robots - 1).astype(int)

    def _run_iteration(self):
        if self._cached_state is None:
            population = self._initialize_population()
            fitness = np.array([self.env.calculate_fitness(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx].copy()
            best_fitness = fitness[best_idx]
            iteration = 0
            self._cached_state = (population, fitness, best_solution, best_fitness, iteration)
        else:
            population, fitness, best_solution, best_fitness, iteration = self._cached_state

        new_population = []
        fitness_trend = np.polyfit(np.arange(len(fitness)), fitness, deg=1)[0] if len(fitness) >= 3 else 0

        for i in range(self.population_size):
            curr = population[i].copy()
            step_scale = 0.2 if fitness_trend < 0 else 0.05
            noise = np.random.randn(self.num_tasks) * step_scale * self.num_robots
            candidate = np.clip(np.round(curr + noise), 0, self.num_robots - 1).astype(int)

            # Levy飞行扰动
            if np.random.rand() < (0.2 + 0.3 * np.sin(np.pi * iteration / self.max_iterations)):
                candidate = self._levy_flight(curr, alpha=0.02)

            # 与历史精英交叉
            if np.random.rand() < 0.1 and self.history_pool:
                elite = self.history_pool[np.random.randint(len(self.history_pool))]
                cross_point = np.random.randint(0, self.num_tasks)
                candidate[:cross_point] = elite[:cross_point]

            # 插值局部搜索
            if np.random.rand() < 0.2:
                interpolated = self._interpolate_elites()
                if interpolated is not None:
                    candidate = interpolated

            # 评估新解
            candidate_fitness = self.env.calculate_fitness(candidate)

            # Metropolis准则退火接受劣解
            delta = candidate_fitness - fitness[i]
            if delta < 0 or np.random.rand() < np.exp(-delta / max(self.temperature, 1e-8)):
                new_population.append(candidate)
                fitness[i] = candidate_fitness
                if candidate_fitness < best_fitness:
                    best_fitness = candidate_fitness
                    best_solution = candidate.copy()
                    if len(self.history_pool) < 10:
                        self.history_pool.append(candidate.copy())
            else:
                new_population.append(curr)

        self.temperature *= 0.98  # 降温

        updated_population = np.array(new_population)
        updated_fitness = np.array([self.env.calculate_fitness(ind) for ind in updated_population])
        self._cached_state = (updated_population, updated_fitness, best_solution, best_fitness, iteration + 1)

        return best_solution, best_fitness

    def optimize(self):
        population = self._initialize_population()
        fitness = np.array([self.env.calculate_fitness(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        self._cached_state = (population, fitness, best_solution, best_fitness, 0)
        self.temperature = 1.0

        history = [best_fitness]
        iteration_times = []
        total_start = time.time()

        for _ in range(self.max_iterations):
            start = time.time()
            best_solution, best_fitness = self._run_iteration()
            history.append(best_fitness)
            iteration_times.append(time.time() - start)

        total_time = time.time() - total_start
        return best_solution, best_fitness, history, iteration_times, total_time

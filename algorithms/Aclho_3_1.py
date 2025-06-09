import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor

class ACLHO_3_1:
    def __init__(self, environment, config):
        self.env = environment
        self.config = config
        self.population_size = config['population_size']
        self.max_iterations = config['max_iterations']
        self.num_tasks = environment.num_tasks
        self.num_robots = environment.num_robots
        self.history_pool = []
        self._cached_state = None  # 缓存当前群体状态

        # Surrogate model parameters
        self.use_surrogate = True
        self.surrogate_model = KNeighborsRegressor(n_neighbors=3)
        self.surrogate_training_X = []
        self.surrogate_training_y = []

    def _initialize_population(self):
        population = np.random.randint(0, self.num_robots, size=(self.population_size, self.num_tasks))
        return population

    def _levy_flight(self, solution, alpha=0.01):
        levy = np.random.standard_cauchy(size=solution.shape)
        perturb = solution + alpha * levy
        return np.clip(np.round(perturb), 0, self.num_robots - 1).astype(int)

    def _interpolate_elites(self):
        if len(self.history_pool) < 2:
            return None
        a, b = np.random.choice(self.history_pool, 2, replace=False)
        lam = np.random.rand()
        interp = np.round(lam * a + (1 - lam) * b).astype(int)
        return np.clip(interp, 0, self.num_robots - 1)

    def _update_surrogate_model(self):
        if len(self.surrogate_training_X) >= 20:
            self.surrogate_model.fit(self.surrogate_training_X, self.surrogate_training_y)

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
            perturbation = np.random.randn(self.num_tasks) * step_scale * self.num_robots
            candidate = np.clip(np.round(curr + perturbation), 0, self.num_robots - 1).astype(int)

            if np.random.rand() < (0.2 + 0.3 * np.sin(np.pi * iteration / self.max_iterations)):
                candidate = self._levy_flight(curr, alpha=0.02)

            if np.random.rand() < 0.1 and self.history_pool:
                elite = self.history_pool[np.random.randint(len(self.history_pool))]
                cross_point = np.random.randint(0, self.num_tasks)
                candidate[:cross_point] = elite[:cross_point]

            if np.random.rand() < 0.2:
                interpolated = self._interpolate_elites()
                if interpolated is not None:
                    candidate = interpolated

            # ✨ Surrogate model prediction + selective evaluation
            use_real = True
            if self.use_surrogate and len(self.surrogate_training_X) >= 20:
                pred_fitness = self.surrogate_model.predict([candidate])[0]
                if pred_fitness > np.max(fitness):
                    use_real = False
                    candidate_fitness = pred_fitness + 0.5  # 惩罚预测
                else:
                    candidate_fitness = self.env.calculate_fitness(candidate)
            else:
                candidate_fitness = self.env.calculate_fitness(candidate)

            if use_real:
                self.surrogate_training_X.append(candidate.tolist())
                self.surrogate_training_y.append(candidate_fitness)

            if candidate_fitness < fitness[i]:
                new_population.append(candidate)
                if candidate_fitness < best_fitness:
                    best_fitness = candidate_fitness
                    best_solution = candidate.copy()
                    if len(self.history_pool) < 10:
                        self.history_pool.append(candidate.copy())
            else:
                new_population.append(curr)

        updated_population = np.array(new_population)
        updated_fitness = np.array([self.env.calculate_fitness(ind) for ind in updated_population])
        self._cached_state = (updated_population, updated_fitness, best_solution, best_fitness, iteration + 1)

        # 定期更新代理模型
        if iteration % 5 == 0:
            self._update_surrogate_model()

        return best_solution, best_fitness

    def optimize(self):
        population = self._initialize_population()
        fitness = np.array([self.env.calculate_fitness(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        self._cached_state = (population, fitness, best_solution, best_fitness, 0)

        history = [best_fitness]
        iteration_times = []
        total_start = time.time()

        for iteration in range(self.max_iterations):
            iter_start = time.time()
            best_solution, best_fitness = self._run_iteration()
            iteration_times.append(time.time() - iter_start)
            history.append(best_fitness)

        total_time = time.time() - total_start
        return best_solution, best_fitness, history, iteration_times, total_time

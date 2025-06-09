import numpy as np
from scipy.special import gamma

class ACLHO_1:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.population_size = config['population_size']
        self.max_iterations = config['max_iterations']
        self.num_tasks = env.num_tasks
        self.num_robots = env.num_robots

        # 初始化种群位置和速度
        self.population = np.random.randint(-1, self.num_robots, (self.population_size, self.num_tasks))
        self.velocities = np.zeros((self.population_size, self.num_tasks))

        # 初始化个体最优和全局最优
        self.pbest = self.population.copy()
        self.pbest_fitness = np.array([np.inf] * self.population_size)
        self.gbest = None
        self.gbest_fitness = np.inf

        # 控制参数
        self.w = 0.6  # 惯性权重
        self.c1 = 1.5  # 个体引导因子
        self.c2 = 1.5  # 全局引导因子

        self.current_iteration = 0
        self.history = []
        self.no_improve_counter = 0
        self.patience = 5  # 自适应扰动耐心

    def _levy(self, dim, beta=1.5):
        sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(dim) * sigma_u
        v = np.random.randn(dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def _run_iteration(self):
        improved = False

        for i in range(self.population_size):
            solution = self.population[i]
            fitness = self.env.calculate_fitness(solution)

            if fitness < self.pbest_fitness[i]:
                self.pbest[i] = solution.copy()
                self.pbest_fitness[i] = fitness
                improved = True

            if fitness < self.gbest_fitness:
                self.gbest = solution.copy()
                self.gbest_fitness = fitness
                improved = True

        # 记录历史
        self.history.append(self.gbest_fitness)

        # PSO引导更新 + Lévy扰动融合
        for i in range(self.population_size):
            r1 = np.random.rand(self.num_tasks)
            r2 = np.random.rand(self.num_tasks)

            cognitive = self.c1 * r1 * (self.pbest[i] - self.population[i])
            social = self.c2 * r2 * (self.gbest - self.population[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive + social

            # Lévy跳跃扰动
            levy = self._levy(self.num_tasks)
            new_position = self.population[i] + self.velocities[i] + levy
            new_position = np.round(new_position).astype(int)
            new_position = np.clip(new_position, -1, self.num_robots - 1)

            self.population[i] = new_position

        # 自适应扰动（若连续未提升）
        if not improved:
            self.no_improve_counter += 1
            if self.no_improve_counter >= self.patience:
                for i in range(self.population_size):
                    self.population[i] += np.random.randint(-1, 2, self.num_tasks)
                    self.population[i] = np.clip(self.population[i], -1, self.num_robots - 1)
                self.no_improve_counter = 0
        else:
            self.no_improve_counter = 0

        self.current_iteration += 1
        return self.gbest, self.gbest_fitness

    def optimize(self):
        self.current_iteration = 0
        for _ in range(self.max_iterations):
            best_solution, best_fitness = self._run_iteration()
            if (_ + 1) % 10 == 0:
                print(f"[ACLHO++] Iteration {_ + 1}/{self.max_iterations}, Fitness: {best_fitness:.4f}")
        return self.gbest, self.gbest_fitness

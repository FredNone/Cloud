import numpy as np


class ACLHO_BASE:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        self.population_size = config['population_size']
        self.max_iterations = config['max_iterations']
        self.num_tasks = env.num_tasks
        self.num_robots = env.num_robots

        # 初始化种群（随机分配任务到机器人，-1表示任务未分配）
        self.population = np.random.randint(-1, self.num_robots,
                                            size=(self.population_size, self.num_tasks))

        # 最优解与适应度
        self.best_solution = None
        self.best_fitness = float('inf')

        self.current_iteration = 0
        self.history = []

    def _run_iteration(self):
        fitness_values = np.array([self.env.calculate_fitness(sol) for sol in self.population])

        # 更新最优解
        min_idx = np.argmin(fitness_values)
        if fitness_values[min_idx] < self.best_fitness:
            self.best_fitness = fitness_values[min_idx]
            self.best_solution = self.population[min_idx].copy()

        # 简单启发式更新操作，示例：基于最优解进行扰动生成新解
        new_population = []
        for sol in self.population:
            new_sol = self._perturb_solution(sol, self.best_solution)
            new_population.append(new_sol)
        self.population = np.array(new_population)

        self.current_iteration += 1
        return self.best_solution, self.best_fitness

    def _perturb_solution(self, sol, best_sol):
        """对解进行轻微扰动，向最优解靠近"""
        new_sol = sol.copy()
        # 随机选择若干任务位置进行替换
        num_changes = max(1, int(0.1 * self.num_tasks))  # 10%任务可变
        change_indices = np.random.choice(self.num_tasks, num_changes, replace=False)
        for idx in change_indices:
            # 以一定概率直接采纳best_sol对应任务的机器人分配
            if np.random.rand() < 0.7:
                new_sol[idx] = best_sol[idx]
            else:
                # 随机分配一个机器人（包括-1未分配）
                new_sol[idx] = np.random.randint(-1, self.num_robots)

        # 保证任务分配ID合法
        new_sol = np.clip(new_sol, -1, self.num_robots - 1)
        return new_sol

    def optimize(self):
        self.current_iteration = 0
        for _ in range(self.max_iterations):
            best_sol, best_fit = self._run_iteration()
            self.history.append(best_fit)

            if (_ + 1) % 10 == 0 or _ == self.max_iterations - 1:
                print(f"Iteration {_ + 1}/{self.max_iterations}, Best Fitness: {best_fit:.4f}")

        return self.best_solution, self.best_fitness

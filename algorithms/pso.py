import numpy as np

class PSO:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.population_size = config['population_size']
        self.max_iterations = config['max_iterations']
        self.num_tasks = env.num_tasks
        self.num_robots = env.num_robots
        
        # 初始化种群
        self.population = np.random.randint(
            -1, self.num_robots,
            size=(self.population_size, self.num_tasks)
        )
        
        # 初始化速度
        self.velocities = np.zeros_like(self.population, dtype=float)
        
        # 初始化个体最优和全局最优
        self.pbest = self.population.copy()
        self.pbest_fitness = np.array([float('inf')] * self.population_size)
        self.gbest = None
        self.gbest_fitness = float('inf')
        
        # 当前迭代次数
        self.current_iteration = 0
        
        # 收敛历史
        self.history = []
    
    def _run_iteration(self):
        """运行单次迭代"""
        # 评估当前种群
        fitness_values = np.array([
            self.env.calculate_fitness(solution)
            for solution in self.population
        ])
        
        # 更新个体最优
        improved = fitness_values < self.pbest_fitness
        self.pbest[improved] = self.population[improved]
        self.pbest_fitness[improved] = fitness_values[improved]
        
        # 更新全局最优
        best_idx = np.argmin(fitness_values)
        if fitness_values[best_idx] < self.gbest_fitness:
            self.gbest = self.population[best_idx].copy()
            self.gbest_fitness = fitness_values[best_idx]
        
        # 更新速度和位置
        w = self.config['w']  # 惯性权重
        c1 = self.config['c1']  # 个体学习因子
        c2 = self.config['c2']  # 社会学习因子
        
        r1 = np.random.random()
        r2 = np.random.random()
        
        # 更新速度
        self.velocities = (w * self.velocities +
                          c1 * r1 * (self.pbest - self.population) +
                          c2 * r2 * (self.gbest - self.population))
        
        # 更新位置
        self.population = self.population + self.velocities.astype(int)
        
        # 保持解在有效范围内
        self.population = np.clip(
            self.population, -1, self.num_robots - 1
        ).astype(int)
        
        # 更新迭代计数
        self.current_iteration += 1
        
        return self.gbest, self.gbest_fitness
    
    def optimize(self):
        """优化主循环"""
        self.current_iteration = 0
        
        for _ in range(self.max_iterations):
            best_solution, best_fitness = self._run_iteration()
            self.history.append(best_fitness)
            
            if (_ + 1) % 10 == 0:
                print(f"Iteration {_ + 1}/{self.max_iterations}, "
                      f"Best Fitness: {best_fitness:.4f}")
        
        return self.gbest, self.gbest_fitness

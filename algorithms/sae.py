import numpy as np

class SAE:
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
        
        # 初始化最优解
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # 模拟退火参数
        self.initial_temp = 100.0
        self.final_temp = 1.0
        self.cooling_rate = (self.final_temp / self.initial_temp) ** (1.0 / self.max_iterations)
        self.current_temp = self.initial_temp
        
        # 当前迭代次数
        self.current_iteration = 0
        
        # 收敛历史
        self.history = []
    
    def _run_iteration(self):
        """运行单次迭代"""
        # 对每个个体进行模拟退火进化
        for i in range(self.population_size):
            # 生成新解
            new_solution = self.population[i].copy()
            
            # 随机选择一个任务进行变异
            task_idx = np.random.randint(0, self.num_tasks)
            new_solution[task_idx] = np.random.randint(-1, self.num_robots)
            
            # 评估新解
            new_fitness = self.env.calculate_fitness(new_solution)
            current_fitness = self.env.calculate_fitness(self.population[i])
            
            # 计算能量差
            delta_e = new_fitness - current_fitness
            
            # Metropolis准则
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / self.current_temp):
                self.population[i] = new_solution
                
                # 更新全局最优
                if new_fitness < self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness
        
        # 进行交叉操作
        for i in range(0, self.population_size - 1, 2):
            if np.random.random() < 0.8:  # 交叉概率
                # 单点交叉
                crossover_point = np.random.randint(1, self.num_tasks)
                temp = self.population[i, crossover_point:].copy()
                self.population[i, crossover_point:] = self.population[i+1, crossover_point:]
                self.population[i+1, crossover_point:] = temp
        
        # 更新温度
        self.current_temp *= self.cooling_rate
        
        # 更新迭代计数
        self.current_iteration += 1
        
        return self.best_solution, self.best_fitness
    
    def optimize(self):
        """优化主循环"""
        self.current_iteration = 0
        self.current_temp = self.initial_temp
        
        for _ in range(self.max_iterations):
            best_solution, best_fitness = self._run_iteration()
            self.history.append(best_fitness)
            
            if (_ + 1) % 10 == 0:
                print(f"Iteration {_ + 1}/{self.max_iterations}, "
                      f"Temperature: {self.current_temp:.4f}, "
                      f"Best Fitness: {best_fitness:.4f}")
        
        return self.best_solution, self.best_fitness

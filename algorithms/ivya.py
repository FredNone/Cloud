import numpy as np

class IVYA:
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
        
        # 当前迭代次数
        self.current_iteration = 0
        
        # 收敛历史
        self.history = []
        
        # 从配置中获取IVYA特定参数
        ivya_config = config.get('ivya', {})
        self.vortex_radius = ivya_config.get('vortex_radius', 0.5)
        self.vortex_strength = ivya_config.get('vortex_strength', 0.8)
        self.attraction_factor = ivya_config.get('attraction_factor', 0.1)
        self.mutation_rate = ivya_config.get('mutation_rate', 0.1)
    
    def _calculate_vortex_force(self, particle, center):
        """计算涡流力"""
        distance = np.linalg.norm(particle - center)
        if distance < self.vortex_radius:
            # 计算切向力
            tangent = np.array([-particle[1] + center[1], particle[0] - center[0]])
            tangent = tangent / np.linalg.norm(tangent)
            force = self.vortex_strength * tangent
            return force
        return np.zeros_like(particle)
    
    def _run_iteration(self):
        """运行单次迭代"""
        # 评估当前种群
        fitness_values = np.array([
            self.env.calculate_fitness(solution)
            for solution in self.population
        ])
        
        # 更新最优解
        best_idx = np.argmin(fitness_values)
        if fitness_values[best_idx] < self.best_fitness:
            self.best_solution = self.population[best_idx].copy()
            self.best_fitness = fitness_values[best_idx]
        
        # 计算种群中心
        center = np.mean(self.population, axis=0)
        
        # 更新每个粒子
        new_population = []
        for i in range(self.population_size):
            particle = self.population[i].copy()
            
            # 计算涡流力
            vortex_force = self._calculate_vortex_force(particle, center)
            
            # 计算向最优解的吸引力
            attraction = self.best_solution - particle
            
            # 更新位置
            particle = particle + vortex_force + self.attraction_factor * attraction
            
            # 变异
            if np.random.random() < self.mutation_rate:
                mutation_point = np.random.randint(0, self.num_tasks)
                particle[mutation_point] = np.random.randint(-1, self.num_robots)
            
            # 保持解在有效范围内
            particle = np.clip(particle, -1, self.num_robots - 1)
            particle = particle.astype(int)
            
            new_population.append(particle)
        
        self.population = np.array(new_population)
        
        # 更新迭代计数
        self.current_iteration += 1
        
        return self.best_solution, self.best_fitness
    
    def optimize(self):
        """优化主循环"""
        self.current_iteration = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        
        for _ in range(self.max_iterations):
            best_solution, best_fitness = self._run_iteration()
            self.history.append(best_fitness)
            
            if (_ + 1) % 10 == 0:
                print(f"Iteration {_ + 1}/{self.max_iterations}, "
                      f"Best Fitness: {best_fitness:.4f}")
        
        return self.best_solution, self.best_fitness 
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular

class SAEO:
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
        
        # 初始化代理模型参数
        self.surrogate_model = None
        self.surrogate_history = []
        self.true_evaluations = 0
        
        # 从配置中获取SAEO特定参数
        saeo_config = config.get('saeo', {})
        self.max_true_evaluations = saeo_config.get('max_true_evaluations', 1000)
        self.regularization = saeo_config.get('regularization', 1e-6)
        self.min_distance = saeo_config.get('min_distance', 1e-6)
        self.mutation_rate = saeo_config.get('mutation_rate', 0.1)
        
        # 初始化最优解
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # 当前迭代次数
        self.current_iteration = 0
        
        # 收敛历史
        self.history = []
    
    def _build_surrogate_model(self, X, y):
        """构建RBF代理模型"""
        # 计算RBF核矩阵
        distances = cdist(X, X)
        
        # 处理过小的距离
        distances = np.maximum(distances, self.min_distance)
        
        # 使用平均距离作为带宽
        sigma = np.mean(distances)
        K = np.exp(-distances**2 / (2 * sigma**2))
        
        # 添加正则化项
        K = K + self.regularization * np.eye(len(X))
        
        try:
            # 使用Cholesky分解求解
            L = np.linalg.cholesky(K)
            alpha = solve_triangular(L, y, lower=True)
            alpha = solve_triangular(L.T, alpha, lower=False)
        except np.linalg.LinAlgError:
            # 如果Cholesky分解失败，使用伪逆
            alpha = np.linalg.pinv(K) @ y
        
        return {'alpha': alpha, 'X': X, 'sigma': sigma}
    
    def _predict_surrogate(self, model, x):
        """使用代理模型进行预测"""
        distances = cdist([x], model['X'])[0]
        distances = np.maximum(distances, self.min_distance)
        K = np.exp(-distances**2 / (2 * model['sigma']**2))
        return np.dot(K, model['alpha'])
    
    def _run_iteration(self):
        """运行单次迭代"""
        # 评估当前种群
        if self.true_evaluations < self.max_true_evaluations:
            fitness_values = np.array([
                self.env.calculate_fitness(solution)
                for solution in self.population
            ])
            self.true_evaluations += len(self.population)
            
            # 更新代理模型
            if len(self.surrogate_history) > 0:
                X = np.vstack([h['solution'] for h in self.surrogate_history])
                y = np.array([h['fitness'] for h in self.surrogate_history])
                
                # 移除重复样本
                unique_indices = np.unique(X, axis=0, return_index=True)[1]
                X = X[unique_indices]
                y = y[unique_indices]
                
                if len(X) > 1:  # 确保有足够的样本点
                    self.surrogate_model = self._build_surrogate_model(X, y)
            
            # 记录历史
            for solution, fitness in zip(self.population, fitness_values):
                self.surrogate_history.append({
                    'solution': solution,
                    'fitness': fitness
                })
        else:
            # 使用代理模型预测适应度
            if self.surrogate_model is not None:
                fitness_values = np.array([
                    self._predict_surrogate(self.surrogate_model, solution)
                    for solution in self.population
                ])
            else:
                # 如果代理模型不可用，使用最后一次真实评估的结果
                fitness_values = np.array([self.best_fitness] * self.population_size)
        
        # 更新最优解
        best_idx = np.argmin(fitness_values)
        if fitness_values[best_idx] < self.best_fitness:
            self.best_solution = self.population[best_idx].copy()
            self.best_fitness = fitness_values[best_idx]
        
        # 生成新解
        new_population = []
        for i in range(self.population_size):
            # 选择父代
            parent1_idx = np.random.randint(0, self.population_size)
            parent2_idx = np.random.randint(0, self.population_size)
            
            # 交叉
            crossover_point = np.random.randint(0, self.num_tasks)
            child = np.concatenate([
                self.population[parent1_idx][:crossover_point],
                self.population[parent2_idx][crossover_point:]
            ])
            
            # 变异
            if np.random.random() < self.mutation_rate:
                mutation_point = np.random.randint(0, self.num_tasks)
                child[mutation_point] = np.random.randint(-1, self.num_robots)
            
            new_population.append(child)
        
        self.population = np.array(new_population)
        
        # 更新迭代计数
        self.current_iteration += 1
        
        return self.best_solution, self.best_fitness
    
    def optimize(self):
        """优化主循环"""
        self.current_iteration = 0
        self.true_evaluations = 0
        self.surrogate_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        
        for _ in range(self.max_iterations):
            best_solution, best_fitness = self._run_iteration()
            self.history.append(best_fitness)
            
            if (_ + 1) % 10 == 0:
                print(f"Iteration {_ + 1}/{self.max_iterations}, "
                      f"Best Fitness: {best_fitness:.4f}, "
                      f"True Evaluations: {self.true_evaluations}")
        
        return self.best_solution, self.best_fitness 
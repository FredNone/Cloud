import numpy as np
from utils.metrics import levy_flight

class HybridPSOGWO:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.population_size = config['population_size']
        self.max_iterations = config['max_iterations']
        self.num_tasks = env.num_tasks
        self.num_robots = env.num_robots
        
        # 初始化种群（确保初始解不会有太多未分配任务）
        self.population = np.random.randint(
            0, self.num_robots,  # 初始时全部分配
            size=(self.population_size, self.num_tasks)
        )
        # 随机将一小部分任务设为未分配（-1）
        mask = np.random.random((self.population_size, self.num_tasks)) < 0.1
        self.population[mask] = -1
        
        # PSO参数
        self.velocities = np.zeros_like(self.population, dtype=float)
        self.pbest = self.population.copy()
        self.pbest_fitness = np.array([float('inf')] * self.population_size)
        self.gbest = None
        self.gbest_fitness = float('inf')
        
        # GWO参数
        self.alpha = None
        self.beta = None
        self.delta = None
        self.alpha_fitness = float('inf')
        self.beta_fitness = float('inf')
        self.delta_fitness = float('inf')
        
        # 当前迭代次数
        self.current_iteration = 0
        
        # 收敛历史
        self.history = []
        
        # 自适应参数
        self.w_max = 0.9  # 最大惯性权重
        self.w_min = 0.4  # 最小惯性权重
        self.c1_init = 2.5  # 初始个体学习因子
        self.c2_init = 2.5  # 初始社会学习因子
    
    def _run_iteration(self):
        """运行单次迭代"""
        # 评估当前种群
        fitness_values = np.array([
            self.env.calculate_fitness(solution)
            for solution in self.population
        ])
        
        # 更新PSO的个体最优和全局最优
        self._update_pso_best(fitness_values)
        
        # 更新GWO的alpha、beta和delta狼
        self._update_gwo_wolves(fitness_values)
        
        # 自适应参数调整
        progress = self.current_iteration / self.max_iterations
        w = self.w_max - (self.w_max - self.w_min) * progress
        c1 = self.c1_init - self.c1_init * progress
        c2 = self.c2_init + (2.5 - self.c2_init) * progress
        
        # 根据迭代次数决定使用PSO还是GWO更新
        if self.current_iteration < self.max_iterations // 5:  # 前20%使用PSO
            self._pso_update(w, c1, c2)
        else:  # 后80%使用混合更新
            self._hybrid_update(w, c1, c2, progress)
        
        # 应用Lévy飞行进行局部搜索
        if self.current_iteration > self.max_iterations // 2:  # 在后半段使用
            best_idx = np.argmin(fitness_values)
            levy_step = levy_flight(self.num_tasks, scale=0.1)
            self.population[best_idx] = np.clip(
                self.population[best_idx] + levy_step.astype(int),
                -1, self.num_robots - 1
            )
        
        # 保持解在有效范围内
        self.population = np.clip(
            self.population, -1, self.num_robots - 1
        ).astype(int)
        
        # 更新迭代计数
        self.current_iteration += 1
        
        return self.gbest, self.gbest_fitness
    
    def _repair_solution(self, solution):
        """修复解决方案中的连续未分配问题"""
        # 获取每个机器人当前的负载
        robot_loads = np.bincount(solution[solution >= 0], minlength=self.num_robots)
        
        # 计算理想的平均负载
        total_tasks = np.sum(solution >= 0)
        avg_load = total_tasks / self.num_robots if total_tasks > 0 else 0
        
        # 找出连续的未分配任务
        unassigned_mask = solution == -1
        consecutive_unassigned = np.zeros_like(solution, dtype=bool)
        
        # 标记连续的未分配任务（当有3个或更多连续未分配时）
        for i in range(len(solution)):
            if i >= 2 and unassigned_mask[i-2:i+1].all():
                consecutive_unassigned[i-2:i+1] = True
        
        # 对连续未分配的任务进行重新分配
        if np.any(consecutive_unassigned):
            # 找出负载较轻的机器人
            underloaded_robots = np.where(robot_loads < avg_load)[0]
            if len(underloaded_robots) == 0:
                underloaded_robots = np.arange(self.num_robots)
            
            # 为连续未分配的任务分配机器人
            consecutive_indices = np.where(consecutive_unassigned)[0]
            for idx in consecutive_indices:
                # 选择负载最小的机器人
                robot_id = underloaded_robots[np.argmin(robot_loads[underloaded_robots])]
                solution[idx] = robot_id
                robot_loads[robot_id] += 1
        
        return solution

    def _hybrid_update(self, w, c1, c2, progress):
        """混合更新策略"""
        # GWO部分
        a = 2 * (1 - progress)  # 线性递减
        A1 = 2 * a * np.random.random((self.population_size, self.num_tasks)) - a
        A2 = 2 * a * np.random.random((self.population_size, self.num_tasks)) - a
        A3 = 2 * a * np.random.random((self.population_size, self.num_tasks)) - a
        
        C1 = 2 * np.random.random((self.population_size, self.num_tasks))
        C2 = 2 * np.random.random((self.population_size, self.num_tasks))
        C3 = 2 * np.random.random((self.population_size, self.num_tasks))
        
        # 计算GWO的位置更新
        D_alpha = np.abs(C1 * self.alpha - self.population)
        D_beta = np.abs(C2 * self.beta - self.population)
        D_delta = np.abs(C3 * self.delta - self.population)
        
        X1 = self.alpha - A1 * D_alpha
        X2 = self.beta - A2 * D_beta
        X3 = self.delta - A3 * D_delta
        
        # PSO部分
        r1 = np.random.random()
        r2 = np.random.random()
        
        # 分别计算PSO和GWO的整数位置
        pso_velocity = (w * self.velocities +
                       c1 * r1 * (self.pbest - self.population) +
                       c2 * r2 * (self.gbest - self.population))
        
        # 先分别转换为整数
        gwo_position = np.round((X1 + X2 + X3) / 3).astype(int)
        pso_position = np.round(self.population + pso_velocity).astype(int)
        
        # 自适应权重
        gwo_weight = progress  # GWO的权重随迭代增加
        pso_weight = 1 - progress  # PSO的权重随迭代减少
        
        # 概率选择使用GWO或PSO的结果
        random_choice = np.random.random((self.population_size, self.num_tasks))
        self.population = np.where(random_choice < gwo_weight, 
                                 gwo_position, 
                                 pso_position)
        
        # 更新速度（保持浮点数以维持PSO的搜索能力）
        self.velocities = pso_velocity
        
        # 确保解在有效范围内
        self.population = np.clip(self.population, -1, self.num_robots - 1)
        
        # 对每个解进行修复
        for i in range(self.population_size):
            self.population[i] = self._repair_solution(self.population[i])
    
    def _pso_update(self, w, c1, c2):
        """PSO更新"""
        r1 = np.random.random()
        r2 = np.random.random()
        
        # 更新速度
        self.velocities = (w * self.velocities + 
                          c1 * r1 * (self.pbest - self.population) +
                          c2 * r2 * (self.gbest - self.population))
        
        # 更新位置
        self.population = self.population + self.velocities.astype(int)
    
    def _update_pso_best(self, fitness_values):
        """更新PSO的最优解"""
        # 更新个体最优
        improved = fitness_values < self.pbest_fitness
        self.pbest[improved] = self.population[improved]
        self.pbest_fitness[improved] = fitness_values[improved]
        
        # 更新全局最优
        best_idx = np.argmin(fitness_values)
        if fitness_values[best_idx] < self.gbest_fitness:
            self.gbest = self.population[best_idx].copy()
            self.gbest_fitness = fitness_values[best_idx]
    
    def _update_gwo_wolves(self, fitness_values):
        """更新GWO的狼群等级"""
        sorted_idx = np.argsort(fitness_values)
        
        # 更新alpha狼（最优解）
        if fitness_values[sorted_idx[0]] < self.alpha_fitness:
            self.alpha = self.population[sorted_idx[0]].copy()
            self.alpha_fitness = fitness_values[sorted_idx[0]]
            
            # 同步更新全局最优（用于PSO部分）
            if fitness_values[sorted_idx[0]] < self.gbest_fitness:
                self.gbest = self.alpha.copy()
                self.gbest_fitness = self.alpha_fitness
        
        # 更新beta狼（次优解）
        if fitness_values[sorted_idx[1]] < self.beta_fitness:
            self.beta = self.population[sorted_idx[1]].copy()
            self.beta_fitness = fitness_values[sorted_idx[1]]
        
        # 更新delta狼（第三优解）
        if fitness_values[sorted_idx[2]] < self.delta_fitness:
            self.delta = self.population[sorted_idx[2]].copy()
            self.delta_fitness = fitness_values[sorted_idx[2]]
    
    def optimize(self):
        """优化主循环"""
        self.current_iteration = 0
        
        for _ in range(self.max_iterations):
            best_solution, best_fitness = self._run_iteration()
            self.history.append(best_fitness)
            
            if (_ + 1) % 100 == 0:  # 减少打印频率
                print(f"Iteration {_ + 1}/{self.max_iterations}, "
                      f"Best Fitness: {best_fitness:.4f}")
        
        return self.gbest, self.gbest_fitness
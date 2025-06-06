import numpy as np

class GWO:
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
        
        # 初始化alpha、beta和delta狼
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
    
    def _run_iteration(self):
        """运行单次迭代"""
        # 评估当前种群
        fitness_values = np.array([
            self.env.calculate_fitness(solution)
            for solution in self.population
        ])
        
        # 更新alpha、beta和delta狼
        sorted_idx = np.argsort(fitness_values)
        
        # 更新alpha（最优解）
        if fitness_values[sorted_idx[0]] < self.alpha_fitness:
            self.alpha = self.population[sorted_idx[0]].copy()
            self.alpha_fitness = fitness_values[sorted_idx[0]]
        
        # 更新beta（次优解）
        if fitness_values[sorted_idx[1]] < self.beta_fitness:
            self.beta = self.population[sorted_idx[1]].copy()
            self.beta_fitness = fitness_values[sorted_idx[1]]
        
        # 更新delta（第三优解）
        if fitness_values[sorted_idx[2]] < self.delta_fitness:
            self.delta = self.population[sorted_idx[2]].copy()
            self.delta_fitness = fitness_values[sorted_idx[2]]
        
        # 更新a参数（线性递减）
        a = self.config['a']
        a_linear = a - self.current_iteration * (a / self.max_iterations)
        
        # 更新A和C参数
        A1 = 2 * a_linear * np.random.random((self.population_size, self.num_tasks)) - a_linear
        A2 = 2 * a_linear * np.random.random((self.population_size, self.num_tasks)) - a_linear
        A3 = 2 * a_linear * np.random.random((self.population_size, self.num_tasks)) - a_linear
        
        C1 = 2 * np.random.random((self.population_size, self.num_tasks))
        C2 = 2 * np.random.random((self.population_size, self.num_tasks))
        C3 = 2 * np.random.random((self.population_size, self.num_tasks))
        
        # 更新位置
        D_alpha = np.abs(C1 * self.alpha - self.population)
        D_beta = np.abs(C2 * self.beta - self.population)
        D_delta = np.abs(C3 * self.delta - self.population)
        
        X1 = self.alpha - A1 * D_alpha
        X2 = self.beta - A2 * D_beta
        X3 = self.delta - A3 * D_delta
        
        # 计算新位置并确保在有效范围内
        new_positions = ((X1 + X2 + X3) / 3).astype(int)
        new_positions = np.clip(new_positions, -1, self.num_robots - 1)
        
        # 应用负载均衡
        for i in range(self.population_size):
            new_positions[i] = self._balance_load(new_positions[i])
        
        self.population = new_positions
        
        # 更新迭代计数
        self.current_iteration += 1
        
        return self.alpha, self.alpha_fitness
    
    def _balance_load(self, solution):
        """平衡任务负载"""
        # 确保所有机器人ID在有效范围内
        solution = np.clip(solution, -1, self.num_robots - 1)
        
        # 计算每个机器的当前负载
        robot_loads = np.bincount(solution[solution >= 0], minlength=self.num_robots)
        
        # 计算每个机器的带宽使用情况
        bandwidth_usage = np.zeros(self.num_robots)
        connection_counts = np.zeros(self.num_robots, dtype=int)
        
        for task_id, robot_id in enumerate(solution):
            if robot_id >= 0:
                bandwidth_usage[robot_id] += (self.env.packet_size * 
                                           self.env.comm_overhead * 
                                           self.env.task_complexity[task_id])
                connection_counts[robot_id] += 1
        
        # 找出过载的机器
        overloaded_mask = ((robot_loads > np.mean(robot_loads) * 1.5) | 
                          (bandwidth_usage > self.env.bandwidth * 0.8) |
                          (connection_counts > self.env.max_connections))
        
        if np.any(overloaded_mask):
            # 找出负载较轻的机器
            underloaded_mask = ~overloaded_mask
            if np.any(underloaded_mask):
                underloaded_robots = np.where(underloaded_mask)[0]
                
                # 重新分配过载机器的部分任务
                for robot_id in np.where(overloaded_mask)[0]:
                    # 找出分配给该机器的任务
                    task_indices = np.where(solution == robot_id)[0]
                    
                    # 随机选择一些任务重新分配
                    num_tasks_to_move = len(task_indices) // 3  # 移动1/3的任务
                    if num_tasks_to_move > 0:
                        tasks_to_move = np.random.choice(
                            task_indices, 
                            size=num_tasks_to_move, 
                            replace=False
                        )
                        
                        # 将任务分配给负载较轻的机器
                        for task_id in tasks_to_move:
                            # 选择当前负载最小的机器
                            new_robot = underloaded_robots[
                                np.argmin(robot_loads[underloaded_robots])
                            ]
                            solution[task_id] = new_robot
                            robot_loads[new_robot] += 1
                            robot_loads[robot_id] -= 1
        
        return solution
    
    def optimize(self):
        """优化主循环"""
        self.current_iteration = 0
        
        for _ in range(self.max_iterations):
            best_solution, best_fitness = self._run_iteration()
            self.history.append(best_fitness)
            
            if (_ + 1) % 10 == 0:
                print(f"Iteration {_ + 1}/{self.max_iterations}, "
                      f"Best Fitness: {best_fitness:.4f}")
        
        return self.alpha, self.alpha_fitness

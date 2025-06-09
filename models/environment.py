import numpy as np
from scipy.spatial.distance import cdist
import time

class Environment:
    def __init__(self, config):
        self.config = config
        self.num_tasks = config['num_tasks']
        self.num_robots = config['num_robots']
        self.workspace_size = config['workspace_size']
        
        # 初始化任务和机器人的位置（随机分布在工作空间中）
        self.task_positions = np.random.rand(self.num_tasks, 2) * self.workspace_size
        self.robot_positions = np.random.rand(self.num_robots, 2) * self.workspace_size
        
        # 初始化任务复杂度和机器人能力
        self.task_complexity = np.random.uniform(
            config['task_complexity']['min'],
            config['task_complexity']['max'],
            self.num_tasks
        )
        self.robot_capabilities = np.random.uniform(
            config['robot_capabilities']['min'],
            config['robot_capabilities']['max'],
            self.num_robots
        )
        
        # 预计算所有任务到所有机器人的距离矩阵
        self.distance_matrix = cdist(self.task_positions, self.robot_positions)
        
        # 预计算所有任务和机器人的能力差异矩阵
        self.capability_matrix = np.abs(
            self.task_complexity[:, np.newaxis] - self.robot_capabilities
        )
        
        # 初始化网络相关参数
        self.bandwidth = config['network']['bandwidth']
        self.latency = config['network']['latency']
        self.packet_size = config['network']['packet_size']
        self.max_connections = config['communication']['max_connections']
        self.comm_overhead = config['communication']['overhead']
        
        # 计算机器间的网络延迟矩阵（基于距离）
        self.network_latency_matrix = self._calculate_network_latency()
        
        # 初始化每个机器的当前带宽使用情况
        self.bandwidth_usage = np.zeros(self.num_robots)
        
        # 初始化每个机器的连接数
        self.connection_counts = np.zeros(self.num_robots, dtype=int)
        
        # 预计算常量
        self.max_capability_diff = (config['robot_capabilities']['max'] - 
                                  config['robot_capabilities']['min'])
        self.diagonal_size = np.sqrt(2) * self.workspace_size
        
        # 权重配置
        self.weights = {
            'distance': 0.3,
            'capability': 0.2,
            'assignment': 0.3,
            'balance': 0.2
        }
        # 初始化任务数量和机器人数量
        self.num_tasks = config['num_tasks']
        self.num_robots = config['num_robots']

        # 初始化网络延迟矩阵，单位：秒（模拟传输时延）
        self.network_delay = np.random.uniform(low=0.1, high=1.0, size=(self.num_tasks, self.num_robots))

        # 初始化执行时间矩阵，单位：秒（模拟机器人计算任务时延）
        self.execution_time = np.random.uniform(low=0.5, high=2.0, size=(self.num_tasks, self.num_robots))

    def _calculate_network_latency(self):
        """计算机器间的网络延迟矩阵"""
        # 基础延迟
        base_latency = self.latency
        
        # 计算机器间的距离
        distances = cdist(self.robot_positions, self.robot_positions)
        
        # 将距离转换为延迟（假设延迟与距离成正比）
        latency_matrix = base_latency + (distances / self.workspace_size) * base_latency
        
        return latency_matrix
    
    def _calculate_communication_cost(self, solution):
        """计算通信成本"""
        # 重置带宽使用和连接计数
        self.bandwidth_usage.fill(0)
        self.connection_counts.fill(0)
        
        # 计算每个机器的任务数和连接数
        for task_id, robot_id in enumerate(solution):
            if robot_id >= 0:
                self.connection_counts[robot_id] += 1
                
                # 估算带宽使用（基于任务复杂度）
                self.bandwidth_usage[robot_id] += (self.packet_size * 
                                                 self.comm_overhead * 
                                                 self.task_complexity[task_id])
        
        # 计算带宽超限惩罚
        bandwidth_penalty = np.sum(
            np.maximum(0, self.bandwidth_usage - self.bandwidth)
        ) / self.bandwidth
        
        # 计算连接数超限惩罚
        connection_penalty = np.sum(
            np.maximum(0, self.connection_counts - self.max_connections)
        ) / self.max_connections
        
        return bandwidth_penalty, connection_penalty
    
    def calculate_fitness(self, solution):
        """计算解决方案的适应度值（考虑网络约束）"""
        # 创建分配掩码
        assignment_mask = solution >= 0
        assigned_tasks = np.sum(assignment_mask)
        
        # 如果没有分配任何任务，返回惩罚值
        if assigned_tasks == 0:
            return float('inf')
        
        # 1. 计算总距离（使用预计算的距离矩阵）
        valid_distances = self.distance_matrix[assignment_mask, solution[assignment_mask]]
        normalized_distance = np.mean(valid_distances) / self.diagonal_size
        
        # 2. 计算能力匹配度（使用预计算的能力矩阵）
        valid_capabilities = self.capability_matrix[assignment_mask, solution[assignment_mask]]
        normalized_capability = np.mean(valid_capabilities) / self.max_capability_diff
        
        # 3. 计算任务分配率
        assignment_rate = assigned_tasks / self.num_tasks
        assignment_penalty = 1.0 - assignment_rate
        
        # 4. 计算负载均衡因子
        robot_loads = np.bincount(solution[assignment_mask], minlength=self.num_robots)
        avg_load = assigned_tasks / self.num_robots
        max_load = np.max(robot_loads)
        load_imbalance = (max_load - avg_load) / (avg_load + 1e-10) if avg_load > 0 else 0
        
        # 5. 计算网络相关成本
        bandwidth_penalty, connection_penalty = self._calculate_communication_cost(solution)
        
        # 组合所有指标
        fitness = (self.weights['distance'] * normalized_distance +
                  self.weights['capability'] * normalized_capability +
                  self.weights['assignment'] * assignment_penalty +
                  self.weights['balance'] * load_imbalance +
                  0.2 * (bandwidth_penalty + connection_penalty))  # 网络惩罚权重
        
        return fitness
    
    def performance_test(self, num_tests=1000):
        """性能测试方法"""
        # 生成随机解决方案
        solutions = np.random.randint(-1, self.num_robots, 
                                    size=(num_tests, self.num_tasks))
        
        # 测试适应度计算性能
        start_time = time.time()
        for solution in solutions:
            self.calculate_fitness(solution)
        end_time = time.time()
        
        # 计算平均时间（毫秒）
        avg_time = (end_time - start_time) * 1000 / num_tests
        print(f"Average fitness calculation time: {avg_time:.2f} ms")

    def calculate_total_delay(self, solution):
            """
            计算某一任务分配方案的总传输+执行时延
            solution: 长度为 num_tasks 的数组，表示每个任务分配到哪个机器人
            """
            total_delay = 0.0
            for task_id, robot_id in enumerate(solution):
                transmit_delay = self.network_delay[task_id][robot_id]
                execution_delay = self.execution_time[task_id][robot_id]
                total_delay += transmit_delay + execution_delay
            return total_delay

    #
    # def calculate_total_delay(self, solution):
    #     """
    #     根据任务分配方案，计算任务的总时延。
    #     :param solution: 任务分配向量，例如 [0, 1, 0, 2, ...]
    #     :return: 总时延
    #     """
    #     total_delay = 0.0
    #     for task_id, robot_id in enumerate(solution):
    #         # 你需要根据模型定义通信延迟 + 执行时间
    #         transmit_delay = self.network_delay[task_id][robot_id]
    #         execute_time = self.execution_time[task_id][robot_id]
    #         total_delay += transmit_delay + execute_time
    #     return total_delay
    #
    # def calculate_mismatch_cost(self, solution):
    #     """
    #     计算任务分配中的资源不匹配成本。
    #     可按任务负载与机器人能力差值建模。
    #     """
    #     mismatch = 0.0
    #     for task_id, robot_id in enumerate(solution):
    #         task_load = self.task_loads[task_id]
    #         robot_capacity = self.robot_capacities[robot_id]
    #         mismatch += abs(task_load - robot_capacity)
    #     return mismatch
    #
    # def calculate_priority_penalty(self, solution):
    #     """
    #     根据任务的优先级或紧迫度，对不合理分配加罚。
    #     """
    #     penalty = 0.0
    #     for task_id, robot_id in enumerate(solution):
    #         priority = self.task_priorities[task_id]
    #         delay = self.network_delay[task_id][robot_id] + self.execution_time[task_id][robot_id]
    #         penalty += priority * delay  # 紧急任务晚完成惩罚大
    #     return penalty
    #
    # def calculate_total_delay(self, solution):
    #     """
    #     根据任务分配方案，计算所有任务的总延迟（network delay + execution time）。
    #     :param solution: 一个列表，表示每个任务分配到的机器人索引，例如 [0, 2, 1, 0, ...]
    #     :return: 总时延（浮点数）
    #     """
    #     total_delay = 0.0
    #     for task_id, robot_id in enumerate(solution):
    #         # 假设这两个矩阵在 Environment 初始化时就已经构建：
    #         # self.network_delay[task_id][robot_id]: 任务与机器人之间的传输时延
    #         # self.execution_time[task_id][robot_id]: 机器人处理该任务的时间
    #
    #         transmit_delay = self.network_delay[task_id][robot_id]
    #         execute_time = self.execution_time[task_id][robot_id]
    #
    #         # 总时延 = 传输延迟 + 执行时间
    #         total_delay += transmit_delay + execute_time
    #
    #     return total_delay
    #
    # def calculate_total_delay(self, solution):
    #     """根据任务分配方案计算总时延（传输+执行）"""
    #     total_delay = 0.0
    #     for task_id, robot_id in enumerate(solution):
    #         transmit_delay = self.network_delay[task_id][robot_id]
    #         execute_delay = self.execution_time[task_id][robot_id]
    #         total_delay += transmit_delay + execute_delay
    #     return total_delay
    #
    # def calculate_mismatch_cost(self, solution):
    #     """
    #     计算任务与机器人之间的能力不匹配成本（欧几里得距离）
    #     距离越大表示能力越不匹配
    #     """
    #     total_mismatch = 0.0
    #     for task_id, robot_id in enumerate(solution):
    #         task_req = self.task_requirements[task_id]
    #         robot_cap = self.robot_capabilities[robot_id]
    #         mismatch = np.linalg.norm(task_req - robot_cap)
    #         total_mismatch += mismatch
    #     return total_mismatch
    #
    # def calculate_priority_penalty(self, solution):
    #     """
    #     计算任务优先级惩罚：根据任务重要性和时延，对高优先任务延迟惩罚更多
    #     任务优先级越高（数值越大），则对延迟越敏感
    #     """
    #     if not hasattr(self, 'task_priority'):
    #         # 如果未设置任务优先级，初始化为随机[1~5]
    #         self.task_priority = np.random.randint(1, 6, size=self.num_tasks)
    #
    #     total_penalty = 0.0
    #     for task_id, robot_id in enumerate(solution):
    #         delay = self.network_delay[task_id][robot_id] + self.execution_time[task_id][robot_id]
    #         priority = self.task_priority[task_id]
    #         penalty = priority * delay
    #         total_penalty += penalty
    #     return total_penalty

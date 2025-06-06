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
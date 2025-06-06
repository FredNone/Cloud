import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time
from models.environment import Environment

def plot_convergence_delay(results: Dict, sample_step: int = 10, save_path: str = 'convergence_delay.png'):
    """绘制不同算法的时延收敛曲线"""
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        if 'iteration_times' in result:
            # 计算每次迭代的时延（毫秒）
            delays = result['iteration_times']
            # 生成对应的迭代次数（考虑采样步长）
            iterations = np.arange(len(delays)) * sample_step
            plt.plot(iterations, delays, 
                    label=name,
                    marker='o', markersize=4, alpha=0.7)
    
    plt.xlabel('Iteration Count')
    plt.ylabel('System Delay (ms)')
    plt.title('Algorithm Time Delay vs Iterations')
    plt.xlim(0, 1000)
    plt.grid(True)
    plt.legend(loc='upper right')
    
    # 添加y轴数值标签
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_experiment(algorithm_class, env, alg_config, num_repeats=5):
    """运行重复实验并返回平均时间和标准差"""
    times = []
    for _ in range(num_repeats):
        algorithm = algorithm_class(env, alg_config)
        start_time = time.time()
        algorithm.optimize()
        execution_time = time.time() - start_time
        times.append(execution_time * 1000)  # 转换为毫秒
    return np.mean(times), np.std(times)

def evaluate_task_scaling(algorithms: Dict, sim_config: Dict, alg_config: Dict,
                        task_ranges: List[int], save_path: str = 'task_scaling.png'):
    """评估不同任务数量下的算法性能"""
    plt.figure(figsize=(12, 8))
    print(f"\nEvaluating task scaling performance...")
    
    for name, algorithm_class in algorithms.items():
        means = []
        stds = []
        print(f"\nTesting {name}...")
        
        for num_tasks in task_ranges:
            print(f"  Tasks: {num_tasks}")
            current_sim_config = sim_config.copy()
            current_sim_config['num_tasks'] = num_tasks
            
            # 动态调整机器人数量
            robots_needed = max(10, int(num_tasks * 0.2))
            current_sim_config['num_robots'] = min(robots_needed, num_tasks)
            
            # 调整工作空间大小
            current_sim_config['workspace_size'] = int(100 * np.sqrt(num_tasks / 50))
            
            # 创建环境
            env = Environment(current_sim_config)
            
            # 运行实验
            times = []
            for _ in range(3):  # 减少重复次数以加快测试
                algorithm = algorithm_class(env, alg_config)
                start_time = time.time()
                algorithm.optimize()
                execution_time = (time.time() - start_time) * 1000  # 转换为毫秒
                times.append(execution_time)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"    Mean time: {mean_time:.2f}ms")
            
            means.append(mean_time)
            stds.append(std_time)
        
        plt.errorbar(task_ranges, means, yerr=stds, fmt='o-', capsize=5,
                    label=name, markersize=8, alpha=0.7)
    
    plt.xlabel('Number of Tasks')
    plt.ylabel('System Delay (ms)')
    plt.title('Algorithm Performance vs Number of Tasks')
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_robot_scaling(algorithms: Dict, sim_config: Dict, alg_config: Dict,
                         robot_ranges: List[int], save_path: str = 'machine_scaling.png'):
    """评估不同机器数量下的算法性能"""
    plt.figure(figsize=(12, 8))
    print(f"\nEvaluating machine scaling performance...")
    
    for name, algorithm_class in algorithms.items():
        means = []
        stds = []
        print(f"\nTesting {name}...")
        
        for num_machines in robot_ranges:
            print(f"  Machines: {num_machines}")
            current_sim_config = sim_config.copy()
            current_sim_config['num_robots'] = num_machines
            
            # 调整任务数量
            current_sim_config['num_tasks'] = min(50, num_machines)
            
            # 调整工作空间大小
            current_sim_config['workspace_size'] = int(100 * np.sqrt(num_machines / 10))
            
            # 创建环境
            env = Environment(current_sim_config)
            
            # 运行实验
            times = []
            for _ in range(3):  # 减少重复次数以加快测试
                algorithm = algorithm_class(env, alg_config)
                start_time = time.time()
                algorithm.optimize()
                execution_time = (time.time() - start_time) * 1000  # 转换为毫秒
                times.append(execution_time)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"    Mean time: {mean_time:.2f}ms")
            
            means.append(mean_time)
            stds.append(std_time)
        
        plt.errorbar(robot_ranges, means, yerr=stds, fmt='o-', capsize=5,
                    label=name, markersize=8, alpha=0.7)
    
    plt.xlabel('Number of Machines')
    plt.ylabel('System Delay (ms)')
    plt.title('Algorithm Performance vs Number of Machines')
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_operation_scaling(algorithms: Dict, sim_config: Dict, alg_config: Dict,
                             operation_ranges: List[int], save_path: str = 'operation_scaling.png'):
    """评估不同操作数量下的算法性能"""
    plt.figure(figsize=(12, 8))
    
    for name, algorithm_class in algorithms.items():
        means = []
        stds = []
        for num_operations in operation_ranges:
            # 更新配置
            current_alg_config = alg_config.copy()
            current_alg_config['max_iterations'] = num_operations
            
            # 创建环境
            env = Environment(sim_config)
            
            # 运行重复实验
            mean_time, std_time = run_experiment(algorithm_class, env, current_alg_config)
            means.append(mean_time)
            stds.append(std_time)
        
        # 绘制带有误差线的曲线
        plt.errorbar(operation_ranges, means, yerr=stds, fmt='o-', capsize=5,
                    label=name, markersize=8, alpha=0.7)
    
    plt.xlabel('Numbers of Operations')
    plt.ylabel('System Delay (ms)')
    plt.title('Algorithm Performance vs Number of Operations')
    plt.xlim(50, 300)
    plt.grid(True)
    plt.legend()
    
    # 添加y轴数值标签
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 
import time
import numpy as np
# import env as env
import matplotlib.pyplot as plt

from algorithms.Aclho_1 import ACLHO_1
from algorithms.Aclho_base import ACLHO_BASE as ACLHO_Base
from algorithms.Aclho_2 import ACLHO_2
from algorithms.Aclho_3_2 import ACLHO_3_2
from algorithms.Aclho_3_1 import ACLHO_3_1
from sklearn.model_selection import ParameterGrid
from config import SIMULATION_CONFIG, ALGORITHM_CONFIG
from models.environment import Environment
from algorithms.pso import PSO
from algorithms.gwo import GWO
from algorithms.saeo import SAEO
from algorithms.ivya import IVYA
from utils.performance import (plot_convergence_delay,plot_convergence_curves, evaluate_task_scaling,
                               evaluate_robot_scaling, evaluate_operation_scaling)



def run_algorithm_with_timing(algorithm):
    """运行算法并记录每次迭代的时间"""
    iteration_times = []
    start_time = time.time()

    # 保存原始的optimize方法
    original_optimize = algorithm.optimize

    def timed_optimize():

        best_solution = None
        best_fitness = float('inf')
        history = []

        # 设置采样步长
        sample_step = 10  # 每10次迭代记录一次时间
        accumulated_time = 0

        for i in range(algorithm.max_iterations):
            iter_start = time.time()

            # 运行一次迭代
            current_solution, current_fitness = algorithm._run_iteration()

            # 更新最优解
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness

            # 记录历史
            history.append(best_fitness)

            # 累积时间并按步长记录
            iter_time = (time.time() - iter_start) * 1000  # 转换为毫秒
            accumulated_time += iter_time

            if (i + 1) % sample_step == 0:
                # 记录平均时间
                iteration_times.append(accumulated_time / sample_step)
                accumulated_time = 0

            # 打印进度
            if (i + 1) % 100 == 0:  # 减少打印频率
                print(f"Iteration {i + 1}/{algorithm.max_iterations}, "
                      f"Best Fitness: {best_fitness:.4f}")

        return best_solution, best_fitness, history

1111
    # 替换optimize方法
    algorithm.optimize = timed_optimize

    # 运行算法
    best_solution, best_fitness, history = algorithm.optimize()
    total_time = time.time() - start_time

    # 恢复原始的optimize方法
    algorithm.optimize = original_optimize

    return best_solution, best_fitness, history, iteration_times, total_time

def dynamic_visualization(history, algorithm_name):
    """动态展示收敛过程"""
    plt.ion()
    plt.figure(figsize=(10, 6))
    for i, fitness in enumerate(history):
        plt.clf()
        plt.plot(history[:i + 1], label=f'{algorithm_name} Convergence', color='blue', linewidth=2)
        plt.title(f'{algorithm_name} Convergence Curve', fontsize=16)
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Fitness Value', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.pause(0.1)
    plt.ioff()
    plt.show()


def grid_search(env, algorithm_class, param_grid):
    """网格搜索优化算法参数"""
    best_params = None
    best_fitness = float('inf')

    for params in ParameterGrid(param_grid):
        algorithm = algorithm_class(env, params)
        _, fitness, _, _, _ = run_algorithm_with_timing(algorithm)
        if fitness < best_fitness:
            best_fitness = fitness
            best_params = params

    return best_params, best_fitness


def plot_convergence(history, algorithm_name):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history, label=f'{algorithm_name} Convergence', color='blue', linewidth=2)
    plt.title(f'{algorithm_name} Convergence Curve', fontsize=16)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Fitness Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{algorithm_name}_convergence.png', dpi=300)
    plt.show()


def adaptive_parameters(algorithm, iteration):
    """动态调整算法参数"""
    if iteration < algorithm.max_iterations // 2:
        algorithm.learning_rate = 0.8  # 前半段使用较高学习率





def main():
    # 修改算法配置以支持1000次迭代
    ALGORITHM_CONFIG['max_iterations'] = 1000

    # 初始化环境
    env = Environment(SIMULATION_CONFIG)

    # 运行性能测试
    print("\nRunning performance test...")
    env.performance_test(num_tests=1000)

    # 测试不同算法
    algorithms = {
        # "ACLHO_3_1": ACLHO_3_1,
        # "ACLHO_3_2": ACLHO_3_2,
        # "ACLHO_2": ACLHO_2,

        #"ACLHO_Base": ACLHO_Base,
        "PSO": PSO,
        #"GWO": GWO,
        "SAEO": SAEO,
        "IVYA": IVYA,
        "ACLHO_1": ACLHO_1,
    }

    # 运行算法并收集结果
    results = {}
    execution_times = {}

    for name, algorithm_class in algorithms.items():
        print(f"\nRunning {name}...")
        algorithm = algorithm_class(env, ALGORITHM_CONFIG)

        # 运行算法并记录时间
        best_solution, best_fitness, history, iteration_times, total_time = run_algorithm_with_timing(algorithm)

        results[name] = {
            "solution": best_solution,
            "fitness": best_fitness,
            "history": history,
            "iteration_times": iteration_times
        }
        execution_times[name] = total_time

        print(f"{name} completed in {total_time:.2f} seconds")
        print(f"Best Fitness: {best_fitness:.2f}")

    # 生成性能评估曲线
    plot_convergence_delay(results, sample_step=10)  # 传递采样步长参数

    # 绘制收敛曲线
    histories = {name: [result["history"]] for name, result in results.items()}  # 包装成二维
    plot_convergence_curves(histories, save_path="convergence.png")

    # 评估不同任务数量的性能 (10-100)
    task_ranges = list(range(10, 101, 10))  # [10, 20, ..., 100]
    evaluate_task_scaling(algorithms, SIMULATION_CONFIG, ALGORITHM_CONFIG, task_ranges)

    # 评估不同机器数量的性能 (100-200)
    robot_ranges = list(range(100, 201, 20))  # [100, 120, 140, 160, 180, 200]
    # 临时降低最大迭代次数以加快机器规模测试
    original_max_iterations = ALGORITHM_CONFIG['max_iterations']
    ALGORITHM_CONFIG['max_iterations'] = 100  # 临时设置为较小的值
    evaluate_robot_scaling(algorithms, SIMULATION_CONFIG, ALGORITHM_CONFIG, robot_ranges)
    # 恢复原始迭代次数
    ALGORITHM_CONFIG['max_iterations'] = original_max_iterations

    # 评估不同操作数量的性能 (50-300)
    operation_ranges = list(range(50, 301, 50))  # [50, 100, ..., 300]
    evaluate_operation_scaling(algorithms, SIMULATION_CONFIG, ALGORITHM_CONFIG, operation_ranges)

    # 打印比较结果
    print("\nAlgorithm Comparison:")
    for name, result in results.items():
        print(f"{name}: Fitness = {result['fitness']:.2f}, Time = {execution_times[name]:.2f}s")


if __name__ == "__main__":
    main()

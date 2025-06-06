import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(results):
    """绘制各算法的收敛曲线"""
    plt.figure(figsize=(10, 6))
    
    for name, result in results.items():
        if 'history' in result:
            plt.plot(result['history'], label=name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.title('Algorithm Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence.png')
    plt.close()

def plot_task_allocation(solution, env):
    """可视化任务分配结果"""
    plt.figure(figsize=(12, 8))
    
    # 绘制任务点
    for i, pos in enumerate(env.task_positions):
        if solution[i] >= 0:  # 已分配的任务
            plt.scatter(pos[0], pos[1], c='blue', s=100, alpha=0.6, label='Assigned Task' if i == 0 else "")
        else:  # 未分配的任务
            plt.scatter(pos[0], pos[1], c='red', s=100, alpha=0.6, label='Unassigned Task' if i == 0 else "")
    
    # 绘制机器人
    plt.scatter(env.robot_positions[:, 0], env.robot_positions[:, 1],
               c='green', marker='^', s=200, alpha=0.6, label='Robot')
    
    # 绘制分配关系
    for task_id, robot_id in enumerate(solution):
        if robot_id >= 0:
            task_pos = env.task_positions[task_id]
            robot_pos = env.robot_positions[robot_id]
            plt.plot([task_pos[0], robot_pos[0]], [task_pos[1], robot_pos[1]],
                    'k--', alpha=0.3)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Task Allocation Visualization')
    plt.legend()
    plt.grid(True)
    plt.savefig('task_allocation.png')
    plt.close()
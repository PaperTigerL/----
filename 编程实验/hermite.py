import numpy as np
import matplotlib.pyplot as plt

def hermite_interpolation(x_samples, y_samples, eval_points):
    n = len(x_samples)

    # 初始化差商数组
    f_divided_diff = np.zeros((n, n))

    # 计算一阶差商
    f_divided_diff[:, 0] = y_samples

    # 计算高阶差商
    for j in range(1, n):
        for i in range(n - j):
            f_divided_diff[i, j] = (f_divided_diff[i + 1, j - 1] - f_divided_diff[i, j - 1]) / (x_samples[i + j] - x_samples[i])

    # 计算插值结果
    result = np.zeros_like(eval_points)
    for i in range(n):
        term = f_divided_diff[0, i]
        for j in range(i):
            term *= (eval_points - x_samples[j])
        result += term

    return result

# 定义目标函数
def target_function(x, c, d, e, f):
    return c * np.sin(x) + e * np.cos(x)

# 设置参数
a, b = 0, 2 * np.pi
c, d, e, f = 1, 0, 1, 0
n = 10  # 采样点数
m = 5   # 实验点数

# 生成采样点和实验点
x_samples = np.linspace(a, b, n)
y_samples = target_function(x_samples, c, d, e, f)
eval_points = np.linspace(a, b, 1000)
true_values = target_function(eval_points, c, d, e, f)
experiment_points = np.linspace(a, b, m)

# 使用自己实现的分段三次Hermite插值
hermite_values = hermite_interpolation(x_samples, y_samples, eval_points)

# 计算平均误差
hermite_error = np.mean(np.abs(true_values - hermite_values))

# 绘图对比
plt.figure(figsize=(10, 6))
plt.plot(eval_points, true_values, label='True Function', linewidth=2)
plt.plot(eval_points, hermite_values, label=f'Custom Hermite Interpolation (Error: {hermite_error:.4f})')
plt.scatter(experiment_points, target_function(experiment_points, c, d, e, f), color='red', label='Experiment Points')
plt.legend()
plt.title('Interpolation Methods Comparison')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
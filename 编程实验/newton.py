import numpy as np
import matplotlib.pyplot as plt

def newton_interpolation(x_samples, y_samples, eval_points):
    n = len(x_samples) - 1
    coefficients = y_samples.copy()
    for j in range(1, n + 1):
        coefficients[j:] = (coefficients[j:] - coefficients[j - 1]) / (x_samples[j:] - x_samples[j - 1])
    result = coefficients[-1]
    for i in range(n - 1, -1, -1):
        result = result * (eval_points - x_samples[i]) + coefficients[i]
    return result

def target_function(x, c, d, e, f):
    return c * np.sin(x) + e * np.cos(x)
def compute_average_error(true_values, interpolated_values):
    return np.mean(np.abs(true_values - interpolated_values))
# 设置参数
a, b = 0, 2 * np.pi
c, d, e, f = 1, 0, 1, 0
n = 10
m = 5

# 生成采样点和实验点
x_samples = np.linspace(a, b, n + 1)
y_samples = target_function(x_samples, c, d, e, f)
eval_points = np.linspace(a, b, 1000)
true_values = target_function(eval_points, c, d, e, f)
experiment_points = np.linspace(a, b, m)

# 牛顿插值
newton_values = newton_interpolation(x_samples, y_samples, eval_points)

# 计算平均误差
newton_error = compute_average_error(true_values, newton_values)

# 绘图对比
plt.figure(figsize=(10, 6))
plt.plot(eval_points, true_values, label='True Function', linewidth=2)
plt.plot(eval_points, newton_values, label=f'Newton Interpolation (Error: {newton_error:.4f})')
plt.scatter(experiment_points, target_function(experiment_points, c, d, e, f), color='red', label='Experiment Points')
plt.legend()
plt.title('Interpolation Methods Comparison')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
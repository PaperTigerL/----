import numpy as np
import matplotlib.pyplot as plt


def target_function(x, c, d, e, f):
    return c * np.sin(x) + e * np.cos(x)
def vandermonde_interpolation(x_samples, y_samples, eval_points):
    n = len(x_samples)
    A = np.vander(x_samples, increasing=True)
    coefficients = np.linalg.solve(A, y_samples)

    # Evaluate the interpolating polynomial at the given points
    interpolated_values = np.polyval(coefficients[::-1], eval_points)

    return interpolated_values
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

# 范德蒙德多项式插值
vandermonde_values = vandermonde_interpolation(x_samples, y_samples, eval_points)

# 计算平均误差
vandermonde_error = compute_average_error(true_values, vandermonde_values)

# 绘图对比
plt.figure(figsize=(10, 6))
plt.plot(eval_points, true_values, label='True Function', linewidth=2)
plt.plot(eval_points, vandermonde_values, label=f'Vandermonde Interpolation (Error: {vandermonde_error:.4f})')
plt.scatter(experiment_points, target_function(experiment_points, c, d, e, f), color='red', label='Experiment Points')
plt.legend()
plt.title('Interpolation Methods Comparison')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
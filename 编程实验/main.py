import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def target_function(x, c, d, e, f):
    return c * np.sin(x) + e * np.cos(x)

def vandermonde_interpolation(x_samples, y_samples, eval_points):
    coefficients = np.polyfit(x_samples, y_samples, len(x_samples) - 1)
    interpolated_values = np.polyval(coefficients, eval_points)
    return interpolated_values

def lagrange_interpolation(x_samples, y_samples, eval_points):
    interpolated_values = np.zeros_like(eval_points, dtype=float)
    for i in range(len(x_samples)):
        term = y_samples[i]
        for j in range(len(x_samples)):
            if i != j:
                term *= (eval_points - x_samples[j]) / (x_samples[i] - x_samples[j])
        interpolated_values += term
    return interpolated_values

def newton_interpolation(x_samples, y_samples, eval_points):
    n = len(x_samples) - 1
    coefficients = y_samples.copy()
    for j in range(1, n + 1):
        coefficients[j:] = (coefficients[j:] - coefficients[j - 1]) / (x_samples[j:] - x_samples[j - 1])
    result = coefficients[-1]
    for i in range(n - 1, -1, -1):
        result = result * (eval_points - x_samples[i]) + coefficients[i]
    return result

def piecewise_linear_interpolation(x_samples, y_samples, eval_points):
    return np.interp(eval_points, x_samples, y_samples)

def hermite_interpolation(x_samples, y_samples, eval_points):
    x_samples_flat = x_samples.ravel()
    y_samples_flat = y_samples.ravel()

    hermite_interpolator = CubicSpline(x_samples_flat, y_samples_flat, bc_type='natural')
    return hermite_interpolator(eval_points)

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

# 拉格朗日插值
lagrange_values = lagrange_interpolation(x_samples, y_samples, eval_points)

# 牛顿插值
newton_values = newton_interpolation(x_samples, y_samples, eval_points)

# 分段线性插值
linear_values = piecewise_linear_interpolation(x_samples, y_samples, eval_points)

# 分段三次Hermite插值
hermite_values = hermite_interpolation(x_samples, y_samples, eval_points)

# 计算平均误差
vandermonde_error = compute_average_error(true_values, vandermonde_values)
lagrange_error = compute_average_error(true_values, lagrange_values)
newton_error = compute_average_error(true_values, newton_values)
linear_error = compute_average_error(true_values, linear_values)
hermite_error = compute_average_error(true_values, hermite_values)

# 绘图对比
plt.figure(figsize=(10, 6))
plt.plot(eval_points, true_values, label='True Function', linewidth=2)
plt.plot(eval_points, vandermonde_values, label=f'Vandermonde Interpolation (Error: {vandermonde_error:.4f})')
plt.plot(eval_points, lagrange_values, label=f'Lagrange Interpolation (Error: {lagrange_error:.4f})')
plt.plot(eval_points, newton_values, label=f'Newton Interpolation (Error: {newton_error:.4f})')
plt.plot(eval_points, linear_values, label=f'Piecewise Linear Interpolation (Error: {linear_error:.4f})')
plt.plot(eval_points, hermite_values, label=f'Piecewise Hermite Interpolation (Error: {hermite_error:.4f})')
plt.scatter(experiment_points, target_function(experiment_points, c, d, e, f), color='red', label='Experiment Points')
plt.legend()
plt.title('Interpolation Methods Comparison')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
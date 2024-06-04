import numpy as np
import matplotlib.pyplot as plt

def target_function(x, c):
    return 1 / (1 + c * x**2)

def legendre_polynomial(x, k):
    # 生成勒让德正交多项式的值
    return np.polynomial.legendre.Legendre.basis(k)(x)

def generate_sampling_points(a, b, n):
    # 在区间[a, b]上均匀采样n个点
    return np.linspace(a, b, n)

def generate_experiment_points(a, b, m):
    # 在区间[a, b]上生成m个实验点
    return np.linspace(a, b, m)

def calculate_approximation_error(true_values, approx_values):
    # 计算逼近函数与目标函数的平均误差
    return np.mean(np.abs(true_values - approx_values))

def least_squares_fit(x, y, k):
    # 使用最小二乘法拟合多项式
    A = np.vander(x, k + 1)
    coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.poly1d(coefficients)

# 参数设置
a, b = -1, 1  # 区间[a, b]
c = 2         # 参数c
m = 100       # 实验点个数

# 选择不同的逼近多项式次数k
for k in [1, 2, 3]:
    # 生成采样点
    sampling_points = generate_sampling_points(a, b, k)

    # 计算采样点上的目标函数值
    true_values = target_function(sampling_points, c)

    # 计算最佳平方逼近
    legendre_coefficients = np.polynomial.legendre.legfit(sampling_points, true_values, k)
    legendre_approximation = np.polynomial.legendre.Legendre(legendre_coefficients)

    # 计算最小二乘拟合
    experiment_points = generate_experiment_points(a, b, m)
    ls_fit = least_squares_fit(sampling_points, true_values, k)
    ls_values = ls_fit(experiment_points)

    # 计算平均误差
    legendre_error = calculate_approximation_error(target_function(experiment_points, c), legendre_approximation(experiment_points))
    ls_error = calculate_approximation_error(target_function(experiment_points, c), ls_values)

    # 可视化结果
    plt.figure(figsize=(8, 6))
    plt.plot(experiment_points, target_function(experiment_points, c), label='True Function')
    plt.plot(experiment_points, legendre_approximation(experiment_points), label=f'Legendre Approximation (k={k})')
    plt.plot(experiment_points, ls_values, label=f'Least Squares Fit (k={k})')
    plt.title(f'Comparison of Approximation Methods (k={k})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

    # 输出平均误差
    print(f'Legendre Approximation Error (k={k}): {legendre_error}')
    print(f'Least Squares Fit Error (k={k}): {ls_error}')
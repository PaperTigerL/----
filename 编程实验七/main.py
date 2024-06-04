import numpy as np


def fixed_point_iteration(g, x0, max_iter, epsilon):
    x = x0
    for k in range(max_iter):
        x_old = x
        x = g(x)
        if np.isnan(x) or np.isinf(x) or np.abs(x - x_old) < epsilon:
            return x, k + 1
    return x, max_iter


def steffensen_iteration(g, x0, max_iter, epsilon):
    x = x0
    for k in range(max_iter):
        x_old = x
        y = g(x)
        z = g(y)
        if z - 2 * y + x == 0:
            return x, k + 1
        x = x - (y - x) ** 2 / (z - 2 * y + x)
        if np.isnan(x) or np.isinf(x) or np.abs(x - x_old) < epsilon:
            return x, k + 1
    return x, max_iter


def newton_iteration(f, f_prime, x0, max_iter, epsilon):
    x = x0
    for k in range(max_iter):
        x_old = x
        derivative = f_prime(x)
        if derivative == 0:
            return x, k + 1
        x = x - f(x) / derivative
        if np.isnan(x) or np.isinf(x) or np.abs(x - x_old) < epsilon:
            return x, k + 1
    return x, max_iter


# Define functions and their derivatives
def f1(x):
    return x ** 2 - 3 * x + 2 - np.exp(x)


# 修复函数 g1(x) 并调整初始值
def g1(x):
    return f1(x) / (2 * x - 3)  # 符合固定点迭代要求

x0 = 0.5  # 更接近根的初始值

def f1_prime(x):
    return 2 * x - 3 - np.exp(x)


def f2(x):
    return x ** 3 + 2 * x ** 2 + 10 * x - 20


def g2(x):
    return x - f2(x) / (3 * x ** 2 + 4 * x + 10)


def f2_prime(x):
    return 3 * x ** 2 + 4 * x + 10


# Perform iterations
x0 = 1.0
max_iter = 100
epsilon = 1e-8

print("Fixed Point Iteration for f1:")
result_fp_f1, iterations_fp_f1 = fixed_point_iteration(g1, x0, max_iter, epsilon)
print("Root:", result_fp_f1)
print("Number of Iterations:", iterations_fp_f1)

print("\nSteffensen Iteration for f1:")
result_steff_f1, iterations_steff_f1 = steffensen_iteration(g1, x0, max_iter, epsilon)
print("Root:", result_steff_f1)
print("Number of Iterations:", iterations_steff_f1)

print("\nNewton Iteration for f1:")
result_newton_f1, iterations_newton_f1 = newton_iteration(f1, f1_prime, x0, max_iter, epsilon)
print("Root:", result_newton_f1)
print("Number of Iterations:", iterations_newton_f1)

print("\nFixed Point Iteration for f2:")
result_fp_f2, iterations_fp_f2 = fixed_point_iteration(g2, x0, max_iter, epsilon)
print("Root:", result_fp_f2)
print("Number of Iterations:", iterations_fp_f2)

print("\nSteffensen Iteration for f2:")
result_steff_f2, iterations_steff_f2 = steffensen_iteration(g2, x0, max_iter, epsilon)
print("Root:", result_steff_f2)
print("Number of Iterations:", iterations_steff_f2)

print("\nNewton Iteration for f2:")
result_newton_f2, iterations_newton_f2 = newton_iteration(f2, f2_prime, x0, max_iter, epsilon)
print("Root:", result_newton_f2)
print("Number of Iterations:", iterations_newton_f2)

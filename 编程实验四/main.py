import math

def f(x):
    return math.sqrt(x) * math.log(x)

def trapezoidal_rule(a, b, h):
    n = int((b - a) / h)
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    result *= h
    return result

def romberg_integration(a, b, epsilon):
    k_max = 20  # 最大迭代次数
    h = b - a
    R = [[0] * (k_max + 1) for _ in range(k_max + 1)]

    R[0][0] = 0.5 * h * (f(a) + f(b))
    for k in range(1, k_max + 1):
        h /= 2
        R[k][0] = 0.5 * R[k - 1][0] + h * sum(f(a + (2 * i - 1) * h) for i in range(1, 2 ** (k - 1) + 1))

        for j in range(1, k + 1):
            R[k][j] = R[k][j - 1] + (R[k][j - 1] - R[k - 1][j - 1]) / (4 ** j - 1)

        if abs(R[k][k] - R[k - 1][k - 1]) < epsilon:
            break

    return R[k][k], 2 ** k, h

# 输入参数
a = 1.0
b = 2.0
epsilon = 1e-6

# 使用复化梯形公式计算积分值、划分次数和步长h
trap_result = trapezoidal_rule(a, b, 0.1)

# 使用龙贝格算法计算积分值、划分次数和步长h
romberg_result, romberg_iterations, romberg_h = romberg_integration(a, b, epsilon)

# 输出结果
print("复化梯形公式：")
print("积分值：", trap_result)
print("划分次数：", int((b - a) / 0.1))
print("步长h：", 0.1)

print("\n龙贝格算法：")
print("积分值：", romberg_result)
print("划分次数：", romberg_iterations)
print("步长h：", romberg_h)
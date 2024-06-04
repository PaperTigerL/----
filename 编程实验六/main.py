import numpy as np

def gauss_seidel(A, b, x0, max_iter, epsilon):
    n = len(b)
    x = x0.copy()
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x - x_old, ord=np.inf) < epsilon:
            return x, k + 1
    return x, max_iter

def sor(A, b, x0, max_iter, epsilon, omega):
    n = len(b)
    x = x0.copy()
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (1 - omega) * x_old[i] + omega * (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x - x_old, ord=np.inf) < epsilon:
            return x, k + 1
    return x, max_iter

def test_methods():
    # Given matrices and vectors
    A = np.array([[31, -13, 0, 0, 0, -10, 0, 0, 0],
                  [-13, 35, -9, 0, -11, 0, 0, 0, 0],
                  [0, -9, 31, -10, 0, 0, 0, 0, 0],
                  [0, 0, -10, 79, -30, 0, 0, 0, -9],
                  [0, 0, 0, -30, 57, -7, 0, -5, 0],
                  [0, 0, 0, 0, -7, 47, -30, 0, 0],
                  [0, 0, 0, 0, 0, -30, 41, 0, 0],
                  [0, 0, 0, 0, -5, 0, 0, 27, -2],
                  [0, 0, 0, -9, 0, 0, 0, -2, 29]])

    b = np.array([-15, 27, -23, 0, -20, 12, -7, 7, 10])

    # Initial values
    x0 = np.zeros_like(b)

    # Parameters
    max_iter = 1000
    epsilon = 1e-8

    # Test methods
    omega_values = np.arange(1, 100) / 50.0
    for omega in omega_values:
        print(f"\nTesting with Omega = {omega}:")
        result_gs, iterations_gs = gauss_seidel(A, b, x0, max_iter, epsilon)
        print("Gauss-Seidel Iteration:")
        print("Roots:", result_gs)
        print("Number of Iterations:", iterations_gs)

        result_sor, iterations_sor = sor(A, b, x0, max_iter, epsilon, omega)
        print("\nSOR Iteration:")
        print("Roots:", result_sor)
        print("Number of Iterations:", iterations_sor)

# Run the tests
test_methods()

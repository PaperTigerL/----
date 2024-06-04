import numpy as np


def partial_pivot(augmented_matrix, current_row):
    max_row = np.argmax(np.abs(augmented_matrix[current_row:, current_row])) + current_row
    augmented_matrix[[current_row, max_row], :] = augmented_matrix[[max_row, current_row], :]


def forward_elimination(augmented_matrix):
    n = len(augmented_matrix)
    for i in range(n):
        pivot = augmented_matrix[i, i]
        augmented_matrix[i, :] = augmented_matrix[i, :] / pivot  # Change here
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i]
            augmented_matrix[j, :] -= factor * augmented_matrix[i, :]


def back_substitution(augmented_matrix):
    n = len(augmented_matrix)
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:])
    return x


def gauss_elimination(augmented_matrix):
    n = len(augmented_matrix)

    for i in range(n):
        partial_pivot(augmented_matrix, i)
        forward_elimination(augmented_matrix)

    x = back_substitution(augmented_matrix)
    return augmented_matrix, x


def generate_random_matrix_and_vector(n):
    A = np.random.randint(-10, 10, size=(n, n)).astype(float)  # Change here
    b = np.random.randint(-10, 10, size=(n, 1)).astype(float)  # Change here
    return A, b


def main():
    # Test with the given matrix A and vector b
    A_given = np.array([[31, -13, 0, 0, 0, -10, 0, 0, 0],
                        [-13, 35, -9, 0, -11, 0, 0, 0, 0],
                        [0, -9, 31, -10, 0, 0, 0, 0, 0],
                        [0, 0, -10, 79, -30, 0, 0, 0, -9],
                        [0, 0, 0, -30, 57, -7, 0, -5, 0],
                        [0, 0, 0, 0, -7, 47, -30, 0, 0],
                        [0, 0, 0, 0, 0, -30, 41, 0, 0],
                        [0, 0, 0, 0, -5, 0, 0, 27, -2],
                        [0, 0, 0, -9, 0, 0, 0, -2, 29]])

    b_given = np.array([[-15],
                        [27],
                        [-23],
                        [0],
                        [-20],
                        [12],
                        [-7],
                        [7],
                        [10]])

    augmented_matrix_given = np.concatenate((A_given, b_given), axis=1)

    # Solve the system of linear equations
    augmented_matrix_result, solution_vector = gauss_elimination(augmented_matrix_given)

    # Print the results
    print("Augmented Matrix after elimination:\n", augmented_matrix_result)
    print("\nSolution vector x:\n", solution_vector)

    # Test with a randomly generated matrix and vector
    n = 20  # You can adjust the value of n as needed
    A_random, b_random = generate_random_matrix_and_vector(n)
    augmented_matrix_random = np.concatenate((A_random, b_random), axis=1)

    # Solve the system of linear equations for the random matrix and vector
    augmented_matrix_result_random, solution_vector_random = gauss_elimination(augmented_matrix_random)

    # Print the results for the random case
    print("\nRandomly generated matrix A:\n", A_random)
    print("\nRandomly generated vector b:\n", b_random)
    print("\nAugmented Matrix after elimination (random case):\n", augmented_matrix_result_random)
    print("\nSolution vector x (random case):\n", solution_vector_random)


if __name__ == "__main__":
    main()

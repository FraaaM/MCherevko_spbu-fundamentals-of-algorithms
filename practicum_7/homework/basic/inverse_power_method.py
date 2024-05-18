import numpy as np
import matplotlib.pyplot as plt

from src.common import NDArrayFloat


def power_method(A: NDArrayFloat, n_iters: int) -> NDArrayFloat:
    n = A.shape[0]
    # Начальное приближение (может быть случайным)
    b_k = np.random.rand(n)
    for _ in range(n_iters):
        # Умножаем матрицу на вектор
        b_k1 = np.dot(A, b_k)
        # Нормируем вектор
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    # Приближение к собственному значению
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k)
    return eigenvalue

def inverse_power_method(A: NDArrayFloat, n_iters: int) -> NDArrayFloat:
    # Обратная матрица
    A_inv = np.linalg.inv(A)
    # Используем метод степенной итерации для обратной матрицы
    smallest_eigenvalue_inv = power_method(A_inv, n_iters)
    # Наименьшее по модулю собственное значение исходной матрицы
    smallest_eigenvalue = 1 / smallest_eigenvalue_inv
    return smallest_eigenvalue

if __name__ == "__main__":
    A = np.array(
        [
            [4.0, 1.0, -1.0, 2.0],
            [1.0, 4.0, 1.0, -1.0],
            [-1.0, 1.0, 4.0, 1.0],
            [2.0, -1.0, 1.0, 1.0],
        ]
    )
    eigval = inverse_power_method(A, n_iters=10)
    print("Наименьшее по модулю собственное значение:", eigval)
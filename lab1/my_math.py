import math
import numpy as np


def integral(f, a, b, N, *params):
    """Расчёт интеграла."""
    dx = (b - a) / N
    result = 0.5 * f(a, *params) + 0.5 * f(b, *params)

    for i in range(1, N):
        x_i = a + i * dx
        result += f(x_i, *params)

    integral_value = result * dx
    return integral_value


def factorial(n):
    """Расчёт факториала."""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def my_sin(x, terms=15):
    """Расчёт синуса."""
    is_array = isinstance(x, np.ndarray)
    x = np.asarray(x)

    x = x % (2 * math.pi)

    if is_array:
        x[x > math.pi] -= 2 * math.pi
    else:
        if x > math.pi:
            x -= 2 * math.pi

    sin_x = np.zeros_like(x, dtype=float)

    for n in range(terms):
        sign = (-1) ** n
        sin_x += sign * (x ** (2 * n + 1)) / factorial(2 * n + 1)

    return sin_x if is_array else float(sin_x)


def my_cos(x, terms=15):
    """Расчёт косинуса."""
    is_array = isinstance(x, np.ndarray)
    x = np.asarray(x)

    x = x % (2 * math.pi)

    if is_array:
        x[x > math.pi] -= 2 * math.pi
    else:
        if x > math.pi:
            x -= 2 * math.pi

    cos_x = np.zeros_like(x, dtype=float)

    for n in range(terms):
        sign = (-1) ** n
        cos_x += sign * (x ** (2 * n)) / factorial(2 * n)

    return cos_x if is_array else float(cos_x)


def my_log2(x, tolerance=1e-10):
    """Расчёт логарифма по основанию 2."""
    if x <= 0:
        raise ValueError("Логарифм определён только для положительных чисел")

    result = 0.0
    while x > 1:
        x /= 2
        result += 1

    while x < 1:
        x *= 2
        result -= 1

    guess = 0
    while abs(x - 1) > tolerance:
        x = (x + 1) / 2
        guess += 1

    return result + guess


def power_fast(x, n):
    """Расчет возведения в степень."""
    if n == 0:
        return 1
    elif n < 0:
        return 1 / power_fast(x, -n)

    half = power_fast(x, n // 2)
    return half * half if n % 2 == 0 else x * half * half


def my_sqrt(x, tolerance=1e-10):
    """Расчёт квадратного корня."""
    if x < 0:
        raise ValueError("Невозможно извлечь квадратный корень из отрицательного числа.")

    guess = x / 2.0

    while abs(guess * guess - x) > tolerance:
        guess = (guess + x / guess) / 2.0

    return guess


def DFT(x):
    """Дискретное преобразование Фурье (ДПФ)."""
    N = len(x)
    X = []
    for k in range(N):
        sum_real = 0
        sum_imag = 0
        for n in range(N):
            angle = (2 * math.pi * k * n) / N
            sum_real += x[n] * my_cos(angle)
            sum_imag += -x[n] * my_sin(angle)
        X.append(sum_real + 1j * sum_imag)
    return np.array(X)


def FFT(x):
    """Быстрое преобразование Фурье (БПФ)."""
    N = len(x)
    if N <= 1:
        return np.array(x)

    even = FFT(x[::2])
    odd = FFT(x[1::2])

    result = np.zeros(N, dtype=complex)

    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle_factor = complex(my_cos(angle), my_sin(angle))
        result[k] = even[k] + twiddle_factor * odd[k]
        result[k + N // 2] = even[k] - twiddle_factor * odd[k]

    return np.array(result)


def IDFT(X):
    """Обратное дискретное преобразование Фурье (ОДПФ)."""
    N = len(X)
    x = []
    for n in range(N):
        sum_real = 0
        sum_imag = 0
        for k in range(N):
            angle = (2 * math.pi * k * n) / N
            sum_real += X[k].real * my_cos(angle) - X[k].imag * my_sin(angle)
            sum_imag += X[k].real * my_sin(angle) + X[k].imag * my_cos(angle)
        x.append((sum_real + 1j * sum_imag) / N)
    return np.array(x)


def IFFT(X):
    """Обратное быстрое преобразование Фурье (ОБПФ)."""
    N = len(X)
    if N <= 1:
        return np.array(X)

    even = IFFT(X[::2])
    odd = IFFT(X[1::2])

    result = np.zeros(N, dtype=complex)

    for k in range(N // 2):
        angle = 2 * math.pi * k / N
        twiddle_factor = complex(my_cos(angle), my_sin(angle))
        result[k] = (even[k] + twiddle_factor * odd[k]) / 2
        result[k + N // 2] = (even[k] - twiddle_factor * odd[k]) / 2

    return np.array(result)

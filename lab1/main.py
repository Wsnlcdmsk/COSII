import matplotlib.pyplot as plt
import numpy as np
import math
from my_math import my_cos, my_sin, integral, power_fast, DFT, FFT, IDFT, IFFT, my_log2, my_sqrt
import time
from wav import save_wave


a1, b1, v1 = 1, 3, 1
a2, b2, v2 = 2, 5, 4
phi0 = -1.9
N = 256
D = 23

epsilon = 1e-15

attenuation_factor = 0.5
cutoff_frequency = 1


def U1(a1, b1, v1, t, phi0):
    return a1 * power_fast(my_sin(v1 * t + phi0), b1)


def U2(a2, b2, v2, t, phi0):
    return a2 * power_fast(my_cos(v2 * t + phi0), b2)


def product_U1_U2(t, a1, b1, v1, a2, b2, v2, phi0):
    return U1(a1, b1, v1, t, phi0) * U2(a2, b2, v2, t, phi0)


def f(t, a1, b1, v1, a2, b2, v2, phi0):
    return U1(a1, b1, v1, t, phi0) + U2(a2, b2, v2, t, phi0)


def modify_spectrum(Y, attenuation_factor, cutoff_frequency, sample_rate):
    """Изменяет амплитуду в частотной области и обнуляет низкие частоты."""
    N = len(Y)
    freq_step = sample_rate / N
    modified_Y = np.array(Y, dtype=complex)

    for i in range(N):
        freq = i * freq_step

        if freq < cutoff_frequency:
            modified_Y[i] = 0
        else:
            modified_Y[i] *= attenuation_factor

    return modified_Y


def calculate_fft_efficiency(N):
    """Расчёт  эффективности."""
    operations_fft = N * my_log2(N)
    operations_dft = N ** 2
    speedup = operations_dft / operations_fft

    x = np.random.random(N)
    start_time = time.time()
    DFT(x)
    dft_time = time.time() - start_time

    start_time = time.time()
    FFT(x)
    fft_time = time.time() - start_time

    print(f"🔹 Количество точек: {N}")
    print(f"📌 Операции ДПФ: {operations_dft:.0f}")
    print(f"📌 Операции БПФ: {operations_fft:.0f}")
    print(f"⚡ БПФ быстрее ДПФ в {speedup:.2f} раз!")
    print(f"⏱ Время ДПФ: {dft_time:.6f} сек")
    print(f"⏱ Время БПФ: {fft_time:.6f} сек")
    print(f"⏳ БПФ быстрее ДПФ на {dft_time - fft_time:.6f} сек")


result = integral(product_U1_U2, phi0, phi0 + 2 * math.pi, N, a1, b1, v1, a2, b2, v2, phi0)
print("Результат интегрирования:", result)

if abs(result) < epsilon:
    print(f"Результат интегрирования близок к 0 с точностью {epsilon}, " +
          "следователтно фунцкии ортогональны")
else:
    print(f"Результат интегрирования не равен 0 с точностью {epsilon} " +
          "следователтно фунцкии не ортогональны")

t = np.linspace(0, D, N)
y = f(t, a1, b1, v1, a2, b2, v2, phi0)

start_time = time.time()
y_dft = DFT(y)
dft_time = time.time() - start_time

start_time = time.time()
y_fft = FFT(y)
fft_time = time.time() - start_time

start_time = time.time()
y_reconstructed_dft = IDFT(y_dft).real
idft_time = time.time() - start_time

start_time = time.time()
y_reconstructed_fft = IFFT(y_fft).real
ifft_time = time.time() - start_time

calculate_fft_efficiency(N)


def calculate_frequencies(N, D):
    sample_rate = N / D
    freq = [(i * sample_rate) / N for i in range(N // 2)]
    return freq


freq = calculate_frequencies(N, D)


def calculate_amplitude(Y):
    amplitude = [my_sqrt(y.real**2 + y.imag**2) for y in Y]
    return amplitude


amplitude_dft = calculate_amplitude(y_dft)
amplitude_fft = calculate_amplitude(y_fft)

y_fft_modified = modify_spectrum(y_fft, attenuation_factor, cutoff_frequency, N / D)
amplitude_fft_modified = calculate_amplitude(y_fft_modified)

print(f"📌Время выполнения ДПФ: {dft_time:.5f} сек")
print(f"📌Время выполнения БПФ: {fft_time:.5f} сек")
print(f"⏳Время выполнения БПФ на {dft_time - fft_time:.5f} сек, быстрее чем ДПФ")
print(f"📌Время выполнения ОДПФ: {idft_time:.5f} сек")
print(f"📌Время выполнения ОБПФ: {ifft_time:.5f} сек")
print(f"⏳Время выполнения ОБПФ на {idft_time - ifft_time:.5f} сек, быстрее чем ОДПФ")

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(t, y, label="Оригинальный f(t)")
plt.title("График f(t)")
plt.legend()
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(freq[:N//2], amplitude_fft_modified[:N//2], label="Измененная АЧХ", color='purple')
plt.title("Амплитудный спектр после изменений")
plt.legend()
plt.grid()

plt.subplot(3, 2, 3)
plt.plot(freq[:N//2], amplitude_fft[:N//2], label="АЧХ (ДПФ)", color='b')
plt.title("Амплитудный спектр (ДПФ)")
plt.legend()
plt.grid()

plt.subplot(3, 2, 4)
plt.plot(freq[:N//2], amplitude_fft[:N//2], label="АЧХ (БПФ)", color='r')
plt.title("Амплитудный спектр (БПФ)")
plt.legend()
plt.grid()

plt.subplot(3, 2, 5)
plt.plot(t, y_reconstructed_dft, label="Восстановленный сигнал (ОДПФ)", color='g')
plt.title("Восстановленный сигнал (ОДПФ)")
plt.legend()
plt.grid()

plt.subplot(3, 2, 6)
plt.plot(t, y_reconstructed_fft, label="Восстановленный сигнал (ОБПФ)", color='m')
plt.title("Восстановленный сигнал (ОБПФ)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

save_wave("🎧y_reconstructed_dft.wav", y_reconstructed_dft.real, round(N/D))
print("🎵Аудиофайл 'y_reconstructed_dft.wav' сохранен.")

save_wave("🎧y_reconstructed_fft.wav", y_reconstructed_fft.real, round(N/D))
print("🎵Аудиофайл 'y_reconstructed_fft.wav' сохранен.")

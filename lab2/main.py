import numpy as np
import matplotlib.pyplot as plt
from wav import save_wave
from my_math import (my_cos, my_sin, power_fast, convolution_manual,
                     correlation_manual, my_log2, correlation_fft,
                     convolution_fft)

a1, b1, v1 = 1, 3, 1
a2, b2, v2 = 2, 5, 4
phi0 = -1.9
N = 256
D = 23


num_operations_conv = 2 * (N * N)
num_operations_corr = 2 * (N * N)
num_operations_conv_fft = N * my_log2(N)
num_operations_corr_fft = N * my_log2(N)

t = np.linspace(0, D, N)

x_t = a1 * power_fast(my_sin(v1 * t + phi0), b1)
y_t = a2 * power_fast(my_cos(v2 * t + phi0), b2)

############################################### x y
z1_t = convolution_manual(x_t, y_t)
t1_conv = np.linspace(0, D, len(z1_t))
z1_corr = correlation_manual(x_t, y_t)
t1_corr = np.linspace(0, D, len(z1_corr))
z2_t = convolution_fft(x_t, y_t)
t2_conv = np.linspace(0, D, len(z2_t))
z2_corr = correlation_fft(x_t, y_t)
t2_corr = np.linspace(0, D, len(z2_corr))
################################################ x x
z3_t = convolution_manual(x_t, x_t)
t3_conv = np.linspace(0, D, len(z3_t))
z3_corr = correlation_manual(x_t, x_t)
t3_corr = np.linspace(0, D, len(z3_corr))
z4_t = convolution_fft(x_t, x_t)
t4_conv = np.linspace(0, D, len(z4_t))
z4_corr = correlation_fft(x_t, x_t)
t4_corr = np.linspace(0, D, len(z4_corr))
################################################ y y
z5_t = convolution_manual(y_t, y_t)
t5_conv = np.linspace(0, D, len(z5_t))
z5_corr = correlation_manual(y_t, y_t)
t5_corr = np.linspace(0, D, len(z5_corr))
z6_t = convolution_fft(y_t, y_t)
t6_conv = np.linspace(0, D, len(z6_t))
z6_corr = correlation_fft(y_t, y_t)
t6_corr = np.linspace(0, D, len(z6_corr))
###############################################


# Создаем три окна для графиков

# Окно 1: для первых 4 графиков
fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))
axs1[0, 0].plot(t1_conv, z1_t)
axs1[0, 0].set_title('Convolution: x_t * y_t')
axs1[0, 1].plot(t1_corr, z1_corr)
axs1[0, 1].set_title('Correlation: x_t * y_t')
axs1[1, 0].plot(t2_conv, z2_t)
axs1[1, 0].set_title('Convolution (FFT): x_t * y_t')
axs1[1, 1].plot(t2_corr, z2_corr)
axs1[1, 1].set_title('Correlation (FFT): x_t * y_t')
fig1.tight_layout()
plt.show()

# Окно 2: для следующих 4 графиков
fig2, axs2 = plt.subplots(2, 2, figsize=(10, 8))
axs2[0, 0].plot(t3_conv, z3_t)
axs2[0, 0].set_title('Convolution: x_t * x_t')
axs2[0, 1].plot(t3_corr, z3_corr)
axs2[0, 1].set_title('Correlation: x_t * x_t')
axs2[1, 0].plot(t4_conv, z4_t)
axs2[1, 0].set_title('Convolution (FFT): x_t * x_t')
axs2[1, 1].plot(t4_corr, z4_corr)
axs2[1, 1].set_title('Correlation (FFT): x_t * x_t')
fig2.tight_layout()
plt.show()

# Окно 3: для последних 4 графиков
fig3, axs3 = plt.subplots(2, 2, figsize=(10, 8))
axs3[0, 0].plot(t5_conv, z5_t)
axs3[0, 0].set_title('Convolution: y_t * y_t')
axs3[0, 1].plot(t5_corr, z5_corr)
axs3[0, 1].set_title('Correlation: y_t * y_t')
axs3[1, 0].plot(t6_conv, z6_t)
axs3[1, 0].set_title('Convolution (FFT): y_t * y_t')
axs3[1, 1].plot(t6_corr, z6_corr)
axs3[1, 1].set_title('Correlation (FFT): y_t * y_t')
fig3.tight_layout()
plt.show()


print("Э")

# Сохранение в музыкальный файл
save_wave("convolution_output.wav", z1_t)
save_wave("convolution_fft_output.wav", z2_t)
print("Файл convolution_output.wav сохранён!")
print("Файл convolution_fft_output.wav сохранён!")

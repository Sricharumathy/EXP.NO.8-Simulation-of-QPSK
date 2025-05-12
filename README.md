# EXP.NO.8-Simulation-of-QPSK

# AIM
To simulate Quadrature Phase Shift Keying (QPSK) modulation using Python and visualize its I (in-phase), Q (quadrature), and combined waveforms along with the input bit stream.

# SOFTWARE REQUIRED
1.Python (Version 3.6 or above)<br>
2.NumPy Library<br>
3.Matplotlib Library<br>
4.Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)<br>

# ALGORITHMS
## 1.Input Generation
Generate a random binary sequence where each QPSK symbol represents two bits.
## 2.Bit Mapping
Split the sequence into even and odd bits (I and Q respectively).<br>
Map 0 → -1 and 1 → +1.
## 3.Carrier Generation
Generate cosine (for I) and sine (for Q) carrier signals.<br>
Multiply mapped bits with respective carriers.<br>
## 4.QPSK Signal Construction
Combine I and Q carriers for each symbol.<br>
Concatenate all symbols to form the complete QPSK signal.<br>
## 5.Plotting
Plot the input bits as step signals.<br>
Plot I (cosine) and Q (sine) signals.<br>
Plot the combined QPSK modulated waveform.<br>

# PROGRAM
```
import numpy as np
import matplotlib.pyplot as plt

num_symbols = 10  # Number of QPSK symbols (each with 2 bits)
T = 1.0           # Symbol period
fs = 100.0        # Sampling frequency
t = np.arange(0, T, 1/fs)

# Generate 2 bits per symbol
bits = np.random.randint(0, 2, num_symbols * 2)

# Separate into I (cosine) and Q (sine) bits
i_bits = bits[0::2]  # Even-indexed bits
q_bits = bits[1::2]  # Odd-indexed bits

# Map bits: 0 → -1, 1 → +1
i_values = 2 * i_bits - 1
q_values = 2 * q_bits - 1

# Initialize signal arrays
i_signal = np.array([])
q_signal = np.array([])
combined_signal = np.array([])
i_bit_signal = np.array([])
q_bit_signal = np.array([])
symbol_times = []

for i in range(num_symbols):
    i_carrier = i_values[i] * np.cos(2 * np.pi * t / T)
    q_carrier = q_values[i] * np.sin(2 * np.pi * t / T)
    
    i_bit_wave = np.ones_like(t) * i_bits[i]
    q_bit_wave = np.ones_like(t) * q_bits[i]
    
    symbol_times.append(i * T)
    
    i_signal = np.concatenate((i_signal, i_carrier))
    q_signal = np.concatenate((q_signal, q_carrier))
    combined_signal = np.concatenate((combined_signal, i_carrier + q_carrier))
    i_bit_signal = np.concatenate((i_bit_signal, i_bit_wave))
    q_bit_signal = np.concatenate((q_bit_signal, q_bit_wave))

t_total = np.arange(0, num_symbols * T, 1/fs)

# Plotting
plt.figure(figsize=(14, 11))

# Input bits plot
plt.subplot(4, 1, 1)
plt.plot(t_total, i_bit_signal, label='I bits (even)', drawstyle='steps-post', color='blue')
plt.plot(t_total, q_bit_signal, label='Q bits (odd)', drawstyle='steps-post', color='orange')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 1.1, f'{i_bits[i]}{q_bits[i]}', fontsize=11, color='black')
plt.ylim(-0.5, 1.5)
plt.title('Input Bit Stream (I and Q Bits)')
plt.xlabel('Time')
plt.ylabel('Bit Value')
plt.grid(True)
plt.legend()

# In-phase (cosine) component
plt.subplot(4, 1, 2)
plt.plot(t_total, i_signal, label='In-phase (cos)', color='blue')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0.8, f'{i_bits[i]}', fontsize=12, color='black')
plt.title('In-phase Component (Cosine)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Quadrature (sine) component
plt.subplot(4, 1, 3)
plt.plot(t_total, q_signal, label='Quadrature (sin)', color='orange')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0.8, f'{q_bits[i]}', fontsize=12, color='black')
plt.title('Quadrature Component (Sine)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Combined QPSK waveform
plt.subplot(4, 1, 4)
plt.plot(t_total, combined_signal, label='QPSK Signal = I + Q', color='green')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0.8, f'{i_bits[i]}{q_bits[i]}', fontsize=12, color='black')
plt.title('Combined QPSK Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```
# OUTPUT
![qpsk wfrm](https://github.com/user-attachments/assets/3a914a2a-db46-4ef2-bfd7-23bde09a6c06)
<br>
<br>
<br>

# RESULT / CONCLUSIONS
The QPSK modulation was successfully simulated.Each symbol carries 2 bits, split between the in-phase and quadrature components.The resultant QPSK signal accurately reflects the phase changes based on the bit values.
The graphical outputs validate the principle of QPSK where the signal changes phase according to the input bit pair.

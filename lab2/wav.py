import wave
import struct


def save_wave(filename, signal, sample_rate=44100):
    """Сохранение сигнала в WAV-файл"""
    with wave.open(filename, 'w') as wav_file:
        n_channels = 1
        sampwidth = 2
        n_frames = len(signal)
        comp_type = "NONE"
        comp_name = "not compressed"
        wav_file.setparams((n_channels, sampwidth, sample_rate,
                            n_frames, comp_type, comp_name))

        max_amplitude = max(abs(signal))
        signal = (signal / max_amplitude) * 32767

        for s in signal:
            wav_file.writeframes(struct.pack('<h', int(s)))

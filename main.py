import pyedflib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, cutoff_frequency, sampling_frequency, order=4):
    """
    Applies a low-pass Butterworth filter to the data.

    Parameters:
        data (array-like): Input signal.
        cutoff_frequency (float): Cutoff frequency for the filter in Hz.
        sampling_frequency (float): Sampling rate of the signal in Hz.
        order (int): Order of the Butterworth filter.

    Returns:
        array-like: Filtered signal.
    """
    nyquist_frequency = 0.5 * sampling_frequency
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def compute_fft(signal, sampling_frequency):
    """
    Computes the FFT and corresponding frequency values for a given signal.

    Parameters:
        signal (array-like): Input signal.
        sampling_frequency (float): Sampling rate in Hz.

    Returns:
        tuple: (frequency_values, fft_magnitude)
    """
    number_of_samples = len(signal)
    fft_magnitude = np.abs(fft(signal)[: number_of_samples // 2])
    frequency_values = fftfreq(number_of_samples, 1 / sampling_frequency)[: number_of_samples // 2]
    return frequency_values, fft_magnitude


def load_edf_file(file_path):
    """
    Loads the first signal from an EDF file.

    Parameters:
        file_path (str): Path to the EDF file.

    Returns:
        tuple: (raw_signal, sampling_frequency, signal_labels, number_of_signals)
    """
    with pyedflib.EdfReader(file_path) as edf_reader:
        number_of_signals = edf_reader.signals_in_file
        signal_labels = edf_reader.getSignalLabels()
        sampling_frequency = edf_reader.getSampleFrequency(0)
        raw_signal = edf_reader.readSignal(0)
    return raw_signal, sampling_frequency, signal_labels, number_of_signals


def analyze_signal(raw_signal, sampling_frequency, cutoff_frequency=200):
    """
    Filters the raw signal and computes statistics along with its FFT.

    Parameters:
        raw_signal (array-like): Input raw signal.
        sampling_frequency (float): Sampling rate in Hz.
        cutoff_frequency (float): Cutoff frequency for the low-pass filter.

    Returns:
        tuple: (filtered_signal, stats, dominant_frequency, frequency_values, fft_filtered)
            - filtered_signal: Signal after low-pass filtering.
            - stats: Dictionary containing RMS, average, std, variance, and peak-to-peak amplitude.
            - dominant_frequency: Frequency with maximum FFT magnitude in the filtered signal.
            - frequency_values: Frequency axis for FFT.
            - fft_filtered: FFT magnitude of the filtered signal.
    """
    # Filter the signal
    filtered_signal = butter_lowpass_filter(raw_signal, cutoff_frequency, sampling_frequency)

    # Compute statistics
    rms = np.sqrt(np.mean(filtered_signal ** 2))
    average = np.mean(filtered_signal)
    std = np.std(filtered_signal)
    variance = np.var(filtered_signal)
    ptp = np.max(filtered_signal) - np.min(filtered_signal)

    stats = {
        "rms": rms,
        "average": average,
        "std": std,
        "variance": variance,
        "ptp": ptp
    }

    # Frequency analysis on the filtered signal
    frequency_values, fft_filtered = compute_fft(filtered_signal, sampling_frequency)
    dominant_frequency = frequency_values[np.argmax(fft_filtered)]

    return filtered_signal, stats, dominant_frequency, frequency_values, fft_filtered


def plot_results(time_values, raw_signal, filtered_signal, optimal_sine_wave,
                 stats, dominant_frequency, freq_values, fft_raw, fft_filtered):
    """
    Plots the original signal, filtered signal with optimal sine wave overlay,
    and the FFTs for both the raw and filtered signals. The figure is saved as a
    vector PDF file.

    Parameters:
        time_values (array-like): Time vector in seconds.
        raw_signal (array-like): Original unfiltered signal.
        filtered_signal (array-like): Filtered signal.
        optimal_sine_wave (array-like): Sine wave generated from the dominant frequency.
        stats (dict): Dictionary of computed statistics.
        dominant_frequency (float): Dominant frequency of the filtered signal.
        freq_values (array-like): Frequency axis for FFT plots.
        fft_raw (array-like): FFT magnitude of the raw signal.
        fft_filtered (array-like): FFT magnitude of the filtered signal.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(11, 8.5))  # 8.5 x 11 inches landscape

    # Subplot 1: Original signal with y-ticks every 500 mV
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time_values, raw_signal, color="blue", label="Original Signal (mV)")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude (mV)")
    ax1.set_title("Original Signal")
    ax1.legend()
    ylim1 = ax1.get_ylim()
    ax1.set_yticks(np.arange(np.floor(ylim1[0] / 500) * 500,
                             np.ceil(ylim1[1] / 500) * 500 + 500, 500))

    # Subplot 2: Filtered signal vs. Optimal sine wave with y-ticks every 500 mV
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(time_values, filtered_signal, color="blue", label="Filtered Signal (mV)")
    ax2.plot(time_values, optimal_sine_wave, color="orange",
             label=f"Optimal Sine Wave ({dominant_frequency:.2f} Hz)", linestyle="--")
    ax2.axhline(stats['average'], color="red", linestyle="--",
                label=f"Mean: {stats['average']:.3f} mV")
    ax2.axhline(stats['rms'], color="green", linestyle="--",
                label=f"RMS: {stats['rms']:.3f} mV")
    ax2.axhline(-stats['rms'], color="green", linestyle="--")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude (mV)")
    ax2.set_title("Filtered Signal vs. Optimal Sine Wave")
    ax2.legend()
    ylim2 = ax2.get_ylim()
    ax2.set_yticks(np.arange(np.floor(ylim2[0] / 500) * 500,
                             np.ceil(ylim2[1] / 500) * 500 + 500, 500))

    # Subplot 3: FFT of the raw signal with logarithmic y-axis
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(freq_values, fft_raw, color="blue", label="FFT of Raw Signal")
    ax3.set_yscale("log")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Magnitude (log scale)")
    ax3.set_title("FFT of Raw Signal")
    ax3.legend()

    # Subplot 4: FFT of the filtered signal with logarithmic y-axis
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(freq_values, fft_filtered, color="orange", label="FFT of Filtered Signal")
    ax4.set_yscale("log")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Magnitude (log scale)")
    ax4.set_title("FFT of Filtered Signal")
    ax4.legend()

    plt.tight_layout()

    # Save the figure as a vector PDF
    plt.savefig("vector_graph.pdf", format="pdf")

    plt.show()


def main():
    # File to be processed
    file_path = "waveform.edf"

    # Load EDF file and print header information
    raw_signal, sampling_frequency, signal_labels, number_of_signals = load_edf_file(file_path)
    print(f"EDF file contains {number_of_signals} signals: {signal_labels}")
    print(f"Sampling Frequency: {sampling_frequency} Hz")
    print("First 10 samples of signal:", raw_signal[:10])

    # Analyze signal (filtering, stats, FFT)
    cutoff_frequency = 200  # in Hz
    filtered_signal, stats, dominant_frequency, freq_values, fft_filtered = analyze_signal(
        raw_signal, sampling_frequency, cutoff_frequency
    )

    # Compute FFT for the raw (unfiltered) signal
    freq_values_raw, fft_raw = compute_fft(raw_signal, sampling_frequency)

    # Generate time vector
    time_values = np.arange(len(filtered_signal)) / sampling_frequency

    # Generate an optimal sine wave based on the dominant frequency.
    # Scale the sine wave so that its RMS matches the RMS of the real signal.
    amplitude_scaled = stats['rms'] * np.sqrt(2)
    optimal_sine_wave = amplitude_scaled * np.sin(2 * np.pi * dominant_frequency * time_values)

    # Plot the results and export as a vector PDF
    plot_results(time_values, raw_signal, filtered_signal, optimal_sine_wave,
                 stats, dominant_frequency, freq_values, fft_raw, fft_filtered)

    # Print computed statistics
    print(f"Root Mean Square: {stats['rms']:.6f} mV")
    print(f"Average Value: {stats['average']:.6f} mV")
    print(f"Standard Deviation: {stats['std']:.6f} mV")
    print(f"Variance: {stats['variance']:.6f}")
    print(f"Peak-to-Peak Amplitude: {stats['ptp']:.6f} mV")
    print(f"Dominant Frequency: {dominant_frequency:.2f} Hz")


if __name__ == "__main__":
    main()

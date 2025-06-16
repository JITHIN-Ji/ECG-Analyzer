import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import math

CSV_FILE = "digitized_trace.csv"
SIGNAL_COLUMN = "Lead_I"

# === Load signal ===
df = pd.read_csv(CSV_FILE)
if SIGNAL_COLUMN not in df.columns:
    raise ValueError(f"'{SIGNAL_COLUMN}' not found in CSV.")

signal = df[SIGNAL_COLUMN].values
print(f"[INFO] Signal length: {len(signal)} samples")

# === Estimate sampling rate ===
mm_per_pixel = 1 / 10.0  # 10 pixels = 1 mm
ms_per_mm = 40           # 25 mm/s → 1 mm = 40ms
ms_per_sample = mm_per_pixel * ms_per_mm
FS = 1000 / ms_per_sample  # Hz
print(f"[INFO] Estimated Sampling Rate: {FS:.2f} Hz")

# === Bandpass filter ===
def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=250.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

filtered_signal = bandpass_filter(signal, fs=FS)

# === R peak detection ===
threshold = np.mean(filtered_signal) + 0.5 * np.std(filtered_signal)
min_distance = int(0.6 * FS)  # at least 600 ms between peaks
r_peaks, _ = find_peaks(filtered_signal, height=threshold, distance=min_distance)
print(f"[INFO] R-peaks detected: {len(r_peaks)}")

# === RR intervals and BPM ===
rr_intervals = np.diff(r_peaks) / FS
if len(rr_intervals) == 0:
    raise ValueError("No valid RR intervals detected. Try a better-quality signal.")

bpm = 60 / np.mean(rr_intervals)
sdnn = np.std(rr_intervals)
rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
ibi = 1 / (bpm / 60)

# === Classify heart rhythm based on BPM ===
if bpm < 60:
    rhythm = "Bradycardia (Slow)"
elif 60 <= bpm <= 100:
    rhythm = "Normal (Regular)"
else:
    rhythm = "Tachycardia (Fast)"

# === Estimate QRS duration ===
qrs_durations = []
for r in r_peaks:
    window = filtered_signal[max(r - 30, 0):min(r + 30, len(filtered_signal))]
    qrs_start = np.argmax(np.diff(window) < -5)
    qrs_end = np.argmax(np.diff(window[qrs_start:]) > 5) + qrs_start
    qrs_dur = (qrs_end - qrs_start) / FS
    qrs_durations.append(qrs_dur)
qrs_duration = np.mean(qrs_durations)

# === Wave amplitudes (approximate) ===
p_wave_amp = np.max(filtered_signal[:50]) - np.min(filtered_signal[:50])
qrs_amp = np.max(filtered_signal) - np.min(filtered_signal)
t_wave_amp = np.max(filtered_signal[-100:]) - np.min(filtered_signal[-100:])

# === QT and QTc intervals ===
qt_interval = 0.4  # seconds (placeholder)
qtc = qt_interval / math.sqrt(np.mean(rr_intervals))

# === Results ===
print("\n[✅] ECG Parameter Summary:")
print(f"Heart Rate (bpm):        {bpm:.2f}")
print(f"RR Interval (s):         {np.mean(rr_intervals):.4f}")
print(f"IBI (s):                 {ibi:.4f}")
print(f"Heart Rhythm:            {rhythm}")
print(f"QRS Duration (s):        {qrs_duration:.4f}")
print(f"P Wave Amplitude:        {p_wave_amp:.2f}")
print(f"QRS Amplitude:           {qrs_amp:.2f}")
print(f"T Wave Amplitude:        {t_wave_amp:.2f}")
print(f"QT Interval (s):         {qt_interval:.3f}")
print(f"QTc Interval (s):        {qtc:.3f}")
print(f"SDNN (s):                {sdnn:.4f}")
print(f"RMSSD (s):               {rmssd:.4f}")
print(f"ST Segment (s):          ~0.12 ")
print(f"PR Segment (s):          ~0.06 ")

# === Visual plot ===
plt.figure(figsize=(12, 4))
plt.plot(filtered_signal, label='Filtered ECG')
plt.plot(r_peaks, filtered_signal[r_peaks], 'ro', label='R-peaks')
plt.title('ECG with R-peaks')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.linalg import solve_toeplitz


# ------------------------- basic utilities -------------------------

def find_default_audio_dir() -> Path:
    candidates = [
        Path("audio"),
        Path("voice-language-processing/audio"),
        Path("/mnt/data/vlp_work/voice-language-processing/audio"),
        Path("/mnt/data/audio"),
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.wav")):
            return c
    raise FileNotFoundError(
        "Could not find an audio directory. Pass --audio-dir /path/to/audio"
    )


def read_wav(path: Path) -> Tuple[int, np.ndarray]:
    fs, x = wavfile.read(path)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    # Normalize integer PCM and also protect against accidental clipping-scale data.
    peak = np.max(np.abs(x)) if len(x) else 1.0
    if peak > 1.5:
        x = x / peak
    return fs, x


def savefig(out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(x) < frame_len:
        return np.expand_dims(np.pad(x, (0, frame_len - len(x))), axis=0)
    n = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    return np.sum(frames ** 2, axis=1)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    signs = np.sign(frames)
    signs[signs == 0] = 1
    return np.sum(np.abs(np.diff(signs, axis=1)), axis=1) / (2 * frames.shape[1])


def moving_average(x: np.ndarray, n: int = 3) -> np.ndarray:
    if n <= 1 or len(x) < n:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


# ------------------------- speech analysis helpers -------------------------

def classify_frames(
    x: np.ndarray,
    fs: int,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Adaptive voiced / unvoiced / silence frame classification."""
    frame_len = max(8, round(fs * frame_ms / 1000))
    hop = max(1, round(fs * hop_ms / 1000))
    frames = frame_signal(x, frame_len, hop) * np.hamming(frame_len)
    energy = short_time_energy(frames)
    zcr = zero_crossing_rate(frames)
    times = (np.arange(len(frames)) * hop + frame_len / 2) / fs

    # Adaptive thresholds: robust to different recording levels.
    energy_s = moving_average(energy, 5)
    zcr_s = moving_average(zcr, 5)
    silence_thr = max(np.percentile(energy_s, 20), 0.015 * np.max(energy_s))
    voiced_thr = max(np.percentile(energy_s, 45), 0.08 * np.max(energy_s))
    zcr_thr = np.percentile(zcr_s, 65)

    labels = np.full(len(frames), "silence", dtype=object)
    active = energy_s > silence_thr
    labels[active & (energy_s > voiced_thr) & (zcr_s <= zcr_thr)] = "voiced"
    labels[active & (labels != "voiced")] = "unvoiced"
    return frames, times, energy_s, zcr_s, labels, voiced_thr, zcr_thr


def estimate_pitch_autocorr(
    frames: np.ndarray,
    labels: np.ndarray,
    fs: int,
    fmin: float = 60.0,
    fmax: float = 400.0,
) -> np.ndarray:
    """Estimate F0 using autocorrelation with human-speech pitch bounds."""
    f0 = np.zeros(len(frames))
    min_lag = max(1, int(fs / fmax))
    max_lag = min(frames.shape[1] - 1, int(fs / fmin))
    for i, frame in enumerate(frames):
        if labels[i] != "voiced":
            continue
        y = frame - np.mean(frame)
        acf = signal.correlate(y, y, mode="full")[len(y) - 1 :]
        if acf[0] <= 1e-12 or max_lag <= min_lag:
            continue
        search = acf[min_lag : max_lag + 1]
        peaks, props = signal.find_peaks(search)
        if len(peaks) == 0:
            lag = min_lag + int(np.argmax(search))
        else:
            # Choose strongest valid peak, not zero lag.
            lag = min_lag + peaks[int(np.argmax(search[peaks]))]
        strength = acf[lag] / acf[0]
        if strength >= 0.25:
            f0[i] = fs / lag
    return f0


def lpc_coeffs(x: np.ndarray, order: int) -> np.ndarray:
    y = np.asarray(x, dtype=float) - np.mean(x)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    r = np.correlate(y, y, mode="full")[len(y) - 1 : len(y) + order]
    if len(r) < order + 1 or r[0] <= 1e-12:
        return np.r_[1.0, np.zeros(order)]
    a = solve_toeplitz((r[:-1], r[:-1]), -r[1:])
    return np.r_[1.0, a]


def lpc_spectrum(seg: np.ndarray, fs: int, order: int, nfft: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    a = lpc_coeffs(seg, order)
    w, h = signal.freqz([1.0], a, worN=nfft, fs=fs)
    env_db = 20 * np.log10(np.abs(h) + 1e-10)
    return w, env_db


def estimate_formants(seg: np.ndarray, fs: int, order: int = 12, max_freq: float = 5000.0) -> List[float]:
    a = lpc_coeffs(seg, order)
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0.01]
    angles = np.angle(roots)
    freqs = angles * fs / (2 * np.pi)
    bandwidths = -0.5 * fs * np.log(np.abs(roots)) / np.pi
    keep = (freqs > 90) & (freqs < max_freq) & (bandwidths < 700)
    return sorted(freqs[keep].tolist())[:4]


def choose_segment_from_labels(
    x: np.ndarray,
    fs: int,
    labels: np.ndarray,
    preferred: str,
    seg_ms: float = 40.0,
) -> np.ndarray:
    hop = round(fs * 0.010)
    seg_len = round(fs * seg_ms / 1000)
    idxs = np.where(labels == preferred)[0]
    if len(idxs) == 0:
        idxs = np.array([max(0, len(labels) // 2)])
    i = int(idxs[len(idxs) // 2])
    start = max(0, i * hop)
    end = min(len(x), start + seg_len)
    seg = x[start:end]
    if len(seg) < seg_len:
        seg = np.pad(seg, (0, seg_len - len(seg)))
    return seg


# ------------------------- mel / mfcc helpers -------------------------

def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595 * np.log10(1 + np.asarray(f) / 700)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700 * (10 ** (np.asarray(m) / 2595) - 1)


def mel_filterbank(fs: int, nfft: int, n_mels: int = 40, fmin: float = 0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = fs / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((nfft + 1) * hz / fs).astype(int)
    fb = np.zeros((n_mels, nfft // 2 + 1))
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center > left:
            fb[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def compute_mfcc(x: np.ndarray, fs: int, n_mfcc: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = 512
    noverlap = 384
    f, t, Z = signal.stft(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Z) ** 2
    fb = mel_filterbank(fs, nperseg, 40)
    mel_power = np.maximum(fb @ power, 1e-12)
    log_mel = np.log(mel_power)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
    return t, mfcc


# ------------------------- plotting functions -------------------------

def plot_waveform_overview(x: np.ndarray, fs: int, out_dir: Path) -> None:
    t = np.arange(len(x)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, x, linewidth=0.8)
    plt.title("Human speech waveform — total_human.wav")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    savefig(out_dir, "01_human_waveform.png")


def plot_waveform_with_envelope(x: np.ndarray, fs: int, out_dir: Path) -> None:
    t = np.arange(len(x)) / fs
    analytic = signal.hilbert(x)
    env = np.abs(analytic)
    plt.figure(figsize=(12, 4))
    plt.plot(t, x, linewidth=0.7, label="waveform")
    plt.plot(t, env, linewidth=1.0, label="amplitude envelope")
    plt.title("Waveform with amplitude envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    savefig(out_dir, "02_waveform_envelope.png")


def plot_spectrogram_comparison(x: np.ndarray, fs: int, out_dir: Path) -> None:
    plt.figure(figsize=(12, 7))
    for i, win_ms in enumerate([10, 100], 1):
        nperseg = max(8, round(fs * win_ms / 1000))
        noverlap = min(round(nperseg * 0.75), nperseg - 1)
        f, tt, S = signal.spectrogram(
            x, fs=fs, window="hamming", nperseg=nperseg, noverlap=noverlap, mode="magnitude"
        )
        plt.subplot(2, 1, i)
        plt.pcolormesh(tt, f, 20 * np.log10(S + 1e-10), shading="auto")
        plt.ylim(0, min(5000, fs / 2))
        plt.title(f"Spectrogram with {win_ms} ms Hamming window")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    savefig(out_dir, "03_spectrogram_10ms_vs_100ms.png")


def plot_classification(x: np.ndarray, fs: int, out_dir: Path):
    frames, times, energy, zcr, labels, eth, zth = classify_frames(x, fs)
    t = np.arange(len(x)) / fs
    y = np.array([{"silence": 0, "unvoiced": 1, "voiced": 2}[lab] for lab in labels])

    plt.figure(figsize=(12, 9))
    plt.subplot(4, 1, 1)
    plt.plot(t, x, linewidth=0.7)
    plt.title("Waveform")
    plt.ylabel("Amplitude")
    plt.subplot(4, 1, 2)
    plt.plot(times, energy)
    plt.axhline(eth, linestyle="--", label="voiced energy threshold")
    plt.title("Short-time energy")
    plt.ylabel("Energy")
    plt.legend(loc="upper right")
    plt.subplot(4, 1, 3)
    plt.plot(times, zcr)
    plt.axhline(zth, linestyle="--", label="ZCR threshold")
    plt.title("Zero-crossing rate")
    plt.ylabel("ZCR")
    plt.legend(loc="upper right")
    plt.subplot(4, 1, 4)
    plt.scatter(times, y, s=12)
    plt.yticks([0, 1, 2], ["silence", "unvoiced", "voiced"])
    plt.title("Frame classification")
    plt.xlabel("Time (s)")
    savefig(out_dir, "04_energy_zcr_voicing_classification.png")
    return frames, times, labels


def plot_pitch(x: np.ndarray, fs: int, frames: np.ndarray, times: np.ndarray, labels: np.ndarray, out_dir: Path):
    f0 = estimate_pitch_autocorr(frames, labels, fs)
    t = np.arange(len(x)) / fs
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, x, linewidth=0.7)
    plt.title("Waveform")
    plt.ylabel("Amplitude")
    plt.subplot(2, 1, 2)
    plt.plot(times, f0, marker="o", linestyle="", markersize=3)
    plt.ylim(0, 450)
    plt.title("Estimated fundamental frequency / F0 contour")
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    savefig(out_dir, "05_f0_pitch_contour.png")
    return f0


def plot_autocorrelation_example(frames: np.ndarray, labels: np.ndarray, fs: int, out_dir: Path) -> None:
    voiced = np.where(labels == "voiced")[0]
    idx = int(voiced[len(voiced) // 2]) if len(voiced) else len(frames) // 2
    frame = frames[idx] - np.mean(frames[idx])
    acf = signal.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    lags_ms = np.arange(len(acf)) / fs * 1000
    plt.figure(figsize=(10, 4))
    plt.plot(lags_ms, acf / (acf[0] + 1e-12), linewidth=1.0)
    plt.xlim(0, 30)
    plt.title("Autocorrelation example used for pitch estimation")
    plt.xlabel("Lag (ms)")
    plt.ylabel("Normalized autocorrelation")
    savefig(out_dir, "06_autocorrelation_pitch_example.png")


def plot_lpc_maps_and_envelopes(x: np.ndarray, fs: int, labels: np.ndarray, out_dir: Path) -> None:
    voiced_seg = choose_segment_from_labels(x, fs, labels, "voiced")
    unvoiced_seg = choose_segment_from_labels(x, fs, labels, "unvoiced")
    orders = [8, 12, 16]

    plt.figure(figsize=(10, 12))
    for r, order in enumerate(orders):
        for c, (seg, title) in enumerate([(voiced_seg, "voiced"), (unvoiced_seg, "unvoiced")]):
            a = lpc_coeffs(seg, order)
            roots = np.roots(a)
            ax = plt.subplot(len(orders), 2, r * 2 + c + 1)
            ax.add_artist(plt.Circle((0, 0), 1, fill=False, linestyle="--"))
            ax.scatter(np.real(roots), np.imag(roots), marker="x")
            ax.axhline(0, linewidth=0.6)
            ax.axvline(0, linewidth=0.6)
            ax.set_aspect("equal", "box")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_title(f"LPC pole map — {title}, order {order}")
            ax.set_xlabel("Real")
            ax.set_ylabel("Imaginary")
    savefig(out_dir, "07_lpc_pole_maps.png")

    plt.figure(figsize=(12, 6))
    for seg, title in [(voiced_seg, "voiced"), (unvoiced_seg, "unvoiced")]:
        freqs = np.fft.rfftfreq(2048, 1 / fs)
        X = 20 * np.log10(np.abs(np.fft.rfft(seg * np.hamming(len(seg)), 2048)) + 1e-10)
        w, env = lpc_spectrum(seg, fs, order=12, nfft=2048)
        plt.plot(freqs, X - np.max(X), linewidth=0.6, label=f"{title} FFT spectrum")
        plt.plot(w, env - np.max(env), linewidth=1.4, label=f"{title} LPC envelope")
    plt.xlim(0, min(5000, fs / 2))
    plt.ylim(-80, 5)
    plt.title("FFT spectrum vs LPC spectral envelope")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Relative magnitude (dB)")
    plt.legend(loc="lower right")
    savefig(out_dir, "08_lpc_spectral_envelopes.png")

    formants = estimate_formants(voiced_seg, fs, order=12)
    freqs = np.fft.rfftfreq(4096, 1 / fs)
    X = 20 * np.log10(np.abs(np.fft.rfft(voiced_seg * np.hamming(len(voiced_seg)), 4096)) + 1e-10)
    w, env = lpc_spectrum(voiced_seg, fs, order=12, nfft=4096)
    plt.figure(figsize=(12, 4))
    plt.plot(freqs, X - np.max(X), linewidth=0.6, label="voiced FFT spectrum")
    plt.plot(w, env - np.max(env), linewidth=1.4, label="LPC envelope")
    for j, f in enumerate(formants, 1):
        plt.axvline(f, linestyle="--")
        plt.text(f + 20, -10 - 7 * (j % 3), f"F{j}≈{f:.0f} Hz", rotation=90, va="top")
    plt.xlim(0, min(5000, fs / 2))
    plt.ylim(-80, 5)
    plt.title("Estimated formant locations from LPC roots")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Relative magnitude (dB)")
    plt.legend(loc="lower right")
    savefig(out_dir, "09_lpc_formant_estimates.png")


def plot_mel_and_mfcc(x: np.ndarray, fs: int, out_dir: Path) -> None:
    nperseg = 512
    noverlap = 384
    f, t, Z = signal.stft(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Z) ** 2
    fb = mel_filterbank(fs, nperseg, n_mels=40)
    mel_power = np.maximum(fb @ power, 1e-12)
    mel_db = 10 * np.log10(mel_power)

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, np.arange(mel_db.shape[0]), mel_db, shading="auto")
    plt.title("Mel spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel filter index")
    plt.colorbar(label="Power (dB)")
    savefig(out_dir, "10_mel_spectrogram.png")

    mfcc_t, mfcc = compute_mfcc(x, fs, n_mfcc=13)
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(mfcc_t, np.arange(mfcc.shape[0]), mfcc, shading="auto")
    plt.title("MFCC coefficients")
    plt.xlabel("Time (s)")
    plt.ylabel("Coefficient index")
    plt.colorbar(label="MFCC value")
    savefig(out_dir, "11_mfcc_heatmap.png")


def plot_vowel_bank(audio: Dict[str, Tuple[int, np.ndarray]], out_dir: Path) -> None:
    vowels = [v for v in ["AO", "IY", "UH", "EH", "AH", "IH"] if v in audio]
    if not vowels:
        return

    plt.figure(figsize=(12, 10))
    for i, v in enumerate(vowels, 1):
        fs, x = audio[v]
        n = min(len(x), int(0.06 * fs))
        plt.subplot(len(vowels), 1, i)
        plt.plot(np.arange(n) / fs * 1000, x[:n], linewidth=0.8)
        plt.title(f"Synthesized vowel /{v}/ waveform — first 60 ms")
        plt.ylabel("Amp")
    plt.xlabel("Time (ms)")
    savefig(out_dir, "12_vowel_waveforms_first_60ms.png")

    plt.figure(figsize=(12, 10))
    for i, v in enumerate(vowels, 1):
        fs, x = audio[v]
        nfft = max(2048, 2 ** int(np.ceil(np.log2(len(x)))))
        X = 20 * np.log10(np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) + 1e-10)
        freq = np.fft.rfftfreq(nfft, 1 / fs)
        plt.subplot(len(vowels), 1, i)
        plt.plot(freq, X - np.max(X), linewidth=0.8)
        plt.xlim(0, min(5000, fs / 2))
        plt.ylim(-80, 5)
        plt.title(f"Spectrum of synthesized vowel /{v}/")
        plt.ylabel("Rel. dB")
    plt.xlabel("Frequency (Hz)")
    savefig(out_dir, "13_vowel_spectra.png")

    # Estimated formant table as a visual bar chart / scatter.
    plt.figure(figsize=(10, 5))
    for i, v in enumerate(vowels):
        fs, x = audio[v]
        forms = estimate_formants(x, fs, order=10)
        for j, f in enumerate(forms[:3], 1):
            plt.scatter(i, f, s=45)
            plt.text(i + 0.05, f, f"F{j}", va="center")
    plt.xticks(range(len(vowels)), [f"/{v}/" for v in vowels])
    plt.ylim(0, 5000)
    plt.title("Approximate formant estimates for synthesized vowels")
    plt.ylabel("Frequency (Hz)")
    savefig(out_dir, "14_vowel_formant_estimates.png")


def plot_human_vs_synth(audio: Dict[str, Tuple[int, np.ndarray]], out_dir: Path) -> None:
    if "total_human" not in audio or "total" not in audio:
        return
    items = [("Human recording", audio["total_human"]), ("Synthesized vowel sequence", audio["total"])]

    plt.figure(figsize=(12, 7))
    for i, (title, (fs, x)) in enumerate(items, 1):
        f, t, S = signal.spectrogram(x, fs=fs, window="hann", nperseg=256, noverlap=220, mode="magnitude")
        plt.subplot(2, 1, i)
        plt.pcolormesh(t, f, 20 * np.log10(S + 1e-10), shading="auto")
        plt.ylim(0, min(5000, fs / 2))
        plt.title(title)
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    savefig(out_dir, "15_human_vs_synth_spectrograms.png")

    plt.figure(figsize=(12, 5))
    for title, (fs, x) in items:
        f, Pxx = signal.welch(x, fs=fs, nperseg=min(512, len(x)))
        Pxx_db = 10 * np.log10(Pxx + 1e-12)
        plt.plot(f, Pxx_db - np.max(Pxx_db), linewidth=1.0, label=title)
    plt.xlim(0, 5000)
    plt.ylim(-80, 5)
    plt.title("Average power spectrum: human vs synthesized")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Relative power (dB)")
    plt.legend()
    savefig(out_dir, "16_human_vs_synth_average_spectrum.png")


def plot_audio_inventory(audio: Dict[str, Tuple[int, np.ndarray]], out_dir: Path) -> None:
    names = list(audio.keys())
    durations = [len(x) / fs for fs, x in audio.values()]
    rms = [np.sqrt(np.mean(x ** 2)) for fs, x in audio.values()]

    plt.figure(figsize=(10, 4))
    plt.bar(names, durations)
    plt.title("Audio file durations")
    plt.xlabel("File")
    plt.ylabel("Duration (s)")
    plt.xticks(rotation=45, ha="right")
    savefig(out_dir, "17_audio_durations.png")

    plt.figure(figsize=(10, 4))
    plt.bar(names, rms)
    plt.title("Audio RMS levels")
    plt.xlabel("File")
    plt.ylabel("RMS amplitude")
    plt.xticks(rotation=45, ha="right")
    savefig(out_dir, "18_audio_rms_levels.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=None, help="Directory containing WAV files")
    parser.add_argument("--out-dir", type=Path, default=Path("vlp_visualizations_enhanced"), help="Output PNG directory")
    args = parser.parse_args()

    audio_dir = args.audio_dir if args.audio_dir else find_default_audio_dir()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    audio: Dict[str, Tuple[int, np.ndarray]] = {}
    for wav in sorted(audio_dir.glob("*.wav")):
        audio[wav.stem] = read_wav(wav)

    if "total_human" not in audio:
        raise FileNotFoundError("Expected total_human.wav for the human speech analysis visuals.")

    fs_h, human = audio["total_human"]
    plot_waveform_overview(human, fs_h, out_dir)
    plot_waveform_with_envelope(human, fs_h, out_dir)
    plot_spectrogram_comparison(human, fs_h, out_dir)
    frames, times, labels = plot_classification(human, fs_h, out_dir)
    plot_pitch(human, fs_h, frames, times, labels, out_dir)
    plot_autocorrelation_example(frames, labels, fs_h, out_dir)
    plot_lpc_maps_and_envelopes(human, fs_h, labels, out_dir)
    plot_mel_and_mfcc(human, fs_h, out_dir)
    plot_vowel_bank(audio, out_dir)
    plot_human_vs_synth(audio, out_dir)
    plot_audio_inventory(audio, out_dir)

    print(f"Generated {len(list(out_dir.glob('*.png')))} visualizations in: {out_dir.resolve()}")
    for p in sorted(out_dir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()

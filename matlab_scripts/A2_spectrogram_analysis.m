% =========================================================================
% A2 — Spectrogram Analysis
%
% Computes and plots the short-time Fourier transform (STFT) spectrogram
% of a recorded speech signal using two different Hamming window lengths:
%   - 10 ms  : high temporal resolution, lower frequency resolution
%   - 100 ms : high frequency resolution, lower temporal resolution
%
% The comparison illustrates the time-frequency trade-off in spectral
% analysis of non-stationary signals such as speech.
%
% Input
%   myRecording.wav : mono WAV file produced by A1_record_audio.m
% =========================================================================

% --- Load audio ---
filename = 'myRecording.wav';
[x, fs] = audioread(filename);

% --- Window and overlap parameters ---
window_length_short = round(fs * 0.01);   % 10 ms window
window_length_long  = round(fs * 0.10);   % 100 ms window
overlap             = round(fs * 0.005);  % 5 ms hop size

% --- Compute spectrograms ---
[S_short, f_short, t_short] = spectrogram(x, hamming(window_length_short), overlap, [], fs);
[S_long,  f_long,  t_long ] = spectrogram(x, hamming(window_length_long),  overlap, [], fs);

% --- Plot ---
figure('Name', 'Spectrogram Comparison', 'NumberTitle', 'off');

subplot(2, 1, 1);
surf(t_short, f_short, 10*log10(abs(S_short)), 'EdgeColor', 'none');
view(2);
axis tight;
title('Spectrogram — Hamming Window 10 ms');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
colormap jet;

subplot(2, 1, 2);
surf(t_long, f_long, 10*log10(abs(S_long)), 'EdgeColor', 'none');
view(2);
axis tight;
title('Spectrogram — Hamming Window 100 ms');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
colormap jet;

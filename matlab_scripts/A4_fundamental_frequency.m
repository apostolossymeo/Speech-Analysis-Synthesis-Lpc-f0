% =========================================================================
% A4 — Fundamental Frequency (Pitch) Estimation
%
% Estimates the fundamental frequency (F0 / pitch) of each voiced frame
% in a speech signal using autocorrelation-based pitch detection.
%
% Frames are first classified as voiced or non-voiced using the same
% energy/ZCR decision rule as A3. For voiced frames the autocorrelation
% function is computed, thresholded, and the lag of the dominant peak is
% used to derive F0.  Non-voiced frames receive F0 = 0 Hz.
%
% Outputs a two-panel figure:
%   Top    : original waveform
%   Bottom : F0 contour over time (voiced frames only)
%
% Input
%   myRecording.wav : mono WAV file produced by A1_record_audio.m
% =========================================================================

% --- Load audio ---
filename = 'myRecording.wav';
[x, fs] = audioread(filename);

% --- Framing parameters ---
windowSize      = round(fs * 0.02);   % 20 ms frame
overlap         = round(fs * 0.01);   % 10 ms hop
energyThreshold = 0.1;
zcrThreshold    = 0.1;

% --- Frame and classify ---
numSamples = length(x);
numWindows = floor((numSamples - windowSize) / overlap) + 1;
segments   = zeros(numWindows, windowSize);
labels     = cell(numWindows, 1);

for i = 1:numWindows
    startIdx = (i - 1) * overlap + 1;
    endIdx   = startIdx + windowSize - 1;
    frame    = x(startIdx:endIdx);

    energy = computeEnergy(frame);
    zcr    = computeZeroCrossingRate(frame);

    if energy < energyThreshold
        labels{i} = ternary(zcr < zcrThreshold, 'silence', 'unvoiced');
    else
        labels{i} = 'voiced';
    end

    segments(i, :) = frame;
end

% --- Pitch estimation ---
fundamental_freqs = estimatePitch(segments, fs, labels);

% --- Visualisation ---
t_signal = (0:length(x) - 1) / fs;
t_frames = (windowSize/2 : overlap : windowSize/2 + overlap*(numWindows-1)) / fs;

figure('Name', 'Pitch (F0) Contour', 'NumberTitle', 'off');

subplot(2, 1, 1);
plot(t_signal, x);
title('Waveform');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2, 1, 2);
% Only scatter voiced frames
voiced_mask = fundamental_freqs > 0;
scatter(t_frames(voiced_mask), fundamental_freqs(voiced_mask), 20, 'filled');
title('Estimated Fundamental Frequency (F0)');
xlabel('Time (s)');
ylabel('F0 (Hz)');
ylim([0, fs / 2]);

% =========================================================================
% Local functions
% =========================================================================

function energy = computeEnergy(signal)
    energy = sum(signal .^ 2);
end

function zcr = computeZeroCrossingRate(signal)
    zcr = sum(abs(diff(sign(signal)))) / (2 * length(signal));
end

function result = ternary(condition, a, b)
    if condition; result = a; else; result = b; end
end

function f0_contour = estimatePitch(segments, fs, labels)
    % Autocorrelation-based pitch estimator.
    %
    % For each voiced frame, computes the normalized autocorrelation,
    % zeroes out values below a relative threshold, and reads the lag of
    % the dominant peak after the zeroth lag. F0 = fs / peak_lag.

    numFrames  = size(segments, 1);
    windowSize = size(segments, 2);
    threshold  = 0.3;             % Relative autocorrelation threshold
    f0_contour = zeros(numFrames, 1);

    for i = 1:numFrames
        if ~strcmp(labels{i}, 'voiced')
            continue;  % Leave unvoiced / silence frames as 0
        end

        frame    = segments(i, :);
        acf      = xcorr(frame);

        % Zero-out lags below the relative threshold
        acf(acf < threshold * max(acf)) = 0;

        % Find the dominant peak after the zero-lag centre
        centre = length(acf) / 2;
        [~, rel_idx] = max(acf(windowSize + 1 : end));
        lag = rel_idx + windowSize - centre;

        if lag > 0
            f0_contour(i) = fs / lag;
        end
    end
end

% =========================================================================
% A3 — Voiced / Unvoiced / Silence Segmentation
%
% Segments a speech signal into three acoustic classes using a
% frame-by-frame decision rule based on short-time energy (STE) and
% zero-crossing rate (ZCR):
%
%   energy < energyThreshold  AND  zcr < zcrThreshold  →  silence
%   energy < energyThreshold  AND  zcr ≥ zcrThreshold  →  unvoiced
%   energy ≥ energyThreshold                            →  voiced
%
% Results are visualised across four subplot panels:
%   1. Waveform
%   2. Zero-crossing rate per frame
%   3. Short-time energy per frame
%   4. Colour-coded segment classification (red = voiced,
%      green = unvoiced, blue = silence)
%
% Input
%   myRecording.wav : mono WAV file produced by A1_record_audio.m
% =========================================================================

% --- Load audio ---
filename = 'myRecording.wav';
[x, fs] = audioread(filename);

% --- Framing parameters ---
windowSize      = round(fs * 0.02);  % 20 ms frame
overlap         = round(fs * 0.01);  % 10 ms hop
energyThreshold = 0.1;
zcrThreshold    = 0.1;

% --- Frame the signal and classify each frame ---
numSamples  = length(x);
numWindows  = floor((numSamples - windowSize) / overlap) + 1;
segments    = zeros(numWindows, windowSize);
decisions   = cell(numWindows, 1);
zcr_values  = zeros(numWindows, 1);
energy_values = zeros(numWindows, 1);

for i = 1:numWindows
    startIdx = (i - 1) * overlap + 1;
    endIdx   = startIdx + windowSize - 1;
    frame    = x(startIdx:endIdx);

    energy_values(i) = computeEnergy(frame);
    zcr_values(i)    = computeZeroCrossingRate(frame);

    if energy_values(i) < energyThreshold
        if zcr_values(i) < zcrThreshold
            decisions{i} = 'silence';
        else
            decisions{i} = 'unvoiced';
        end
    else
        decisions{i} = 'voiced';
    end

    segments(i, :) = frame;
end

% --- Visualisation ---
t        = (0:length(x) - 1) / fs;
t_frames = (windowSize/2 : overlap : windowSize/2 + overlap*(numWindows-1)) / fs;

figure('Name', 'Speech Segmentation', 'NumberTitle', 'off');

subplot(4, 1, 1);
plot(t, x);
title('Waveform');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(4, 1, 2);
plot(t_frames, zcr_values);
title('Zero-Crossing Rate per Frame');
xlabel('Time (s)');
ylabel('ZCR');

subplot(4, 1, 3);
plot(t_frames, energy_values);
title('Short-Time Energy per Frame');
xlabel('Time (s)');
ylabel('Energy');

subplot(4, 1, 4);
for i = 1:numWindows
    switch decisions{i}
        case 'voiced';   color = 'r';
        case 'unvoiced'; color = 'g';
        case 'silence';  color = 'b';
    end
    rectangle('Position', [t_frames(i) - overlap/fs, 0, overlap/fs, 1], ...
              'FaceColor', color, 'EdgeColor', 'none');
    hold on;
end
title('Segment Classification  (red = voiced | green = unvoiced | blue = silence)');
xlabel('Time (s)');
ylabel('Class');

% --- Play voiced segments ---
% Stack segments into a mono signal (one channel only)
mono_segments = segments(:, 1);   % take first channel
soundsc(mono_segments, fs);

% =========================================================================
% Local functions
% =========================================================================

function energy = computeEnergy(signal)
    % Short-time energy: sum of squared samples within a frame.
    energy = sum(signal .^ 2);
end

function zcr = computeZeroCrossingRate(signal)
    % Zero-crossing rate: fraction of adjacent sample pairs with opposite sign.
    zcr = sum(abs(diff(sign(signal)))) / (2 * length(signal));
end

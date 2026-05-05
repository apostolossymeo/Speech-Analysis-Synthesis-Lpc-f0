% =========================================================================
% A5 — Linear Predictive Coding (LPC) Analysis
%
% Applies LPC analysis to a 30 ms voiced segment and a 30 ms unvoiced
% segment extracted from the recorded speech signal.
%
% For each of three predictor orders (p = 8, 12, 16):
%   - LPC coefficients are estimated via the autocorrelation method.
%   - The all-pole transfer function H(z) = 1/A(z) is visualised as a
%     pole-zero map (zplane) — poles trace the formant structure.
%   - The prediction error (residual energy) is computed.
%
% Voiced segments show clustered poles near the unit circle corresponding
% to formant resonances; unvoiced segments show a flatter spectrum with
% poles distributed more uniformly.
%
% Unvoiced segments are located automatically using a simplified
% autocorrelation-based voicing detector (myRAPT).
%
% Input
%   myRecording.wav : mono WAV file produced by A1_record_audio.m
% =========================================================================

function A5_lpc_analysis()

    % --- Load audio ---
    filename = 'myRecording.wav';
    [x, fs]  = audioread(filename);

    % --- Segment parameters ---
    seg_ms     = 30;                            % Segment length in milliseconds
    seg_len    = round(seg_ms * fs / 1000);     % Segment length in samples

    % --- Select voiced segment (fixed index — adjust if needed) ---
    voiced_start   = 10000;
    voiced_segment = x(voiced_start : voiced_start + seg_len - 1);

    % --- Locate and select an unvoiced segment ---
    uv_indices = findUnvoicedIndices(x, fs);
    if isempty(uv_indices)
        error('No unvoiced segments found in the recording.');
    end
    uv_start        = uv_indices(randi(numel(uv_indices)));
    uv_end          = min(uv_start + seg_len - 1, numel(x));
    unvoiced_segment = x(uv_start : uv_end);

    % --- LPC analysis ---
    p_values          = [8, 12, 16];
    num_orders        = numel(p_values);
    lpc_voiced        = cell(num_orders, 1);
    lpc_unvoiced      = cell(num_orders, 1);
    err_voiced        = zeros(num_orders, 1);
    err_unvoiced      = zeros(num_orders, 1);

    for i = 1:num_orders
        p = p_values(i);

        lpc_voiced{i}   = lpc(voiced_segment,   p);
        lpc_unvoiced{i} = lpc(unvoiced_segment, p);

        err_voiced(i)   = sum(filter(lpc_voiced{i},   1, voiced_segment)   .^ 2);
        err_unvoiced(i) = sum(filter(lpc_unvoiced{i}, 1, unvoiced_segment) .^ 2);
    end

    % --- Display prediction error summary ---
    fprintf('\n%-6s  %-20s  %-20s\n', 'Order', 'Voiced Error', 'Unvoiced Error');
    fprintf('%s\n', repmat('-', 1, 50));
    for i = 1:num_orders
        fprintf('p=%-4d  %-20.4f  %-20.4f\n', p_values(i), err_voiced(i), err_unvoiced(i));
    end

    % --- Pole-zero plots ---
    figure('Name', 'LPC Pole-Zero Maps', 'NumberTitle', 'off');
    for i = 1:num_orders
        subplot(num_orders, 2, 2*i - 1);
        zplane(1, lpc_voiced{i});
        title(sprintf('Voiced — p = %d', p_values(i)));

        subplot(num_orders, 2, 2*i);
        zplane(1, lpc_unvoiced{i});
        title(sprintf('Unvoiced — p = %d', p_values(i)));
    end

end  % function A5_lpc_analysis

% =========================================================================
% Local functions
% =========================================================================

function uv_indices = findUnvoicedIndices(x, fs)
    % Scans the signal in 30 ms steps and marks frames whose estimated
    % F0 falls below 100 Hz (treated as unvoiced / aperiodic).

    win_len    = round(30 * fs / 1000);   % 30 ms window
    step       = round(5  * fs / 1000);   % 5 ms step
    f0_thresh  = 100;                     % Hz
    uv_indices = [];

    for i = 1 : step : (numel(x) - win_len + 1)
        frame      = x(i : i + win_len - 1);
        [f0, ~]    = estimateF0(frame, fs);
        if f0 < f0_thresh
            uv_indices(end + 1) = i; %#ok<AGROW>
        end
    end
end

function [f0, vuv] = estimateF0(x, fs)
    % Simplified autocorrelation-based pitch estimator (RAPT-like).
    % Returns F0 in Hz and a voiced/unvoiced binary vector (vuv).

    acf             = xcorr(x);
    [~, peak_idx]   = max(acf);
    lag             = peak_idx - numel(x);          % Lag in samples

    if lag > 0
        f0 = fs / lag;
    else
        f0 = 0;
    end

    energy_threshold = 0.1;
    energy           = sum(x .^ 2);
    vuv              = double(energy > energy_threshold) * ones(size(x));
end

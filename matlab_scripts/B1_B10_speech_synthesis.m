% =========================================================================
% B1–B10 — Articulatory Speech Synthesis
%
% Implements the source-filter model of speech production:
%
%   S(z) = P(z) · G(z) · V(z) · R(z)
%
% where:
%   P(z) — pitch pulse train (periodic excitation)
%   G(z) — glottal pulse shaping filter
%   V(z) — vocal tract resonance filter (three-formant all-pole model)
%   R(z) — radiation load (first-difference high-pass filter)
%
% Six English vowels are synthesised using published formant frequencies
% (Peterson & Barney, 1952):
%   /AO/  /IY/  /UH/  /EH/  /AH/  /IH/
%
% All synthesised vowels are concatenated into total.wav. A human
% recording of the same sequence is saved as total_human.wav.
% Spectrograms of both are compared side by side.
%
% Outputs
%   snsound.wav      : first 0.5 s of the /AO/ base signal (B5)
%   AO.wav, IY.wav, UH.wav, EH.wav, AH.wav, IH.wav : synthesised vowels
%   total.wav        : concatenated synthesised vowels
%   total_human.wav  : recorded human utterance of the same vowel sequence
% =========================================================================

clear all; clc; close all; %#ok<CLALL>

% -------------------------------------------------------------------------
% Shared synthesis constants
% -------------------------------------------------------------------------
Fs  = 10000;          % Sampling frequency (Hz)
T   = 1 / Fs;         % Sampling period (s)
Np  = 80;             % Samples per pitch period  (8 ms at 10 kHz)
NNN = 5000;           % Number of output samples per vowel

% =========================================================================
% B1 — Pitch Pulse Train  P(z)
%
% A periodic impulse train with exponentially decaying amplitude:
%   p[n] = 0.9999^k  at  n = k·Np,  for k = 0, 1, 2, …
% =========================================================================
duration  = 3;                         % seconds
kys       = Fs * duration / Np;        % Total number of pitch periods
k_vec     = linspace(0, kys - 1, kys);
nlen      = Np * kys;
n_vec     = linspace(1, nlen, nlen);

pn = zeros(1, nlen);
for nn = n_vec
    for kk = k_vec
        if nn == kk * Np
            pn(nn) = 0.9999 ^ kk;
        end
    end
end

figure('Name', 'B1 — Pitch Pulse p[n]', 'NumberTitle', 'off');
stem(n_vec / Fs, pn);
title('Pitch Pulse Train p[n]');
xlabel('Time (s)'); ylabel('Amplitude');

% Spectrum of p[n]
Y    = fft(pn);
freq = (0:numel(pn)-1) * (Fs / numel(pn));
figure('Name', 'B1 — Spectrum of p[n]', 'NumberTitle', 'off');
plot(freq, abs(Y));
title('Spectrum of Pitch Pulse p[n]');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 5000]); grid on;

% Transfer function P(z) and pole-zero map
num_Pz = [1, zeros(1, Np)];
den_Pz = [1, zeros(1, Np-1), -0.9999];
Pz     = tf(num_Pz, den_Pz, T);
figure('Name', 'B1 — Pole-Zero Map of P(z)', 'NumberTitle', 'off');
pzmap(Pz);
title('Pole-Zero Map — P(z)');

% =========================================================================
% B2 — Glottal Pulse  G(z)
%
% A two-phase model:
%   Opening phase  (0 ≤ n < N1)    : raised-cosine ramp
%   Closing phase  (N1 ≤ n < N1+N2): quarter-cosine fall
% =========================================================================
N1  = 25; N2 = 10;
gn  = zeros(1, N1 + N2 - 1);
for i = 0 : N1 + N2 - 2
    if i < N1
        gn(i+1) = 0.5 * (1 - cos(pi * (i+1) / N1));
    else
        gn(i+1) = cos(0.5 * pi * (i - (N1-1)) / N2);
    end
end
n_g = 0 : N1 + N2 - 2;

figure('Name', 'B2 — Glottal Pulse g[n]', 'NumberTitle', 'off');
stem(1000 * n_g / Fs, gn);
title('Glottal Pulse g[n]');
xlabel('Time (ms)'); ylabel('Amplitude');

Y_g    = fft(gn);
freq_g = (0:numel(gn)-1) * (Fs / numel(gn));
figure('Name', 'B2 — Spectrum of g[n]', 'NumberTitle', 'off');
plot(freq_g, log(abs(Y_g)));
title('Spectrum of Glottal Pulse g[n]  (log scale)');
xlabel('Frequency (Hz)'); ylabel('Magnitude (log)');
xlim([0 5000]); grid on;

% =========================================================================
% B3 — Vocal Tract Filter  V(z) for /AO/
%
% Three-formant all-pole filter. Each resonance is modelled as a
% complex-conjugate pole pair:
%   H_k(z) = 1 / (1 - 2·e^{-πBk·T}·cos(2πFk·T)·z^{-1}
%                   + e^{-2πBk·T}·z^{-2})
%
% Formant frequencies and bandwidths (Peterson & Barney, 1952):
%   F1 = 570 Hz,  B1 = 60 Hz
%   F2 = 840 Hz,  B2 = 100 Hz  (note: bandwidth halved in den.)
%   F3 = 2410 Hz, B3 = 120 Hz
% =========================================================================
Fk_AO = [570,  840, 2410];
sk_AO = [60,  100,  120] ./ 2;   % σk = Bk / 2

Vz_AO = makeVocalTract(Fk_AO, sk_AO, Fs, T);

figure('Name', 'B3 — Impulse Response V(z) /AO/', 'NumberTitle', 'off');
impulse(Vz_AO);
title('Vocal Tract Impulse Response — /AO/');
xlabel('Time (ms)'); ylabel('Amplitude');

[vn_AO, ~] = impulse(Vz_AO);
Y_V    = fft(vn_AO);
freq_V = (0:numel(vn_AO)-1) * (Fs / numel(vn_AO));
figure('Name', 'B3 — Spectrum V(z) /AO/', 'NumberTitle', 'off');
plot(freq_V, abs(Y_V));
title('Spectrum of Vocal Tract Filter — /AO/');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 5000]); grid on;

figure('Name', 'B3 — Pole-Zero Map V(z) /AO/', 'NumberTitle', 'off');
pzmap(Vz_AO);

% =========================================================================
% B4 — Radiation Load  R(z)
%
% Models the acoustic radiation at the lips as a first-difference filter:
%   R(z) = 1 - 0.96·z^{-1}
% =========================================================================
Rz = tf([1, -0.96], [1, 0], T);

figure('Name', 'B4 — Impulse Response R(z)', 'NumberTitle', 'off');
[r_imp, t_r] = impulse(Rz);
stem(1000 * t_r, r_imp);
title('Impulse Response — Radiation Load R(z)');
xlabel('Time (ms)'); ylabel('Amplitude');

Y_R    = fft(r_imp);
freq_R = (0:numel(r_imp)-1) * (Fs / numel(r_imp));
figure('Name', 'B4 — Spectrum R(z)', 'NumberTitle', 'off');
plot(freq_R, abs(Y_R));
title('Spectrum of Radiation Load R(z)');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 5000]);

figure('Name', 'B4 — Pole-Zero Map R(z)', 'NumberTitle', 'off');
pzmap(Rz);

% =========================================================================
% B5 — Combined System  S(z) = G(z)·V(z)·R(z)·P(z)
%
% Cascade all four components. The impulse response s[n] is the
% synthesised voiced sound.
% =========================================================================

% Build G(z) transfer function from the glottal pulse coefficients
Gz = tf(gn, 1, T);

Sz = Gz * Vz_AO * Rz * Pz;
[s, t_s] = impulse(Sz, 40);

figure('Name', 'B5 — Final Voice Signal s[n]', 'NumberTitle', 'off');
plot(t_s * 1000, s);
title('Final Voice Signal s[n]  (Cascade System)');
xlabel('Time (ms)'); ylabel('Amplitude');
xlim([0 25]);

Y_s    = fft(s);
freq_s = (0:numel(s)-1) * (Fs / numel(s));
figure('Name', 'B5 — Spectrum s[n]', 'NumberTitle', 'off');
plot(freq_s, abs(Y_s));
title('Spectrum of Final Voice Signal s[n]');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 5000]); grid on;

figure('Name', 'B5 — Pole-Zero Map S(z)', 'NumberTitle', 'off');
pzmap(Sz);

% Save and play first 0.5 s
numSamples_05 = round(0.5 * Fs);
sound(s(1:numSamples_05), Fs);
audiowrite('snsound.wav', s(1:numSamples_05), Fs);

% =========================================================================
% B6–B10 — Vowel Synthesis using Pole-Filter (IIR) Approach
%
% Each vowel is synthesised by filtering the glottal pulse through the
% vocal tract IIR filter directly in the time domain (impz), avoiding
% the need for transfer-function objects.
%
% Formant data (Peterson & Barney, 1952):
%   Vowel  F1    F2    F3
%   /AO/   570   840  2410
%   /IY/   270  2290  3010
%   /UH/   440  1020  2240
%   /EH/   530  1840  2480
%   /AH/   520  1190  2390
%   /IH/   390  1990  2550
% =========================================================================

% Shared excitation: glottal pulse convolved with pitch-period denominator
pp     = zeros(1, Np + 1);
pp(1)  = 1;
pp(Np+1) = -0.9999;   % Leaky integrator denominator

% Formant table: each row = [F1, F2, F3, s1, s2, s3]
% All bandwidths σk = 30 Hz (= Bk/2)
sigma  = 30;
vowels = struct( ...
    'name', {'AO',  'IY',  'UH',  'EH',  'AH',  'IH'}, ...
    'F',    {[570, 840, 2410], [270, 2290, 3010], ...
             [440, 1020, 2240], [530, 1840, 2480], ...
             [520, 1190, 2390], [390, 1990, 2550]} );

synth = cell(numel(vowels), 1);   % Store all synthesised waveforms

for v = 1:numel(vowels)
    Fk = vowels(v).F;

    % Build vocal tract denominator (convolution of three resonators)
    V1 = resonatorCoeff(Fk(1), sigma, T);
    V2 = resonatorCoeff(Fk(2), sigma, T);
    V3 = resonatorCoeff(Fk(3), sigma, T);
    Vden = conv(conv(V1, V2), V3);

    % Full denominator: vocal tract × pitch period filter
    den_full = conv(Vden, pp);

    % Synthesise via impulse response
    a  = impz(gn, den_full, NNN + 1);
    aa = 4000 * (a(1:NNN) - 0.96 * a(2:NNN+1));   % Apply radiation R(z)

    synth{v} = aa;

    % Waveform plot
    figure('Name', sprintf('Waveform — /%s/', vowels(v).name), 'NumberTitle', 'off');
    plot(aa(1:500));
    title(sprintf('Synthesised Waveform — /%s/', vowels(v).name));
    xlabel('Sample'); ylabel('Amplitude');

    % Spectrum plot
    Y_aa    = fft(aa);
    freq_aa = (0:NNN-1) * (Fs / NNN);
    figure('Name', sprintf('Spectrum — /%s/', vowels(v).name), 'NumberTitle', 'off');
    plot(freq_aa, abs(Y_aa));
    title(sprintf('Spectrum — /%s/', vowels(v).name));
    xlabel('Frequency (Hz)'); ylabel('Magnitude');
    xlim([0 5000]); grid on;

    % Pole-zero map
    figure('Name', sprintf('Pole-Zero — /%s/', vowels(v).name), 'NumberTitle', 'off');
    zplane(1, Vden);
    title(sprintf('Pole-Zero Map V(z) — /%s/', vowels(v).name));

    % Save WAV
    outfile = [vowels(v).name '.wav'];
    audiowrite(outfile, aa, Fs);
    fprintf('Saved %s\n', outfile);
end

% =========================================================================
% B10 — Concatenate and Compare Spectrograms
%
% Concatenates all six synthesised vowels, records the same sequence via
% microphone, and displays a side-by-side spectrogram comparison.
% =========================================================================

% Concatenate synthesised vowels
total = vertcat(synth{:});
audiowrite('total.wav', total, Fs);

% Display spectrogram of synthesised signal
figure('Name', 'Spectrogram — total.wav', 'NumberTitle', 'off');
spectrogram(total, 256, [], [], Fs, 'yaxis');
title('Spectrogram of Synthesised Vowel Sequence');
colorbar;
xlabel('Time (s)'); ylabel('Frequency (Hz)');

% --- Record human utterance ---
fprintf('\nPrepare to speak the vowel sequence /AO IY UH EH AH IH/ (3 seconds).\n');
input('Press Enter to begin recording...', 's');

nBits      = 16;
nChannels  = 1;
recObj     = audiorecorder(Fs, nBits, nChannels);
disp('Recording — speak now.');
recordblocking(recObj, 3);
disp('Recording complete.');

total_human = getaudiodata(recObj);
audiowrite('total_human.wav', total_human, Fs);

% Side-by-side spectrogram comparison
figure('Name', 'Spectrogram Comparison', 'NumberTitle', 'off');

subplot(1, 2, 1);
spectrogram(total,       256, 250, 256, Fs,           'yaxis');
title('Synthesised Vowels — total.wav');

subplot(1, 2, 2);
spectrogram(total_human, 256, 250, 256, Fs,           'yaxis');
title('Human Recording — total\_human.wav');

% =========================================================================
% Local helper functions
% =========================================================================

function Vz = makeVocalTract(Fk, sk, Fs, T)
    % Returns the vocal tract transfer function as a MATLAB tf object.
    % Fk : formant frequencies (Hz), sk : σk = bandwidth/2 (Hz)
    Vz = 1;
    for i = 1:numel(Fk)
        num_i = 1;
        den_i = [1, ...
                 -2 * exp(-2*pi*sk(i)*T) * cos(2*pi*Fk(i)*T), ...
                  exp(-4*pi*sk(i)*T)];
        Vz = Vz * tf(num_i, den_i, T);
    end
end

function den = resonatorCoeff(Fk, sigma, T)
    % Returns the denominator polynomial [1, b1, b2] for a single
    % second-order resonator with formant frequency Fk and half-bandwidth σ.
    den = [1, ...
           -2 * exp(-2*pi*sigma*T) * cos(2*pi*Fk*T), ...
            exp(-4*pi*sigma*T)];
end

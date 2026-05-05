% =========================================================================
% A1 — Audio Recording
%
% Records a short speech sample via the system microphone and saves it
% to a WAV file for use in subsequent analysis scripts (A2–A5).
%
% Parameters
%   fs       : Sampling frequency (Hz)
%   duration : Recording length (seconds)
%   bits     : Bit depth
%   channels : Number of audio channels (1 = mono)
% =========================================================================

fs       = 16000;  % Sampling frequency in Hz
duration = 3;      % Recording duration in seconds
bits     = 16;     % Bit depth
channels = 1;      % Mono

% Create the audio recorder object
recObj = audiorecorder(fs, bits, channels);

disp('Press Enter to begin recording...');
pause;

disp('Recording...');
recordblocking(recObj, duration);  % Blocking call — waits until done
disp('Recording complete.');

% Save the recorded signal to a WAV file
filename = 'myRecording.wav';
audiowrite(filename, getaudiodata(recObj), fs);
disp(['Saved to: ' filename]);

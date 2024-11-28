
% Read the audio data and sample rate
[audioSignal, Fs] = audioread('music.wav');

% If your audio has multiple channels, extract only the first channel
if size(audioSignal, 2) > 1
    audioSignal = audioSignal(:, 1); % Keep only the first channel
end

% Play the audio (optional)
sound(audioSignal, Fs);

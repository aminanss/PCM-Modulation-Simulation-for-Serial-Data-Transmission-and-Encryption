%% Defining our audioSignal
% Read an audio waveform from an MP3 file
[audioSignal, Fs] = audioread('music.wav');

% If your audio has multiple channels, extract only the first channel
if size(audioSignal, 2) > 1
    audioSignal = audioSignal(:, 1); % Keep only the first channel
end
sound(audioSignal, Fs);

% Create a time vector
t = (0:length(audioSignal)-1) / Fs;

% Plot the audio signal
plot(t, audioSignal);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Audio Signal in Time Domain');
grid on;

% Compute the FFT of the audio signal
N = length(audioSignal);
Y = fft(audioSignal);
frequencies = linspace(0, Fs/2, N/2 + 1); % Only positive frequencies

% Plot the magnitude spectrum
figure;
plot(frequencies, abs(Y(1:N/2 + 1)));
title('One-Sided FFT of Audio Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

figure;
audioSignal = audioSignal(:); % Ensure it's a column vector
spectrogram(audioSignal, 1024, 512, [], Fs, 'yaxis');
title('Spectrogram of Audio Signal');


%% filtering

% Design a lowpass filter with a 1 kHz cutoff frequency
cutoff_freq_1k = 1e3; % Hz
d1 = designfilt('lowpassiir', 'FilterOrder', 5, 'HalfPowerFrequency', cutoff_freq_1k, 'SampleRate', Fs);

% Compute and plot the frequency response
[h1, f] = freqz(d1, 1024, Fs);
plot(f, mag2db(abs(h1)));
title('Frequency Response of Lowpass Filter (1 kHz Cutoff)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;

% Design a lowpass filter with a 2 kHz cutoff frequency
cutoff_freq_2k = 2e3; % Hz
d2 = designfilt('lowpassiir', 'FilterOrder', 5, 'HalfPowerFrequency', cutoff_freq_2k, 'SampleRate', Fs);

% Compute and plot the frequency response
[h2, ~] = freqz(d2, 1024, Fs);
figure;
plot(f, mag2db(abs(h2)));
title('Frequency Response of Lowpass Filter (2 kHz Cutoff)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;



% Filter the audio signal with the 1 kHz cutoff filter
filtered_signal_1k = filter(d1, audioSignal);
% Filter the audio signal with the 2 kHz cutoff filter
filtered_signal_2k = filter(d2, audioSignal);



% Compute the FFT of filtered_signal_1k
N_1k = length(filtered_signal_1k);
Y_1k = fft(filtered_signal_1k);
frequencies_1k = linspace(0, 1000/2, N_1k/2 + 1);

% Plot the magnitude spectrum
figure;
subplot(2,1,1);
plot(frequencies_1k, abs(Y_1k(1:N_1k/2 + 1)));
title('One-Sided FFT of Filtered Signal (1 kHz Cutoff)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% Create a spectrogram
subplot(2,1,2);
spectrogram(filtered_signal_1k, 1024, 512, [], 1000, 'yaxis');
title('Spectrogram of Filtered Signal (1 kHz Cutoff)');

% Compute the FFT of filtered_signal_2k
N_2k = length(filtered_signal_2k);
Y_2k = fft(filtered_signal_2k);
frequencies_2k = linspace(0, 2000/2, N_2k/2 + 1);

% Plot the magnitude spectrum
figure;
subplot(2,1,1);
plot(frequencies_2k, abs(Y_2k(1:N_2k/2 + 1)));
title('One-Sided FFT of Filtered Signal (2 kHz Cutoff)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% Create a spectrogram
subplot(2,1,2);
spectrogram(filtered_signal_2k, 1024, 512, [], 2000, 'yaxis');
title('Spectrogram of Filtered Signal (2 kHz Cutoff)');


%% resampling the audio

% Resample to 2 kHz
desiredFs_2k = 2e3;
resampled_signal_2k = resample(filtered_signal_1k, desiredFs_2k, Fs);

% Resample to 4 kHz
desiredFs_4k = 4e3;
resampled_signal_4k = resample(filtered_signal_2k, desiredFs_4k, Fs);

% Plot the resampled signal in the time domain
t_2k = (0:length(resampled_signal_2k)-1) / desiredFs_2k;
figure;
plot(t_2k, resampled_signal_2k);
title('Resampled Filtered Signal (1 kHz Cutoff, 2 kHz Sampling Rate)');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

% Plot the resampled signal in the time domain
t_4k = (0:length(resampled_signal_4k)-1) / desiredFs_4k;
figure;
plot(t_4k, resampled_signal_4k);
title('Resampled Filtered Signal (2 kHz Cutoff, 4 kHz Sampling Rate)');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

%% Quantization (4bit)
% just applied quantization on the 2k sampled audio

% Quantize the audio signal (4-bit)
quantized_4bit = quantizeSignal(resampled_signal_2k, 4);

% Calculate quantization error
error_4bit = resampled_signal_2k - quantized_4bit;

% Create time vector
t = (0:length(resampled_signal_2k)-1) / Fs; % Assuming Fs is the original sampling rate

% Plot original signal and 4-bit quantized signal
figure;
subplot(2,1,1);
plot(t, resampled_signal_2k, 'b', 'linewidth', 2);
hold on;
plot(t, quantized_4bit, 'r--', 'linewidth', 1.5);
legend('Original Signal', '4-Bit Quantized');
title('Comparison: Original vs. 4-Bit Quantized Signals');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

% Plot quantization error
subplot(2,1,2);
plot(t, error_4bit, 'r', 'linewidth', 1.5);
title('Quantization Error (4-Bit)');
xlabel('Time (seconds)');
ylabel('Error');
grid on;


%% Quantization (8bit)
% Quantize the audio signal (8-bit)
quantized_8bit = quantizeSignal(resampled_signal_2k, 8);

% Calculate quantization error
error_8bit = resampled_signal_2k - quantized_8bit;

% Create time vector
t = (0:length(resampled_signal_2k)-1) / Fs; % Assuming Fs is the original sampling rate

% Plot original signal and 8-bit quantized signal
figure;
subplot(2,1,1);
plot(t, resampled_signal_2k, 'b', 'linewidth', 2);
hold on;
plot(t, quantized_8bit, 'g--', 'linewidth', 1.5);
legend('Original Signal', '8-Bit Quantized');
title('Comparison: Original vs. 8-Bit Quantized Signals');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

% Plot quantization error
subplot(2,1,2);
plot(t, error_8bit, 'g', 'linewidth', 1.5);
title('Quantization Error (8-Bit)');
xlabel('Time (seconds)');
ylabel('Error');
grid on;

%% Binary code


% Determine the number of quantization levels (2^bits)
numLevels_4bit = 2^4; % 4-bit quantization
numLevels_8bit = 2^8; % 8-bit quantization

% Map quantized values to integer indices
indices_4bit = round((quantized_4bit - min(quantized_4bit)) / (max(quantized_4bit) - min(quantized_4bit)) * (numLevels_4bit - 1));
indices_8bit = round((quantized_8bit - min(quantized_8bit)) / (max(quantized_8bit) - min(quantized_8bit)) * (numLevels_8bit - 1));

% Convert integer indices to binary
binary_4bit = de2bi(indices_4bit, 4, 'left-msb'); % 4-bit binary representation
binary_8bit = de2bi(indices_8bit, 8, 'left-msb'); % 8-bit binary representation


%% Convert binary to Gray code

% Assuming you have 'binary_4bit' and 'binary_8bit' from the previous step

% Convert binary values to Gray code
gray_4bit = zeros(size(binary_4bit));
gray_8bit = zeros(size(binary_8bit));

for i = 1:size(binary_4bit, 1)
    gray_4bit(i, :) = bin2gray(binary_4bit(i, :), 'pam', 4);
end

for i = 1:size(binary_8bit, 1)
    gray_8bit(i, :) = bin2gray(binary_8bit(i, :), 'pam', 8);
end

%% Generating serial binary seq from gray codes

% Convert each row to a one-dimensional vector
binary_seq_4bit = reshape(gray_4bit', 1, []);
% Convert each row to a one-dimensional vector
binary_seq_8bit = reshape(gray_8bit', 1, []);

%% Generating digital signal (NRZ)
% Parameters
bit_duration = 1/47548; % Duration of each bit (in seconds)
sampling_frequency = 475480; % Sampling frequency (Hz)
total_duration = 1 - 1/sampling_frequency;    % Total duration of the pulse train (in seconds)

% Generate the pulse train
t = 0:1/sampling_frequency:total_duration;
pulse_width = round(bit_duration * sampling_frequency); % Number of points per bit
nrz_pulse_train = repelem(binary_seq_4bit, pulse_width);

% Plot the pulse train
figure;
plot(t, nrz_pulse_train, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Amplitude');
title('NRZ Pulse Train');
grid on;

%% Generating digital signal (RZ)

% Parameters
bit_duration = 1/47548; % Duration of each bit (in seconds)
total_duration = 1 - 1/sampling_frequency; % Adjusted total duration
sampling_frequency = 475480; % Sampling frequency (Hz)

% Generate the RZ pulse train
t = 0:1/sampling_frequency:total_duration;
pulse_width_1 = round(bit_duration * sampling_frequency / 2); % 5 points of 1
pulse_width_0 = round(bit_duration * sampling_frequency); % 10 points of 0

% Initialize the pulse train
rz_pulse_train = zeros(1, length(t));

% Set 5 points of 1 followed by 5 points of 0 for bit 1
for i = 1:length(binary_seq_4bit)
    if binary_seq_4bit(i) == 1
        start_idx = (i - 1) * pulse_width_0 + 1;
        rz_pulse_train(start_idx : start_idx + pulse_width_1 - 1) = 1;
    end
end

figure;
% Plot the RZ pulse train
plot(t, rz_pulse_train, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Amplitude');
title('Return-to-Zero (RZ) Pulse Train');
grid on;


%% FFT from both RZ and NRZ pulse trains
% Compute the FFT of the RZ pulse train
Y_rz = fft(rz_pulse_train);

% Calculate the frequency axis
Fs = sampling_frequency;
nfft_rz = length(Y_rz);
f_rz = (Fs/2) * linspace(0, 1, nfft_rz/2 + 1);

% Take the magnitude of the one-sided spectrum
Y_rz_mag = abs(Y_rz(1:nfft_rz/2 + 1));

% Compute the FFT of the NRZ pulse train
Y_nrz = fft(nrz_pulse_train);

% Calculate the frequency axis
nfft_nrz = length(Y_nrz);
f_nrz = (Fs/2) * linspace(0, 1, nfft_nrz/2 + 1);

% Take the magnitude of the one-sided spectrum
Y_nrz_mag = abs(Y_nrz(1:nfft_nrz/2 + 1));

% Plot the magnitude spectra
figure;
subplot(2, 1, 1);
plot(f_rz, 20*log10(Y_rz_mag), 'b', 'LineWidth', 2);
title('Magnitude Spectrum of RZ Pulse Train');
xlabel('Frequency (Hz)');
ylabel('|FFT(rz\_pulse\_train)| (dB)');
grid on;

subplot(2, 1, 2);
plot(f_nrz, 20*log10(Y_nrz_mag), 'r', 'LineWidth', 2);
title('Magnitude Spectrum of NRZ Pulse Train');
xlabel('Frequency (Hz)');
ylabel('|FFT(nrz\_pulse\_train)| (dB)');
grid on;

sgtitle('One-Sided FFT of RZ and NRZ Pulse Trains');


%% find power of both signals
% Parameters (adjust as needed)
total_duration = 1 - 1/sampling_frequency; % Total duration (in seconds)

% Compute the energy for RZ pulse train
energy_rz = sum(rz_pulse_train.^2);

% Compute the power for RZ pulse train
power_rz = energy_rz / total_duration;

% Compute the energy for NRZ pulse train
energy_nrz = sum(nrz_pulse_train.^2);

% Compute the power for NRZ pulse train
power_nrz = energy_nrz / total_duration;

% Display the results
disp(['Power of RZ pulse train: ', num2str(power_rz), ' Watts']);
disp(['Power of NRZ pulse train: ', num2str(power_nrz), ' Watts']);

%% Generating AWGN for SNR = 10 dB

% Parameters
SNR_dB = 10; % Desired SNR in dB

% Convert power to dBW for RZ pulse train
rz_power_dB = 10*log10(power_rz);
rz_noise_power_dB = rz_power_dB - SNR_dB;
rz_noise_power_linear = 10^(rz_noise_power_dB/10);

% Convert power to dBW for NRZ pulse train
nrz_power_dB = 10*log10(power_nrz);
nrz_noise_power_dB = nrz_power_dB - SNR_dB;
nrz_noise_power_linear = 10^(nrz_noise_power_dB/10);

% Generate white Gaussian noise samples for RZ and NRZ
rng('default'); % Set the random seed for reproducibility
noise_rz = sqrt(rz_noise_power_linear) * randn(size(rz_pulse_train)) / (length(t)/100);
noise_nrz = sqrt(nrz_noise_power_linear) * randn(size(nrz_pulse_train)) / (length(t)/100);

% Add noise to the pulse trains
rz_pulse_train_noisy = rz_pulse_train + noise_rz;
nrz_pulse_train_noisy = nrz_pulse_train + noise_nrz;

% Plot the noise
t = 0:1/sampling_frequency:total_duration;
figure;
subplot(2, 1, 1);
plot(t, noise_rz, 'b', 'LineWidth', 2);
title('Noise for RZ Pulse Train');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
plot(t, noise_nrz, 'r', 'LineWidth', 2);
title('Noise for NRZ Pulse Train');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
% Plot the noisy pulse trains
t = 0:1/sampling_frequency:total_duration;
figure;
subplot(2, 1, 1);
plot(t, rz_pulse_train_noisy, 'b', 'LineWidth', 2);
title('Noisy RZ Pulse Train');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
plot(t, nrz_pulse_train_noisy, 'r', 'LineWidth', 2);
title('Noisy NRZ Pulse Train');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

sgtitle('Pulse Trains with White Gaussian Noise (SNR = 10 dB)');

%% Generating AWGN for SNR = -10 dB

% Parameters
SNR_dB = -10; % Desired SNR in dB

% Convert power to dBW for RZ pulse train
rz_power_dB = 10*log10(power_rz);
rz_noise_power_dB = rz_power_dB - SNR_dB;
rz_noise_power_linear = 10^(rz_noise_power_dB/10);

% Convert power to dBW for NRZ pulse train
nrz_power_dB = 10*log10(power_nrz);
nrz_noise_power_dB = nrz_power_dB - SNR_dB;
nrz_noise_power_linear = 10^(nrz_noise_power_dB/10);

% Generate white Gaussian noise samples for RZ and NRZ
rng('default'); % Set the random seed for reproducibility
noise_rz = sqrt(rz_noise_power_linear) * randn(size(rz_pulse_train)) / (length(t)/10);
noise_nrz = sqrt(nrz_noise_power_linear) * randn(size(nrz_pulse_train)) / (length(t)/10);

% Add noise to the pulse trains
rz_pulse_train_noisy = rz_pulse_train + noise_rz;
nrz_pulse_train_noisy = nrz_pulse_train + noise_nrz;

% Plot the noisy pulse trains
t = 0:1/sampling_frequency:total_duration;
figure;
subplot(2, 1, 1);
plot(t, rz_pulse_train_noisy, 'b', 'LineWidth', 2);
title('Noisy RZ Pulse Train');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
plot(t, nrz_pulse_train_noisy, 'r', 'LineWidth', 2);
title('Noisy NRZ Pulse Train');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

sgtitle('Pulse Trains with White Gaussian Noise (SNR = 10 dB)');

%% Reciever

% Parameters
threshold = 0.5;
sample_interval = 10;

% Sample the noisy pulse trains
sampled_rz = rz_pulse_train_noisy(3:sample_interval:end);
sampled_nrz = nrz_pulse_train_noisy(3:sample_interval:end);

% Apply threshold and generate binary sequences
rz_detected_sequence = (sampled_rz > threshold)';
nrz_detected_sequence = (sampled_nrz > threshold)';


%% detecting digital signal

nrz_detected_signal_gray = reshape(nrz_detected_sequence, 4, [])';
rz_detected_signal_gray = reshape(rz_detected_sequence, 4, [])';

%% convert gray format into binary format

% Convert Gray code to binary
binary_rz_detected_signal = zeros(size(rz_detected_signal_gray));
for i = 1:size(rz_detected_signal_gray, 1)
    binary_rz_detected_signal(i, :) = bitxor(rz_detected_signal_gray(i, :), [0 rz_detected_signal_gray(i, 1:3)]);
end

% Convert Gray code to binary
binary_nrz_detected_signal = zeros(size(nrz_detected_signal_gray));
for i = 1:size(nrz_detected_signal_gray, 1)
    binary_nrz_detected_signal(i, :) = bitxor(nrz_detected_signal_gray(i, :), [0 nrz_detected_signal_gray(i, 1:3)]);
end

%% Decoding binary codes to voltage levels

% Assuming binary_rz_detected_signal is a matrix with dimensions 11887x4
% Each row represents a 4-bit binary data point

% Normalize binary values to the range [0, 15]
normalized_values = bi2de(binary_rz_detected_signal, 'left-msb');

% Map normalized values to the desired voltage range
voltage_range_min = -0.5;
voltage_range_max = 0.5;
voltage_levels = voltage_range_min + (normalized_values / 16) * (voltage_range_max - voltage_range_min);

%% play original audio

sound(audioSignal, 44100);
%% play decoded signal
sound(voltage_levels, 2000);

% Write the waveform to a WAV file
audiowrite('pmc_recieved_music_4bit.wav', voltage_levels, 2000);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Doing the same on the 8bit quantized signal

%% Generating digital signal (NRZ)
% Parameters
bit_duration = 1/95096; % Duration of each bit (in seconds)
sampling_frequency = 950960; % Sampling frequency (Hz)
total_duration = 1 - 1/sampling_frequency;    % Total duration of the pulse train (in seconds)

% Generate the pulse train
t = 0:1/sampling_frequency:total_duration;
pulse_width = round(bit_duration * sampling_frequency); % Number of points per bit
nrz_pulse_train = repelem(binary_seq_8bit, pulse_width);

% Plot the pulse train
plot(t, nrz_pulse_train, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Amplitude');
title('NRZ Pulse Train');
grid on;

%% Generating digital signal (RZ)


% Generate the RZ pulse train
t = 0:1/sampling_frequency:total_duration;
pulse_width_1 = round(bit_duration * sampling_frequency / 2); % 5 points of 1
pulse_width_0 = round(bit_duration * sampling_frequency); % 10 points of 0

% Initialize the pulse train
rz_pulse_train = zeros(1, length(t));

% Set 5 points of 1 followed by 5 points of 0 for bit 1
for i = 1:length(binary_seq_8bit)
    if binary_seq_8bit(i) == 1
        start_idx = (i - 1) * pulse_width_0 + 1;
        rz_pulse_train(start_idx : start_idx + pulse_width_1 - 1) = 1;
    end
end

% Plot the RZ pulse train
plot(t, rz_pulse_train, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Amplitude');
title('Return-to-Zero (RZ) Pulse Train');
grid on;


%% FFT from both RZ and NRZ pulse trains
% Compute the FFT of the RZ pulse train
Y_rz = fft(rz_pulse_train);

% Calculate the frequency axis
Fs = sampling_frequency;
nfft_rz = length(Y_rz);
f_rz = (Fs/2) * linspace(0, 1, nfft_rz/2 + 1);

% Take the magnitude of the one-sided spectrum
Y_rz_mag = abs(Y_rz(1:nfft_rz/2 + 1));

% Compute the FFT of the NRZ pulse train
Y_nrz = fft(nrz_pulse_train);

% Calculate the frequency axis
nfft_nrz = length(Y_nrz);
f_nrz = (Fs/2) * linspace(0, 1, nfft_nrz/2 + 1);

% Take the magnitude of the one-sided spectrum
Y_nrz_mag = abs(Y_nrz(1:nfft_nrz/2 + 1));

% Plot the magnitude spectra
figure;
subplot(2, 1, 1);
plot(f_rz, 20*log10(Y_rz_mag), 'b', 'LineWidth', 2);
title('Magnitude Spectrum of RZ Pulse Train');
xlabel('Frequency (Hz)');
ylabel('|FFT(rz\_pulse\_train)| (dB)');
grid on;

subplot(2, 1, 2);
plot(f_nrz, 20*log10(Y_nrz_mag), 'r', 'LineWidth', 2);
title('Magnitude Spectrum of NRZ Pulse Train');
xlabel('Frequency (Hz)');
ylabel('|FFT(nrz\_pulse\_train)| (dB)');
grid on;

sgtitle('One-Sided FFT of RZ and NRZ Pulse Trains');


%% find power of both signals
% Parameters (adjust as needed)
total_duration = 1 - 1/sampling_frequency; % Total duration (in seconds)

% Compute the energy for RZ pulse train
energy_rz = sum(rz_pulse_train.^2);

% Compute the power for RZ pulse train
power_rz = energy_rz / total_duration;

% Compute the energy for NRZ pulse train
energy_nrz = sum(nrz_pulse_train.^2);

% Compute the power for NRZ pulse train
power_nrz = energy_nrz / total_duration;

% Display the results
disp(['Power of RZ pulse train: ', num2str(power_rz), ' Watts']);
disp(['Power of NRZ pulse train: ', num2str(power_nrz), ' Watts']);

%% Generating AWGN for SNR = 10 dB

% Parameters
SNR_dB = 10; % Desired SNR in dB

% Convert power to dBW for RZ pulse train
rz_power_dB = 10*log10(power_rz);
rz_noise_power_dB = rz_power_dB - SNR_dB;
rz_noise_power_linear = 10^(rz_noise_power_dB/10);

% Convert power to dBW for NRZ pulse train
nrz_power_dB = 10*log10(power_nrz);
nrz_noise_power_dB = nrz_power_dB - SNR_dB;
nrz_noise_power_linear = 10^(nrz_noise_power_dB/10);

% Generate white Gaussian noise samples for RZ and NRZ
rng('default'); % Set the random seed for reproducibility
noise_rz = sqrt(rz_noise_power_linear) * randn(size(rz_pulse_train)) / (length(t)/10);
noise_nrz = sqrt(nrz_noise_power_linear) * randn(size(nrz_pulse_train)) / (length(t)/10);

% Add noise to the pulse trains
rz_pulse_train_noisy = rz_pulse_train + noise_rz;
nrz_pulse_train_noisy = nrz_pulse_train + noise_nrz;

% Plot the noisy pulse trains
t = 0:1/sampling_frequency:total_duration;
figure;
subplot(2, 1, 1);
plot(t, rz_pulse_train_noisy, 'b', 'LineWidth', 2);
title('Noisy RZ Pulse Train');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
plot(t, nrz_pulse_train_noisy, 'r', 'LineWidth', 2);
title('Noisy NRZ Pulse Train');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

sgtitle('Pulse Trains with White Gaussian Noise (SNR = 10 dB)');

%% Generating AWGN for SNR = -10 dB

% Parameters
SNR_dB = -10; % Desired SNR in dB

% Convert power to dBW for RZ pulse train
rz_power_dB = 10*log10(power_rz);
rz_noise_power_dB = rz_power_dB - SNR_dB;
rz_noise_power_linear = 10^(rz_noise_power_dB/10);

% Convert power to dBW for NRZ pulse train
nrz_power_dB = 10*log10(power_nrz);
nrz_noise_power_dB = nrz_power_dB - SNR_dB;
nrz_noise_power_linear = 10^(nrz_noise_power_dB/10);

% Generate white Gaussian noise samples for RZ and NRZ
rng('default'); % Set the random seed for reproducibility
noise_rz = sqrt(rz_noise_power_linear) * randn(size(rz_pulse_train)) / (length(t)/10);
noise_nrz = sqrt(nrz_noise_power_linear) * randn(size(nrz_pulse_train)) / (length(t)/10);

% Add noise to the pulse trains
rz_pulse_train_noisy = rz_pulse_train + noise_rz;
nrz_pulse_train_noisy = nrz_pulse_train + noise_nrz;

% Plot the noisy pulse trains
t = 0:1/sampling_frequency:total_duration;
figure;
subplot(2, 1, 1);
plot(t, rz_pulse_train_noisy, 'b', 'LineWidth', 2);
title('Noisy RZ Pulse Train');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
plot(t, nrz_pulse_train_noisy, 'r', 'LineWidth', 2);
title('Noisy NRZ Pulse Train');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

sgtitle('Pulse Trains with White Gaussian Noise (SNR = 10 dB)');

%% Reciever

% Parameters
threshold = 0.5;
sample_interval = 10;

% Sample the noisy pulse trains
sampled_rz = rz_pulse_train_noisy(3:sample_interval:end);
sampled_nrz = nrz_pulse_train_noisy(3:sample_interval:end);

% Apply threshold and generate binary sequences
rz_detected_sequence = (sampled_rz > threshold)';
nrz_detected_sequence = (sampled_nrz > threshold)';


%% detecting digital signal

nrz_detected_signal_gray = reshape(nrz_detected_sequence, 8, [])';
rz_detected_signal_gray = reshape(rz_detected_sequence, 8, [])';

%% convert gray format into binary format

% Convert Gray code to binary
binary_rz_detected_signal = zeros(size(rz_detected_signal_gray));
for i = 1:size(rz_detected_signal_gray, 1)
    binary_rz_detected_signal(i, :) = bitxor(rz_detected_signal_gray(i, :), [0 rz_detected_signal_gray(i, 1:7)]);
end

% Convert Gray code to binary
binary_nrz_detected_signal = zeros(size(nrz_detected_signal_gray));
for i = 1:size(nrz_detected_signal_gray, 1)
    binary_nrz_detected_signal(i, :) = bitxor(nrz_detected_signal_gray(i, :), [0 nrz_detected_signal_gray(i, 1:7)]);
end

%% Decoding binary codes to voltage levels

normalized_values = bi2de(binary_rz_detected_signal, 'left-msb');

% Map normalized values to the desired voltage range
voltage_range_min = -1;
voltage_range_max = 1;
voltage_levels = voltage_range_min + (normalized_values / 31) * (voltage_range_max - voltage_range_min);

%% ploting decoded voltage levels

% Create time vector
t = (0:length(voltage_levels)-1) / 2000; % Assuming Fs is the original sampling rate

% Plot original signal and 4-bit quantized signal
figure;
plot(t, voltage_levels, 'b', 'linewidth', 2);
%% play original audio

sound(audioSignal, 44100);
%% play decoded signal
sound(voltage_levels, 2000);

% Write the waveform to a WAV file
audiowrite('pmc_recieved_music_8bit.wav', voltage_levels, 2000);





%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% encryption
% Load the music signal (replace with your actual audio file)
music_signal = audioread('music.wav');
music_signal = music_signal(:, 1); 

% Generate a secret key for DES (56 bits)
secret_key = randi([0, 255], 1, 7);

% Initialize DES cipher
cipher = des(secret_key, 'ECB', 'zeropad');

% Encrypt the music signal using DES
encrypted_signal = cipher.encrypt(music_signal);

% Save the encrypted signal
audiowrite('encrypted_music.wav', encrypted_signal, 44100);

disp('Music signal encrypted and saved as "encrypted_music.wav"');

% ... (Send the encrypted signal to the receiver)

% Read the encrypted signal (receiver's side)
received_encrypted_signal = audioread('encrypted_music.wav');

% Decrypt the signal using the same secret key
decrypted_signal = cipher.decrypt(received_encrypted_signal);

% Play the decrypted music signal
soundsc(decrypted_signal, 44100);

disp('Music signal decrypted and played.');


%%  Recording our own voice

Fs = 44100;  % Sampling rate (adjust as needed)
recObj = audiorecorder(Fs, 16, 1);  % Create an audio recorder object
disp('Start speaking...');
recordblocking(recObj, 5);  % Record for 5 seconds (adjust duration as needed)
disp('End of recording.');
myRecording = getaudiodata(recObj);
audiowrite('my_voice.wav', myRecording, Fs);

%% record and hearing at the same time
% Set your desired sampling rate (e.g., 44.1 kHz)
sampleRate = 44100;

% Create the audioPlayerRecorder object
playRec = audioPlayerRecorder(sampleRate);

% Start recording
record(playRec);

% Play audio (you'll hear it while recording)
% You can play any audio data here (e.g., a sine wave, your voice, etc.)
% For example, to play a sine wave:
t = 0:1/sampleRate:5;  % 5 seconds
freq = 440;  % Frequency (adjust as desired)
audioData = sin(2*pi*freq*t);
play(playRec, audioData);

% Wait for a few seconds (adjust as needed)
pause(5);

% Stop recording
stop(playRec);

% Get the recorded audio
recordedAudio = getaudiodata(playRec);

% Save the audio as a WAV file
audiowrite('my_voice.wav', recordedAudio, sampleRate);



%% Quantization Function
function quantizedSignal = quantizeSignal(signal, numBits)
    % Determine the maximum and minimum values of the signal
    maxVal = max(signal);
    minVal = min(signal);

    % Number of quantization levels
    numLevels = 2^numBits;

    % Step size
    stepSize = (maxVal - minVal) / numLevels;

    % Quantize the signal
    quantizedSignal = round((signal - minVal) / stepSize) * stepSize + minVal;

    % Ensure the quantized signal does not exceed the original range
    quantizedSignal = max(min(quantizedSignal, maxVal), minVal);
end

%% reading signal 

% Read the audio file
[audioSignal, Fs] = audioread('D:\DANESHGAH\term6\SisMox\Project\music.wav');

% Play the 2 kHz resampled signal
sound(audioSignal, Fs);


% Get the time vector
t = (0:length(audioSignal)-1)/Fs;

% Plot the time-domain signal
figure;
subplot(2,1,1);
plot(t, audioSignal);
title('Time Domain Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Perform Fourier Transform
n = length(audioSignal);
Y = fft(audioSignal);

% Calculate the frequency axis
f_2k = (0:n-1)*(Fs/n);

% Get the magnitude of the FFT
P = abs(Y)/n;

% Plot the frequency-domain signal
subplot(2,1,2);
plot(f_2k, P);
title('Frequency Domain Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]); % Limit the x-axis to half the sampling rate for better visualization

%% Filtering audio signals with cut off freqs of 1k and 2k

% Design parameters
Fs_orig = 44100; % Original sampling rate
Fc_2kHz = 1000; % Cutoff frequency for 2 kHz sampling
N = 100; % Filter order

% Design the low-pass filter
lpFilt_2kHz = fir1(N, Fc_2kHz/(Fs_orig/2));

% Plot the frequency response
figure;
freqz(lpFilt_2kHz, 1, 1024, Fs_orig);
title('Low-Pass Filter for 2 kHz Sampling');


% Design parameters
Fc_4kHz = 2000; % Cutoff frequency for 4 kHz sampling

% Design the low-pass filter
lpFilt_4kHz = fir1(N, Fc_4kHz/(Fs_orig/2));

% Plot the frequency response
figure;
freqz(lpFilt_4kHz, 1, 1024, Fs_orig);
title('Low-Pass Filter for 4 kHz Sampling');


% Filter the signal for 2 kHz sampling
filteredSignal_2kHz = filter(lpFilt_2kHz, 1, audioSignal);

% Filter the signal for 4 kHz sampling
filteredSignal_4kHz = filter(lpFilt_4kHz, 1, audioSignal);

% Plot time-domain signal for after filtering
figure;
subplot(2,1,1);
plot(t, filteredSignal_2kHz);
title('Filtered Time Domain Signal for 2 kHz Sampling (1kHz cut off)');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot time-domain signal for 4 kHz sampling
subplot(2,1,2);
plot(t, filteredSignal_4kHz);
title('Filtered Time Domain Signal for 4 kHz Sampling(2kHz cut off)');
xlabel('Time (s)');
ylabel('Amplitude');


%% Resampling signals
% Resample to 2 kHz
targetFs_2k = 2000;  % Desired sampling frequency (2 kHz)
resampledSignal_2k = resample(filteredSignal_2kHz, targetFs_2k, Fs);

% Resample to 4 kHz
targetFs_4k = 4000;  % Desired sampling frequency (4 kHz)
resampledSignal_4k = resample(filteredSignal_4kHz, targetFs_4k, Fs);


% Time vector for 2 kHz resampled signal
t_2k = (0:length(resampledSignal_2k) - 1) / targetFs_2k;

% Time vector for 4 kHz resampled signal
t_4k = (0:length(resampledSignal_4k) - 1) / targetFs_4k;


% Plot time-domain signal for 2 kHz sampling
figure;
subplot(2,2,1);
plot(t_2k, resampledSignal_2k);
title('Filtered Time Domain Signal for 2 kHz Sampling');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot time-domain signal for 4 kHz sampling
subplot(2,2,2);
plot(t_4k, resampledSignal_4k);
title('Filtered Time Domain Signal for 4 kHz Sampling');
xlabel('Time (s)');
ylabel('Amplitude');

% Compute FFT for 2 kHz resampled signal
Y_2k = fft(resampledSignal_2k);

% Compute FFT for 4 kHz resampled signal
Y_4k = fft(resampledSignal_4k);

NFFT_2k = length(Y_2k);  % Length of FFT
f_2k = (0:NFFT_2k-1) * (targetFs_2k / NFFT_2k);  % Frequency vector

NFFT_4k = length(Y_2k);  % Length of FFT
f_4k = (0:NFFT_4k-1) * (targetFs_2k / NFFT_4k);  % Frequency vector

figure;

% Plot for 2 kHz resampled signal
subplot(2, 1, 1);
plot(f_2k, abs(Y_2k));
title('FFT Spectrum (2 kHz)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

% Plot for 4 kHz resampled signal
subplot(2, 1, 2);
plot(f_4k, abs(Y_4k));
title('FFT Spectrum (4 kHz)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

% Adjust subplot spacing
sgtitle('FFT of Resampled Audio Signals');


%%
% Play the 2 kHz resampled signal
sound(resampledSignal_2k, targetFs_2k);

%%
% Play the 4 kHz resampled signal
sound(resampledSignal_4k, targetFs_4k);


%%
%{
کوانتیزیشن

تفاوت سیگنال اصلی و سیگنال کوانتیزه شده
The quantized signal will have fewer levels than the original signal. With 8-bit quantization, there will be 256 levels, and with 4-bit quantization, there will be only 16 levels. This reduced level of representation causes the quantized signal to have a more "stair-stepped" appearance compared to the original smooth signal.
%}

% Filter the music signal (already done, assuming the filtered signals are available)
% filteredSignal_2kHz and filteredSignal_4kHz

% Quantize the filtered signal for 2 kHz sampling
quantizedSignal_2kHz_8bit = quantizeSignal(filteredSignal_2kHz, 8);
quantizedSignal_2kHz_4bit = quantizeSignal(filteredSignal_2kHz, 4);

% Quantize the filtered signal for 4 kHz sampling
quantizedSignal_4kHz_8bit = quantizeSignal(filteredSignal_4kHz, 8);
quantizedSignal_4kHz_4bit = quantizeSignal(filteredSignal_4kHz, 4);

% Get time vectors for the filtered signals
t_2kHz = (0:length(filteredSignal_2kHz)-1)/Fs_orig;
t_4kHz = (0:length(filteredSignal_4kHz)-1)/Fs_orig;

% Plot the original and quantized signals for 2 kHz sampling, 8-bit
figure;
plot(t_2kHz, filteredSignal_2kHz, 'b'); hold on;
plot(t_2kHz, quantizedSignal_2kHz_8bit, 'r');
title('Quantized Signal for 2 kHz Sampling (8-bit)');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Original Signal', 'Quantized Signal (8-bit)');
hold off;

% Plot the original and quantized signals for 2 kHz sampling, 4-bit
figure;
plot(t_2kHz, filteredSignal_2kHz, 'b'); hold on;
plot(t_2kHz, quantizedSignal_2kHz_4bit, 'r');
title('Quantized Signal for 2 kHz Sampling (4-bit)');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Original Signal', 'Quantized Signal (4-bit)');
hold off;

% Plot the original and quantized signals for 4 kHz sampling, 8-bit
figure;
plot(t_4kHz, filteredSignal_4kHz, 'b'); hold on;
plot(t_4kHz, quantizedSignal_4kHz_8bit, 'r');
title('Quantized Signal for 4 kHz Sampling (8-bit)');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Original Signal', 'Quantized Signal (8-bit)');
hold off;

% Plot the original and quantized signals for 4 kHz sampling, 4-bit
figure;
plot(t_4kHz, filteredSignal_4kHz, 'b'); hold on;
plot(t_4kHz, quantizedSignal_4kHz_4bit, 'r');
title('Quantized Signal for 4 kHz Sampling (4-bit)');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Original Signal', 'Quantized Signal (4-bit)');
hold off;


%% 2.3.3 خطا

% Calculate the quantization error for 2 kHz sampling, 8-bit
quantError_2kHz_8bit = filteredSignal_2kHz - quantizedSignal_2kHz_8bit;

% Calculate the quantization error for 2 kHz sampling, 4-bit
quantError_2kHz_4bit = filteredSignal_2kHz - quantizedSignal_2kHz_4bit;

% Calculate the quantization error for 4 kHz sampling, 8-bit
quantError_4kHz_8bit = filteredSignal_4kHz - quantizedSignal_4kHz_8bit;

% Calculate the quantization error for 4 kHz sampling, 4-bit
quantError_4kHz_4bit = filteredSignal_4kHz - quantizedSignal_4kHz_4bit;

% Plot quantization error for 2 kHz sampling, 8-bit
figure;
plot(t_2kHz, quantError_2kHz_8bit);
title('Quantization Error for 2 kHz Sampling (8-bit)');
xlabel('Time (s)');
ylabel('Error');

% Plot quantization error for 2 kHz sampling, 4-bit
figure;
plot(t_2kHz, quantError_2kHz_4bit);
title('Quantization Error for 2 kHz Sampling (4-bit)');
xlabel('Time (s)');
ylabel('Error');

% Plot quantization error for 4 kHz sampling, 8-bit
figure;
plot(t_4kHz, quantError_4kHz_8bit);
title('Quantization Error for 4 kHz Sampling (8-bit)');
xlabel('Time (s)');
ylabel('Error');

% Plot quantization error for 4 kHz sampling, 4-bit
figure;
plot(t_4kHz, quantError_4kHz_4bit);
title('Quantization Error for 4 kHz Sampling (4-bit)');
xlabel('Time (s)');
ylabel('Error');


%%
%{
Encoding 2.4.1
استفاده از کد گری (Gray code) در مخابرات دیجیتال در مرحله‌ی اندازه‌گیری به دلایل زیر مفید است:
کاهش خطاهای بیتی: کد گری به صورتی طراحی شده است که بین هر دو عدد متوالی تنها یک بیت متغیر باشد. به عبارت دیگر، اگر در فرآیند اندازه‌گیری از یک وضعیت به وضعیت دیگر برویم، تنها یک بیت تغییر می‌کند. این ویژگی باعث می‌شود که اگر در انتقال داده‌ها خطا رخ دهد و یک بیت تغییر کند، این خطا در اندازه‌گیری بسیار کمتر اثرگذار باشد، زیرا فقط یک بیت تغییر کرده است و احتمال اینکه این تغییر به وضعیت دیگری منتقل شود کمتر است.
کاربرد در اندازه‌گیری های دیجیتال: در اندازه‌گیری‌های دیجیتال، مانند ADC (Analog-to-Digital Converter) که ورودی آنها یک سیگنال آنالوگ است که به سیگنال دیجیتال تبدیل می‌شود، استفاده از کد گری می‌تواند کمک کند تا نویز و خطاهایی که به علت محدودیت‌های سیستم ایجاد می‌شود، کاهش یابد.
سادگی پیاده‌سازی: کد گری از نظر پیاده‌سازی نسبتاً ساده است و به راحتی می‌توان آن را در سخت‌افزار و نرم‌افزار پیاده‌سازی کرد. این سادگی اجازه می‌دهد تا در اندازه‌گیری‌های بسیار سریع و با دقت، از آن استفاده شود.
به طور کلی، استفاده از کد گری در مخابرات دیجیتال و اندازه‌گیری‌های دیجیتال به دلیل ویژگی‌های خاصی که در کاهش خطاها و افزایش دقت اندازه‌گیری‌ها موثر هستند، بسیار مفید است و به عنوان یک استاندارد در برنامه‌ها و سیستم‌های مخابراتی استفاده می‌شود.
%}
% Function to convert binary to Gray code


% Quantize the signal again (assuming quantized signals already available)
quantizedSignal_2kHz_8bit = quantizeSignal(filteredSignal_2kHz, 8);
quantizedSignal_2kHz_4bit = quantizeSignal(filteredSignal_2kHz, 4);

% Convert quantized samples to binary
binary_2kHz_8bit = decimalToBinary(quantizedSignal_2kHz_8bit, 8);
binary_2kHz_4bit = decimalToBinary(quantizedSignal_2kHz_4bit, 4);

% Convert binary to Gray code
gray_2kHz_8bit = binaryToGray(binary_2kHz_8bit);
gray_2kHz_4bit = binaryToGray(binary_2kHz_4bit);

% Display a portion of the result for verification
disp('First 10 values of 2kHz 8-bit Gray code:');
disp(gray_2kHz_8bit(1:10, :));

disp('First 10 values of 2kHz 4-bit Gray code:');
disp(gray_2kHz_4bit(1:10, :));

% Convert quantized samples for 4 kHz sampling
quantizedSignal_4kHz_8bit = quantizeSignal(filteredSignal_4kHz, 8);
quantizedSignal_4kHz_4bit = quantizeSignal(filteredSignal_4kHz, 4);

% Convert quantized samples to binary
binary_4kHz_8bit = decimalToBinary(quantizedSignal_4kHz_8bit, 8);
binary_4kHz_4bit = decimalToBinary(quantizedSignal_4kHz_4bit, 4);

% Convert binary to Gray code
gray_4kHz_8bit = binaryToGray(binary_4kHz_8bit);
gray_4kHz_4bit = binaryToGray(binary_4kHz_4bit);

% Display a portion of the result for verification
disp('First 10 values of 4kHz 8-bit Gray code:');
disp(gray_4kHz_8bit(1:10, :));

disp('First 10 values of 4kHz 4-bit Gray code:');
disp(gray_4kHz_4bit(1:10, :));

%%
%{
2.4.2

Serial Communication Protocols:Serial communication protocols are fundamental for transferring data between devices using a series of bits over a single communication line. This method contrasts with parallel communication, which uses multiple lines for data transfer. Serial communication is favored for its simplicity and effectiveness over longer distances, despite being slower than parallel communication.
I2C (Inter-Integrated Circuit): This protocol is used for short-distance communication within a single device, such as between different chips on a circuit board. I2C uses two wires (SDA and SCL) and supports multiple devices on the same bus with speeds up to 400 kbps�
C2I Protocol:The C2I (Command and Control Interface) protocol is a communication standard used in specific applications, such as military or aerospace, where reliable and secure communication is critical. It provides structured and controlled communication between command centers and deployed units or devices. Details about the specific technical implementation of C2I protocols can vary widely based on the application and security requirements.
%}


%% 2.4.3 & 2.4.4
% Example binary sequence
binarySequence = [1 0 1 1 0 0 1 0 1 1 1 0];

% Baud rate and pulse duration
B = 2000;  % Example baud rate of 2000 bits per second
T_pulse = 1 / B;  % Pulse duration

% Time vector for the entire sequence
% The length of the time vector should cover all bits in the sequence
t = 0:T_pulse:T_pulse*(length(binarySequence)-1)/B;  % Corrected time vector

% Generate RZ and NRZ pulses
RZ_pulse = zeros(1, length(t));
NRZ_pulse = zeros(1, length(t));

for i = 1:length(binarySequence)
    if binarySequence(i) == 1
        % RZ pulse: set the first half of the bit period to 1
        RZ_pulse((i-1)*B + 1 : (i-1)*B + B/2) = 1;
        % NRZ pulse: set the entire bit period to 1
        NRZ_pulse((i-1)*B + 1 : i*B) = 1;
    end
end

% Plot Time Domain
figure;
subplot(2,1,1);
stairs(t, RZ_pulse, 'LineWidth', 2);
title('RZ Pulse in Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-0.5, 1.5]);

subplot(2,1,2);
stairs(t, NRZ_pulse, 'LineWidth', 2);
title('NRZ Pulse in Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-0.5, 1.5]);

% Compute Fourier Transform
RZ_freq = fft(RZ_pulse);
NRZ_freq = fft(NRZ_pulse);
f_2k = (0:length(t)-1)*(B/length(t));

% Plot Frequency Domain
figure;
subplot(2,1,1);
plot(f_2k, abs(RZ_freq));
title('RZ Pulse in Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

subplot(2,1,2);
plot(f_2k, abs(NRZ_freq));
title('NRZ Pulse in Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

%%
%{
کانال مخابراتی
P=N1�∑n=0N−1�x[n]
%}
% Define the amplitude of the pulses
amplitude = 1;

% Example binary sequence
binarySequence = [1 0 1 1 0 0 1 0 1 1 1 0];

% Baud rate and pulse duration
B = 2000;  % Example baud rate of 2000 bits per second
T_pulse = 1 / B;  % Pulse duration

% Time vector for the entire sequence
t = 0:T_pulse:(length(binarySequence)*T_pulse - T_pulse);

% Generate RZ and NRZ pulses
RZ_pulse = zeros(1, length(t));
NRZ_pulse = zeros(1, length(t));

for i = 1:length(binarySequence)
    if binarySequence(i) == 1
        % RZ pulse: set the first half of the bit period to the amplitude
        RZ_pulse((i-1)*B + 1 : (i-1)*B + B/2) = amplitude;
        % NRZ pulse: set the entire bit period to the amplitude
        NRZ_pulse((i-1)*B + 1 : i*B) = amplitude;
    end
end

% Calculate power for RZ pulses
power_RZ = mean(RZ_pulse.^2);

% Calculate power for NRZ pulses
power_NRZ = mean(NRZ_pulse.^2);

% Display the results
fprintf('Power of RZ pulse signal: %f\n', power_RZ);
fprintf('Power of NRZ pulse signal: %f\n', power_NRZ);

%%
%{
3.1.2 & 3.1.3
Determining Proper SNR for Modulation
SNR = 10: Typically, an SNR of 10 dB is considered reasonable for many communication systems, providing a good balance between signal quality and noise. It ensures that the signal is distinguishable from the noise, resulting in reliable data transmission.
SNR = -10: An SNR of -10 dB indicates that the noise power is significantly higher than the signal power, leading to a very noisy signal. This condition is generally not suitable for reliable communication as the signal is likely to be overwhelmed by noise.
For PCM (Pulse Code Modulation) and similar digital modulation schemes, an SNR of 10 dB or higher is typically preferred to ensure robust and error-free communication.
%}

% Define the amplitude of the pulses
amplitude = 1;

% Example binary sequence
binarySequence = [1 0 1 1 0 0 1 0 1 1 1 0];

% Baud rate and pulse duration
B = 2000;  % Example baud rate of 2000 bits per second
T_pulse = 1 / B;  % Pulse duration

% Time vector for the entire sequence
t = 0:T_pulse:T_pulse*(length(binarySequence)-1)/B;  % Corrected time vector

% Generate RZ and NRZ pulses
RZ_pulse = zeros(1, length(t));
NRZ_pulse = zeros(1, length(t));

for i = 1:length(binarySequence)
    if binarySequence(i) == 1
        % RZ pulse: set the first half of the bit period to the amplitude
        RZ_pulse((i-1)*B + 1 : (i-1)*B + B/2) = amplitude;
        % NRZ pulse: set the entire bit period to the amplitude
        NRZ_pulse((i-1)*B + 1 : i*B) = amplitude;
    end
end

% Choose one of the pulses for further processing
signal = NRZ_pulse;  % Can be changed to RZ_pulse

% Calculate power of the original signal
signal_power = mean(signal.^2);

% Define desired SNR values
SNR_10 = 10;  % SNR = 10 dB
SNR_neg10 = -10;  % SNR = -10 dB

% Calculate noise power
noise_power_10 = signal_power / (10^(SNR_10 / 10));
noise_power_neg10 = signal_power / (10^(SNR_neg10 / 10));

% Generate white noise for both SNR conditions
noise_10 = sqrt(noise_power_10) * randn(size(signal));
noise_neg10 = sqrt(noise_power_neg10) * randn(size(signal));

% Add noise to the signal
noisy_signal_10 = signal + noise_10;
noisy_signal_neg10 = signal + noise_neg10;

% Plot the original and noisy signals in time domain
figure;
subplot(3,1,1);
stairs(1:length(signal), signal, 'LineWidth', 2);
title('Original Signal in Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-1.5, 1.5]);

subplot(3,1,2);
stairs(1:length(noisy_signal_neg10), noisy_signal_10, 'LineWidth', 2);
title('Noisy Signal with SNR = 10 dB in Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-1.5, 1.5]);

subplot(3,1,3);
stairs(1:length(noisy_signal_neg10), noisy_signal_neg10, 'LineWidth', 2);
title('Noisy Signal with SNR = -10 dB in Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-1.5, 1.5]);

% Compute Fourier Transform for noisy signals
noisy_signal_10_freq = fft(noisy_signal_10);
noisy_signal_neg10_freq = fft(noisy_signal_neg10);
f_2k = (0:length(noisy_signal_10)-1)*(B/length(noisy_signal_10));

% Compute Fourier Transform for noise signals
noise_10_freq = fft(noise_10);
noise_neg10_freq = fft(noise_neg10);

% Plot Frequency Domain
figure;
subplot(2,1,1);
plot(f_2k, abs(noisy_signal_10_freq));
title('Noisy Signal with SNR = 10 dB in Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

subplot(2,1,2);
plot(f_2k, abs(noisy_signal_neg10_freq));
title('Noisy Signal with SNR = -10 dB in Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');


% Plot the white noise in time domain
figure;
subplot(2,1,1);
plot(t, noise_10, 'LineWidth', 2);
title('White Noise with SNR = 10 dB in Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-2*sqrt(noise_power_10), 2*sqrt(noise_power_10)]);

subplot(2,1,2);
plot(t, noise_neg10, 'LineWidth', 2);
title('White Noise with SNR = -10 dB in Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-2*sqrt(noise_power_neg10), 2*sqrt(noise_power_neg10)]);

% Plot the white noise in frequency domain
figure;
subplot(2,1,1);
plot(f_2k, abs(noise_10_freq));
title('White Noise with SNR = 10 dB in Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

subplot(2,1,2);
plot(f_2k, abs(noise_neg10_freq));
title('White Noise with SNR = -10 dB in Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

%%
%{
گیرنده
4.1.1 & 4.1.2 & 4.1.3 & 4.1.4
4.1.4:
Reconstruction Filter:
The filter is an FIR low-pass filter designed using the window method.
This filter is essential for removing high-frequency components introduced during the sampling process, thereby reconstructing the original analog signal.
4.1.2:
معیار BER به معنای "Bit Error Rate" یا نرخ خطا در بیت، یکی از معیارهای مهم و رایج در ارتباطات دیجیتال است که در شبکه‌های انتقال داده استفاده می‌شود. BER به تعداد بیت‌هایی اشاره دارد که به اشتباه ارسال یا دریافت می‌شوند نسبت به کل بیت‌های ارسالی یا دریافتی. به عبارت دیگر، BER نشان‌دهنده درصد بیت‌هایی است که در هر دوره ارسال یا دریافت خطا دارند.
%}

% Define the amplitude of the pulses
amplitude = 1;

% Original binary sequence
binarySequence = [1 0 1 1 0 0 1 0 1 1 1 0];

% Baud rate and pulse duration
B = 2000;  % Example baud rate of 2000 bits per second
T_pulse = 1 / B;  % Pulse duration
fs = B;  % Sampling frequency

% Generate the time vector for the original signal
t = 0:T_pulse:(length(binarySequence)*T_pulse - T_pulse);

% Generate NRZ signal for transmission
NRZ_pulse = zeros(1, length(binarySequence) * fs);
for i = 1:length(binarySequence)
    if binarySequence(i) == 1
        NRZ_pulse((i-1)*fs + 1 : i*fs) = amplitude;
    end
end

% Assume received noisy signals
% Using previously generated noisy signals for illustration
received_signal_10 = noisy_signal_10 > 0.5;  % Thresholding for binary decisions
received_signal_neg10 = noisy_signal_neg10 > 0.5;  % Thresholding for binary decisions

% Generate indices to extract the sequence correctly
indices = round(linspace(1, length(received_signal_10), length(binarySequence)));

% Extract the sequence from the received signal
extracted_sequence_10 = received_signal_10(indices);
extracted_sequence_neg10 = received_signal_neg10(indices);

% Calculate BER
BER_10 = sum(binarySequence ~= extracted_sequence_10) / length(binarySequence);
BER_neg10 = sum(binarySequence ~= extracted_sequence_neg10) / length(binarySequence);

% Display BER results
fprintf('BER for SNR = 10 dB: %f\n', BER_10);
fprintf('BER for SNR = -10 dB: %f\n', BER_neg10);

% Plot the extracted digital signal in the time domain
t_digital = 0:T_pulse:(length(extracted_sequence_10) * T_pulse - T_pulse);

figure;
subplot(2,1,1);
stairs(t_digital, extracted_sequence_10, 'LineWidth', 2);
title('Extracted Digital Signal for SNR = 10 dB in Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-0.5, 1.5]);

subplot(2,1,2);
stairs(t_digital, extracted_sequence_neg10, 'LineWidth', 2);
title('Extracted Digital Signal for SNR = -10 dB in Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-0.5, 1.5]);

% DAC and reconstruction filter (Low-pass filter)
% Define a low-pass reconstruction filter
f_cutoff = B / 2;  % Cutoff frequency
order = 50;  % Filter order

% Ensure the normalized cutoff frequency is in the range [0, 1]
normalized_cutoff = f_cutoff / (fs / 2);

% Design a low-pass FIR filter using the window method
reconstruction_filter = fir1(order, normalized_cutoff);

% Apply reconstruction filter to the extracted digital signals
analog_signal_10 = filter(reconstruction_filter, 1, extracted_sequence_10);
analog_signal_neg10 = filter(reconstruction_filter, 1, extracted_sequence_neg10);

% Plot the analog signals in the time domain
figure;
subplot(2,1,1);
plot(t_digital, analog_signal_10, 'LineWidth', 2);
title('Reconstructed Analog Signal for SNR = 10 dB');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
plot(t_digital, analog_signal_neg10, 'LineWidth', 2);
title('Reconstructed Analog Signal for SNR = -10 dB');
xlabel('Time (s)');
ylabel('Amplitude');

% Explanation of the Reconstruction Filter
fprintf('The reconstruction filter used is a low-pass FIR filter designed using the window method.\n');
fprintf('This filter helps in reconstructing the analog signal from the sampled digital sequence.\n');


%% Encoding Functions
function grayCode = binaryToGray(binaryCode)
    % Convert the binary code to numeric array for processing
    binaryCode = binaryCode - '0';
    % Initialize the Gray code array with the same size as binary code
    grayCode = binaryCode;
    % The most significant bit (MSB) is the same for both binary and Gray code
    grayCode(:, 1) = binaryCode(:, 1);
    % Each subsequent bit is obtained by XORing the current bit with the previous bit
    grayCode(:, 2:end) = xor(binaryCode(:, 1:end-1), binaryCode(:, 2:end));
    % Convert the numeric Gray code array back to character array
    grayCode = char(grayCode + '0');
end

% Function to convert decimal to binary with specified number of bits
function binaryCode = decimalToBinary(decimal, numBits)
    % Ensure the decimal values are within the range of the quantization levels
    maxVal = 2^numBits - 1;
    decimal = max(0, min(decimal, maxVal));  % Clamp the values between 0 and maxVal
    % Convert to binary string array with specified number of bits
    binaryCode = dec2bin(decimal, numBits);
end

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

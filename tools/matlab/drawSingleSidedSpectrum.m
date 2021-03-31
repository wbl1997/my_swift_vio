function drawSingleSidedSpectrum(t, X)
T = median(diff(t));   % Sampling period
Fs = 1/T;              % Sampling frequency                    
L = length(t);     % Length of signal
Y = fft(X);

P2 = abs(Y/L);
P1 = P2(1:floor(L/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1);

[M, I] = max(P1);
xlim([0, f(10*I)])

title('Single-Sided Amplitude Spectrum of S(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
end

function [xp, yp, peakf] = sinefit(t, X, frequencyRatio)
% fit a sine wave to periodical signals buries in noise
% t: timestamps in seconds
% X: signals

T = median(diff(t));   % Sampling period
Fs = 1/T;              % Sampling frequency                    
L = length(t);         % Length of signal
Y = fft(X);

P2 = abs(Y/L);
P1 = P2(1:floor(L/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
[M, I] = max(P1);
f = Fs*(0:(L/2))/L;
peakf = f(I);

threshold = max(abs(Y)) * frequencyRatio;
pureY = Y;
pureY(abs(Y) < threshold) = 0;

yp = ifft(pureY);
xp = t;

end

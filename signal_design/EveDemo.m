%-------------------------------------------------------------------------
% EVEDemo
% Pete Wyckoff, KA3WCA, May 7, 2026.
%-------------------------------------------------------------------------
clear all; close all; clc;
fprintf('\nStarting Monte Carlo Simluation...\n');

%-------------------------------------------------------------------------
%                Key Parameters from ORI Link Budget
%-------------------------------------------------------------------------
CNo_dB = 0;                           %Carrier to Noise Density (dB-Hz)
fDop = 2.67;                          %Doppler Spread (Hz)
SNR_RBW_dB = CNo_dB - 10*log10(fDop); %SNR in one resolution bandwidth(dB)

%-------------------------------------------------------------------------
%   Waveform Parameters (keep these fixed for now when using FEC due to 
%   mapping encoded bits to symbols.  Can improve later for flexibility!)
%-------------------------------------------------------------------------
M = 4096;                         %M-ary for modulation (#)
NC = 540;                         %non-coherent combinations (#)

%-------------------------------------------------------------------------
%    FEC Parameters (keep these fixed for now due to mapping to 
%    symbols into the modulator.  Can improve later for flexibility!)
%-------------------------------------------------------------------------
fec.n = 127;                      %encoded FEC message length in (bits)
fec.k = 106;                      %user message length in (bits)

%-------------------------------------------------------------------------
%                  Monte Carlo Simulation Parameters
%-------------------------------------------------------------------------
TRIALS = 100;                    %Monte Carlo Trials (#)

%-------------------------------------------------------------------------
%                             Run Tests
%-------------------------------------------------------------------------
for n=1:11
    
  userBitRate = log2(M) / (1/fDop) / NC * (fec.k / fec.n); %(bits/second)  
  success = runTest(SNR_RBW_dB, M, NC, fec, TRIALS);       %does it work?
  store(n) = sum(success);     %keep track of how many messages decoded
  bitRate(n) = userBitRate;    %keep track of user bit rate
  
  fprintf('%i successful / %i trial at %.3f (bits/sec) \n\n', store(n), TRIALS, bitRate(n));
  
  NC = NC - 40;   %decrease # of non-coherent combinations for next test
end


figure;
plot(bitRate, store/TRIALS * 100);
xlabel('User Bit Rate (bits/second)', 'FontSize', 14);
ylabel('Messages Decoded (Percentage)', 'FontSize', 14);
grid on;
title(sprintf('C/N_o = %.1d (dB), Doppler Spread = %.2f (Hz)', CNo_dB, fDop));

%EOF
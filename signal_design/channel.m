function r = channel(y, EsNo_dB, RayleighFlag)
%-------------------------------------------------------------------------
%  Runs channel model.  There are two versions, depending whether the 
%  function is called with 2 or 3 input arguments:
%
%  r = channel(y, EsNo_dB) <- Noise, random phase, but no envelope fading
%  r = channel(y, EsNo_dB, 'Rayleigh') <-Adds Rayleigh envelope fading
%
%  Variable   In/Out   Description
%  --------   ------   -----------
%  EsNo_dB    Input    One or more test points that specify the SNR
%                      within the FFT resolution bandwidth in (dB).
%
%  r          Output   Signal at channel output.
%
%  Pete Wyckoff, KA3WCA, May 8, 2026.
%-------------------------------------------------------------------------
  M = length(y);                                  % # of samples in signal
  w = (1/sqrt(2)) * (randn(1,M) + 1i*randn(1,M)); % additive white noise
  
  %AWGN is unit variance.  Varying the signal level sets operating point.
  %Each pass has independent and random channel phase -- uniformly 
  %distributed on [0, 2pi) -- because each pass is later in time
  %by 1 / Doppler Spread.
  
  signalCoef = 10^(EsNo_dB/20) * (sqrt(M)) * exp(1i*2*pi*rand(1));
  
  if (nargin > 2)  %Futher apply independent Rayleigh fading on envelope
    %The ORI link budget C/No is viewed as average C/No for Venus bounce.
    %Thus, applying additional fading has a mean value of '1' for the 
    %power.  Following the equations for a Rayleigh distribution from
    % Steven Kay, Fundamentals of Statistical Signal Processing Theory,
    % Volume 2, Detection Theory, Prentice-Hall 1998, p. 30, sigma is 
    % computed from the mean value.  Subsequently, each independent
    % realization is computed using a function of two zero mean and 
    % independent Gaussian random variables, that are scaled by sigma, as 
    %follows:
    
    MeanVal = 1;
    sigma = sqrt((MeanVal^2) * (2/pi));
    fadingCoef = sqrt((sigma * randn(1))^2 + (sigma * randn(1))^2);
    
    %adjustment to exponential distribution is 1.05 dB
    signalCoef = 10^((EsNo_dB - 1.05)/20) * (sqrt(M)) * exp(1i*2*pi*rand(1));
    
    %apply envelope fading to signalCoef:
    signalCoef = signalCoef * fadingCoef;
  end
  
  r = signalCoef * y + w;
end

%EOF
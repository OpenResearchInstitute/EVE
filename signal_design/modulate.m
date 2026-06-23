function y = modulate(symbol, M)
%-------------------------------------------------------------------------
%  Modulates 8,192 samples of a tone specified by one input symbol.
%  y = modulate(symbol, M)
%
%  Variable   In/Out   Description
%  --------   ------   -----------
%  symbol     Input    One symbol that ranges from 1 to 4096
%
%  M          Input    4096-ary
%
%  r          Output   Signal at channel output.
%
%  This function is hard coded for the present operating point, but
%  it could have more flexibility with updates in the future if we
%  need them.
%
%  Pete Wyckoff, KA3WCA, May 8, 2026.
%-------------------------------------------------------------------------
  x = zeros(1,M);
  
  %space carriers more than MSK, such that any effect of Doppler Spread
  %leaves the various potential symbols orthogonal, or at least very
  %nearly orthogonal.  
  
  %This code is using an 8,192-pt FFT but restriting to 4096 possible 
  %tones (as symbols) that occupy every other possible tone of the FFT to 
  %keep the symbols further apart in frequency.
  
  mappedSymbol = 2 * (symbol-1) + 1; 
  x(mappedSymbol) = 1;
  y = ifft(x, 8192);                 %tone in the time domain
end %modulate

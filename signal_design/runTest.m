function success = runTest(SNR_RBW_dB, M, NC, fec, TRIALS)
%-------------------------------------------------------------------------
%  Runs Monte Carlo trials that determine whether the message is decoded 
%  successfully.
%
%  success = runTest(EsNo_dB, M, NC, fec, TRIALS)
%
%  Variable    In/Out   Description
%  --------    ------   -----------
%  SNR_RBW_dB  Input    One or more test points that specify the SNR
%                       within the FFT resolution bandwidth in (dB).
%
%  M           Input    M-ary value for the orthogonal FSK waveform (#).
%
%  NC          Input    Number of independent non-coherent combinations
%                       of FFT outputs before making a symbol decision
%                       in the receiver (#).
%
%  fec         Input    Structure that defines n & k as the number of 
%                       BCH encoded data bits and the number of user data
%                       bits per FEC block.
%
%  TRIALS      Input    Number of Monte Carlo trials to run at each EsNo_dB
%                       test point.
%
%  success     Output   True if message decoded correctly.  
%                       False otherwise.
%
%  Pete Wyckoff, KA3WCA, May 8, 2026.
%-------------------------------------------------------------------------

for test=1:length(SNR_RBW_dB)    %loop through Es/No points if more than 1
  bitErrors(test) = 0;           %init bit error counter to 0
  success = [];
  for trial=1:TRIALS             %Monte Carlo trials per test point
    if(mod(trial, 10) == 0)
      fprintf('Trial %i of %i trials for %i Non-Coh Combos.\n', ...
          trial, TRIALS, NC);    
    end
    
    msgTx = gf(randi([0 1], 1, fec.k));        %random message bits
    enc = bchenc(msgTx, fec.n, fec.k);         %BCH encoded bits
    
    %convert BCH encoded bits to modulator input format ------------------
    %(this is a section is hard coded for present M-ary and FEC!!!)
    tempBin = [];
    for symCount = 1:11       %loop through 11 symbols of 4096-ary
      if(symCount < 11)
        txSymGF = enc(12*(symCount-1)+(1:12));
      else
        txSymGF = enc(12*(symCount-1)+(1:7)); 
        txSymGF(8:12) = 0;    %zero pad final few TX symbols
      end
      txSymbol = double((txSymGF.x)) * (2.^(0:11))';
      
      %modulate symbol----------------------------------------------------
      %(this symbol is enough for ONE set of receiver FFT outputs.  For 
      % simulation, the same TX symbol is repeated for subsequent 
      % non-coherent combinations.  Importatnly, each of these repeats 
      % suffers a random and  independent phase as well as AWGN due to the 
      % channel model -- all before reaching the receiver DSP.  Thus, this
      % sprial has a perfectly valid simulation of performance, albiet
      % with some limited mathematical abstraction to "get 'er done.")
      y = modulate(txSymbol + 1, M);
    
      %Monte Carlo trials through noisy channel model---------------------
      xr = zeros(1, 8192);             %init. non-coherent accumulators
      for ncc=1:NC                     %non-coherent combinations loop
        %pass signal through noisy & fading channel
        r = channel(y, SNR_RBW_dB(test), 'Rayleigh'); 
        xr = xr + abs(fft(r));         %non-coherent avg. of FFT outputs
      end
      
      xr = xr(1:2:2*M);                       %keep only possible symbols
      
      %recall we are using every other carrier in the FFT to keep tones
      %spectally distict (orthogonal) despite Doppler spread of channel
      
      [~, rxSymbol(symCount)] = max(abs(xr)); %find the largest
      
      %convert symbol to GF-----------------------------------------------
      temp = dec2bin(rxSymbol(symCount)-1, 12);
      for n=1:12
        tempBin(end+1) = strcmp(temp(13-n), '1');
      end    
    end
    
    %TODO:  Since some bits (tail-end) were zero-padded to form an 
    %       interger number of symbols, those zero-padded symbols
    %       could be forced to zero here prior to BCH decoding, which
    %       would improve performance somewhat.  
    
    %BCH decoder----------------------------------------------------------
    noisycode = gf(tempBin(1:127));
    msgRx = bchdec(noisycode, fec.n, fec.k);         %BCH decoded
    
    %Test whether message decoded successfully----------------------------
    success(trial) = isequal(msgTx, msgRx);          %did it work?
    
  end %trial loop
  
  %presently only stores one test point:
  %if more than one test point, need to store tests here!!!
  
 end %test loop
end %runTest function
    
%EOF
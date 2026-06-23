# Landing Page for the EVE Spiral Efforts
---
## Signal Design and Construction
Top-level file "EveDemo.m" runs tests for Spiral #2.  The outcome plots percentage of messages that were successfully decoded versus the bit rate.  

Spiral #2 modulation is 4096 M-ary FSK generated, with carrier spacing wider than MSK in order to accomodate Doppler spread, and a BCH code with (n=127, k=106) for FEC.  The channel has Rayleigh fading with Doppler spread at 2.67 Hz.  The demodulator computes FFT's every 1/2.67 (seconds) and non-coherently combines these over one symbol period.  These combinations were needed since the C/No is expected to be 0 dB from the link budget, which equates to -4.2 dB in the FFT's resolution bandwidth.  Following non-coherent combinations,  BCH decoding delivers the original message bits.  

Spiral #2 does not include synchronization.  Although Dwingeloo and Stockert have tight synchronization -- sites share a hydrogen-maser via a white rabbit link -- so the design might work as is for those stations.  If the link has worse C/No than anticipated, the same message could be sent multiple times, time aligned, and then non-coherently combined prior to BCH decoding.

Avenues for future work include, code review, adding synchronization for other stations, and considering an improved FEC code (although BCH is not a bad code at all for short block lengths).

73 DE Pete KA3WCA

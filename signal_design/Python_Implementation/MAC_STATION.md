# EVE station on a Mac (Apple Silicon) -- B210 setup

Running the whole EVE station on one MacBook Air (Apple Silicon), alongside the K3NG
rotator. This is a supported path: Ettus documents the B210 on macOS, UHD 4.9 ships for
osx-arm64 (conda-forge even builds it against Python 3.14), and EVE's ~250 kSps rate is
four orders of magnitude below where Mac/USB throughput ever gets tight. ASCII only.

This file is the macOS-specific setup that does NOT depend on the station wrapper. The
exact bench commands (eve_b210.py) get written and BENCH.md's hardware section rewritten
AFTER the B210 arrives and `uhd_usrp_probe` runs clean -- so nothing here is guessed.

================================================================================
1. WHY THIS WORKS ON A MAC (and why the usual warning does not apply)
================================================================================
- The "use Linux for SDR" warning is about USB THROUGHPUT at tens of MSps. EVE runs at
  ~250 kSps (a 23.5 kHz signal). That is nowhere near a bottleneck on any Mac.
- We drive the radio through UHD's STOCK COMMAND-LINE BINARIES (uhd_usrp_probe,
  tx_samples_from_file, rx_samples_to_file), NOT the Python bindings. So there is no
  binding-vs-Python-version matching to fight, and it behaves the same on Mac and Linux.
- Your .sigmf-data is raw interleaved complex float32 (fc32), which tx_samples_from_file
  plays directly, and rx_samples_to_file produces the same format for eve_rx.py.
- The K3NG rotator is a separate serial/USB device. No conflict. One laptop is fine.

================================================================================
2. INSTALL UHD
================================================================================
Pick ONE. MacPorts is Ettus's recommended route and stays current.

MacPorts:
    # install MacPorts first (macports.org) if you do not have it
    sudo port selfupdate
    sudo port install uhd
    #  -> gives uhd_usrp_probe, uhd_find_devices, tx_samples_from_file,
    #     rx_samples_to_file, uhd_images_downloader

Homebrew (you already have /opt/homebrew):
    brew install uhd
    #  same binaries; version can lag MacPorts a little

conda-forge (only if you want the matched Python 3.14 bindings too, later):
    conda install -c conda-forge uhd
    #  osx-arm64 build exists for py3.14; not needed for the CLI-binary path

Then fetch the B210 firmware + FPGA images (required -- the B210 loads these every
power-up):
    uhd_images_downloader

Sanity that the tools are on PATH:
    which uhd_usrp_probe uhd_find_devices tx_samples_from_file rx_samples_to_file

================================================================================
3. THE B210 USB "MORPH" QUIRK ON macOS -- read this before you panic
================================================================================
On power-up the B210 loads firmware/FPGA and RE-ENUMERATES on USB ("morphs" from a
Cypress/Westbridge device into a USRP). On the older white B200/B210 this re-enumeration
FAILS about half the time, and UHD then cannot find the device. This is known and normal
on macOS -- it is not your setup being broken.

Fixes, most reliable first:
  1. Use the B210's EXTERNAL POWER SUPPLY (barrel jack), not USB bus power alone. This
     alone removes most of the flakiness.
  2. If uhd_usrp_probe cannot find it: unplug the USB cable and replug into the SAME port,
     then try again. With external power you can use any port.
  3. There is also a USB reset button (S700) on the board as a last resort.
Expect to replug occasionally. Once uhd_usrp_probe shows the B210, it is stable for the
session.

================================================================================
4. FIRST CONTACT (no RF, just prove the box talks)
================================================================================
    uhd_usrp_probe
      # should print the B210 motherboard + RF daughterboard info, the master clock
      # rate, and TX/RX frontends. If it errors on firmware, see section 3 (replug).
    uhd_find_devices
      # should list one B200-series device with a serial.
Copy the serial number somewhere; you will use it if you ever run two radios.

*** When both of the above run clean, paste the uhd_usrp_probe output back and I will:
    - write eve_b210.py (probe / loopback / run --schedule), using the exact binary
      names and options confirmed on YOUR box, and
    - rewrite BENCH.md section 2 with tested, copy-pasteable commands. ***

================================================================================
5. THE THREE TIMING REGIMES (same wrapper, one parameter: round-trip time)
================================================================================
- BENCH (today): round trip ~ 0. TX and RX run CONCURRENTLY over a cable+attenuator
  (full duplex). No gating, no sequencer. This is the first test.
- EME: round trip ~ 2.56 s. TX a short burst, stop, wait ~2.5 s, capture the echo window.
- EVE: round trip ~ 4.5 min. TX a 164.794 s symbol, stop, wait ~107 s, capture. Same
  shape as EME, just minutes-scale. Driven by eve_gated_*_schedule.json.
The CLI-binary path gives coarse (sub-second) timing, which is fine for all three because
we capture generous windows and the decoder locks onto the burst by energy. Sample-locked
T/R gating for on-air monostatic DSES is a hardware-sequencer job (GPIO), not software.

================================================================================
6. SAFETY REMINDERS FOR THE CABLE LOOPBACK (unchanged from BENCH.md)
================================================================================
- NEVER wire B210 TX -> RX bare. Use a fixed, power-rated 40-60 dB attenuator in that
  path. Start with more attenuation and reduce while watching the RX level (aim
  ~ -30 to -20 dBFS; back off gain if it clips).
- Tee the TX to the spectrum analyzer through a directional coupler (~ -30 dB coupled),
  not a bare tee.
- Start with low tx-gain. Confirm the comb on the SA before trusting any RX numbers.

================================================================================
7. WHAT IS SUPERSEDED
================================================================================
Once eve_b210.py exists, the earlier hardware TEMPLATES -- eve_tx_rx_b210.py and the
eve_loopback.py --hardware path -- are kept for reference only. They use the raw UHD
Python API written without a radio in front of it ("plausible", unverified). Do not run
them on the B210; use eve_b210.py.

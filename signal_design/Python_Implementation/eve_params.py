#!/usr/bin/env python3
"""
eve_params.py defines the EVE waveform's frequency-dependent parameters.
It takes an operating frequency and a target body. It derives tone spacing,
coherent-frame bandwidth (R_bw), coherence time, comb width, a sample-rate floor and a
DC-avoidance offset. There are no frequency-specific magic numbers or hard coding.
The only physical input is the measured Doppler half-width, cited below, and that
number is scaled with carrier frequency. The Spiral scheme is designed around the
short coherence time we have with the EVE channel.

WHAT WE KNOW (these are measured values)
  Venus echo Doppler half-width: 90% of the echo power falls within +/- 1 Hz of the
  sub-radar Doppler, measured from the CAMRAS 22-Mar-2025 EVE dataset at 1299.5 MHz
  (Estevez 2025, EA4GPZ and used in the ORI link-budget validation that gives
  CNR_1Hz = 0.645 dB). This is a HALF-width (this a +/- number). The coherence-relevant
  FULL width is twice it. Our fig_coherence.py caption instead says +/- 0.75 Hz
  from the same dataset. That discrepancy is unresolved. We default to the +/- 1 Hz
  value because the published CNR_1Hz result is built on it. Change ONE constant below
  to switch. This is the only measured number in the file.

  Moon echo Doppler half-width: the lunar libration spread has been measured and
  tabulated by the EME community. We use K1JT's "Frequency-Dependent
  Characteristics of the EME Path" (Joe Taylor, 2010), specifically his w10 column, 
  the full width at the 10% level. In other words the band containing 90% of the echo 
  power. That is the same 90%-power convention as the Venus number above, so it maps 
  straight in. Unlike Venus (one specular measurement we scale with frequency), the 
  Moon's w10 does not appear to scale cleanly with frequency, from the tables, so it is 
  stored as a per-band table and interpolated. K1JT's values assume stations at +/-40 deg 
  latitude. The real spread varies through the month with the libration rate (it goes to 
  zero at libration turning points), so the values are actually not fixed constants.

WHAT WE ASSUME (documented, not measured)
  The half-width scales linearly with carrier frequency for a fixed body and geometry
  (Doppler = 2*v*f/c). So the value at any frequency is anchor * (f / f_anchor). This
  reproduces Pete Wyckoff's Oct-2026 "~2.87 Hz at 2304 MHz" to within the measurement
  spread, which is why 2.87 is no longer stored. We regenerate it every time.
  (This linear-scaling assumption applies to the Venus anchor. The Moon uses the K1JT
  table directly instead of scaling, because its w10 is not linear in frequency.)

CONVENTION (baked into the NAMES so we don't get confused)
  *_halfwidth_hz   one-sided (+/-) width
  *_fullwidth_hz   total width = 2 * halfwidth
  R_bw_hz          coherent-frame bandwidth = fullwidth (a tone must stay inside one
                   bin over a coherent frame). coherence_time_s = 1 / R_bw_hz.
  tone_spacing_hz  = 2 * R_bw_hz  (one empty guard bin between tones from Pete's mapping)
  integ_halfband_hz  half-band for CNR PSD integration = halfwidth (NOT doubled)
"""
from dataclasses import dataclass
from typing import Optional
import math

C_M_S = 299_792_458.0

# --------------------------------------------------------------------------------------
# The measured numbers. Everything else derives from them.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    """A radar target body. Holds the measured Doppler half-width anchor and the
    frequency it was measured at. Halfwidth_hz(f) scales it to any frequency.
    We are doing it this way because the Doppler Spread comes from the rotation
    of the target body.

    Two ways to do this model, pick one per target:
      (a) anchor: one measured half-width at one frequency, scaled ANCHOR * f/f0
          (linear with frequency; a single specular measurement, e.g. Venus).
      (b) table_fullwidth_hz: {freq_hz: 90%-power FULL width}, linearly interpolated
          (use when a body's width does NOT scale cleanly with f, e.g. the Moon).
    halfwidth_hz(f) returns the one-sided (+/-) 90%-power half-width either way."""
    name: str
    provenance: str
    echo_halfwidth_hz_at_anchor: Optional[float] = None   # +/- Hz, 90%-power (anchor mode)
    anchor_freq_hz: Optional[float] = None
    table_fullwidth_hz: Optional[dict] = None             # {freq_hz: FULL width, 90% power}

    def halfwidth_hz(self, freq_hz: float) -> float:
        # (b) per-band table, linearly interpolated on the FULL-width values, then halved
        if self.table_fullwidth_hz:
            fs = sorted(self.table_fullwidth_hz)
            if freq_hz < fs[0] or freq_hz > fs[-1]:
                raise ValueError(
                    "%s: %.4f MHz is outside the tabulated range %.0f-%.0f MHz; "
                    "extrapolation not trusted. Supply spread_halfwidth_hz=... instead."
                    % (self.name, freq_hz / 1e6, fs[0] / 1e6, fs[-1] / 1e6))
            lo = max(f for f in fs if f <= freq_hz)
            hi = min(f for f in fs if f >= freq_hz)
            if hi == lo:
                full = self.table_fullwidth_hz[lo]
            else:
                a = (freq_hz - lo) / (hi - lo)
                full = (1 - a) * self.table_fullwidth_hz[lo] + a * self.table_fullwidth_hz[hi]
            return 0.5 * full
        # (a) single measured anchor, scaled linearly with frequency
        if self.echo_halfwidth_hz_at_anchor is not None:
            return self.echo_halfwidth_hz_at_anchor * (freq_hz / self.anchor_freq_hz)
        raise ValueError(
            "No Doppler-spread model for target '%s'. Supply one explicitly "
            "via spread_halfwidth_hz=... -- do not guess." % self.name)


VENUS = Target(
    name="venus",
    provenance="Estevez 2025 (EA4GPZ), CAMRAS 22-Mar-2025; ORI validation CNR_1Hz=0.645 dB",
    echo_halfwidth_hz_at_anchor=1.0,        # +/- Hz, 90% power  (set 0.75 for fig_coherence value)
    anchor_freq_hz=1299.5e6,
)
MOON = Target(
    name="moon",
    # K1JT 2010, "Frequency-Dependent Characteristics of the EME Path", w10 column =
    # full width at the 10% level (contains 90% of power) which is the same 90%-power convention
    # as the Venus number. Stations at +/-40 deg latitude and representative libration state.
    # w10 does not appear to scale cleanly with f, so this is a per-band interpolated table.
    provenance="K1JT 2010 w10 (90%-power full width), +/-40 deg lat, libration-condition dependent",
    table_fullwidth_hz={
        144e6: 1.1, 432e6: 3.0, 1296e6: 10.0, 2304e6: 20.0,
        3400e6: 31.0, 5760e6: 60.0, 10368e6: 128.0,
    },
)
TARGETS = {t.name: t for t in (VENUS, MOON)}


# --------------------------------------------------------------------------------------
# Derived waveform parameters for a given (frequency, target)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WaveformParams:
    freq_hz: float
    target: str
    # physical widths
    halfwidth_hz: float
    fullwidth_hz: float
    # waveform geometry
    R_bw_hz: float
    tone_spacing_hz: float
    coherence_time_s: float
    M: int
    bits_per_symbol: int
    comb_bandwidth_hz: float
    integ_halfband_hz: float
    # sampling / placement (hardware conveniences; tunable, not physics)
    freq_offset_hz: float
    fs_floor_hz: float
    # bookkeeping
    scaled_from_anchor: bool
    provenance: str

    def summary(self) -> str:
        return (
            "f=%.4f MHz  target=%s\n"
            "  half-width      = +/- %.3f Hz   (%s)\n"
            "  full width      =     %.3f Hz\n"
            "  R_bw            =     %.3f Hz   (= full width)\n"
            "  tone spacing    =     %.3f Hz   (= 2*R_bw, one guard bin)\n"
            "  coherence time  =     %.3f s    (= 1/R_bw)\n"
            "  comb bandwidth  = %.1f Hz  (M=%d, %d bits/sym)\n"
            "  CNR integ band  = +/- %.3f Hz   (half-width, NOT doubled)\n"
            "  freq offset     = %.0f Hz    fs floor = %.0f Hz\n"
            % (self.freq_hz/1e6, self.target, self.halfwidth_hz,
               "scaled/interpolated" if self.scaled_from_anchor else "at anchor freq",
               self.fullwidth_hz, self.R_bw_hz, self.tone_spacing_hz,
               self.coherence_time_s, self.comb_bandwidth_hz, self.M, self.bits_per_symbol,
               self.integ_halfband_hz, self.freq_offset_hz, self.fs_floor_hz))


def compute(freq_hz: float,
            target: str = "venus",
            M: int = 4096,
            spread_halfwidth_hz: Optional[float] = None,
            rbw_margin: float = 1.0,
            dc_guard_hz: float = 2000.0,
            fs: Optional[float] = None,
            fs_headroom: float = 1.2) -> WaveformParams:
    """Derive the full waveform parameter set for a frequency + target.

    spread_halfwidth_hz : override the measured/scaled half-width (e.g. for a body with
                          no model, or to test a value). If None, use the target's model.
    rbw_margin          : R_bw = full_width * rbw_margin (>=1.0 widens the bin for safety).
    dc_guard_hz         : how far to lift the comb off DC/LO leakage (hardware property).
    fs                  : force a sample rate; else report the floor the comb needs.
    """
    tgt = TARGETS[target] if isinstance(target, str) else target
    if spread_halfwidth_hz is not None:
        halfwidth = float(spread_halfwidth_hz)
        scaled = False
        provenance = "user-supplied spread_halfwidth_hz=%.3f Hz" % halfwidth
    else:
        halfwidth = tgt.halfwidth_hz(freq_hz)
        scaled = (tgt.anchor_freq_hz != freq_hz)   # False only at an exact anchor freq
        provenance = tgt.provenance

    fullwidth = 2.0 * halfwidth
    R_bw = fullwidth * rbw_margin
    spacing = 2.0 * R_bw                      # Pete's mapping: one empty guard bin
    coherence = 1.0 / R_bw
    bits = int(round(math.log2(M)))
    comb_bw = M * spacing
    offset = dc_guard_hz                      # one-sided comb sits above DC by this much
    fs_floor = fs_headroom * 2.0 * (offset + comb_bw)   # complex baseband; keep comb < fs/2

    return WaveformParams(
        freq_hz=freq_hz, target=getattr(tgt, "name", str(target)),
        halfwidth_hz=halfwidth, fullwidth_hz=fullwidth,
        R_bw_hz=R_bw, tone_spacing_hz=spacing, coherence_time_s=coherence,
        M=M, bits_per_symbol=bits, comb_bandwidth_hz=comb_bw,
        integ_halfband_hz=halfwidth,
        freq_offset_hz=offset, fs_floor_hz=(fs if fs is not None else fs_floor),
        scaled_from_anchor=scaled, provenance=provenance)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="EVE waveform parameters from frequency + target.")
    ap.add_argument("--rf", type=float, nargs="*", default=[1296e6, 1299.5e6, 2304e6, 2330e6],
                    help="one or more RF frequencies in Hz")
    ap.add_argument("--target", default="venus")
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--spread-halfwidth-hz", type=float, default=None)
    a = ap.parse_args()
    print("EVE waveform parameters  (anchor: %s)\n" % TARGETS.get(a.target, VENUS).provenance)
    for f in a.rf:
        print(compute(f, target=a.target, M=a.M,
                      spread_halfwidth_hz=a.spread_halfwidth_hz).summary())

#!/usr/bin/env python3
"""
eve_moon_libration.py -- adaptive Moon Doppler-spread from lunar libration, per session
and per RECEIVER SET. Replaces the static K1JT table: it computes the actual spread for a
given frequency, time, and observing geometry from JPL ephemerides + lunar orientation.

WHY
  The Moon's echo Doppler spread varies ~50x over a libration cycle AND depends on the
  RECEIVER's location -- the observer's own ~465*cos(lat) m/s ground velocity smears across
  the lunar disk, on top of the Moon's physical libration. So a distant DX receiver can see
  MORE spread than the transmitter and fail to decode. A broadcast signal must therefore be
  sized for the WORST CASE over all possible receivers; a monostatic station (hears its own
  echo) can size to its own geometry and ride libration minima for up to ~10x longer
  coherence. Venus needs none of this (specular, ~2 Hz, observer term sub-Hz).

METHOD (validated against K1JT 2010 "Frequency-Dependent Characteristics of the EME Path"
to ~10% at +/-40 deg latitude):
  limb-to-limb spread   B = 4 * f * R_moon * omega_perp / c
  omega_perp = apparent lunar rotation rate perpendicular to the line of sight = the sweep
  rate of the sub-observer selenographic point. It includes optical + physical libration AND
  the observer's diurnal/parallactic term (the latitude-dependent part), because it is taken
  from the TOPOCENTRIC sub-observer point. Computed by finite-differencing the sub-observer
  lat/lon in the MOON_ME body frame (Skyfield + JPL lunar orientation kernels). NOTE: do NOT
  use Skyfield's frame_latlon_and_rates here -- its rate folds in the Moon's spin; finite-
  differencing the frame_latlon VALUE gives the librational residual.
  90%-power width   w10 = SHAPE_W10 * B    (K1JT's L-band ratio; the ONE approximation here).
  R_bw = w10 ;  coherence = 1 / R_bw.

RECEIVER SETS (rx):
  'self'  + station=(lat,lon)   monostatic: size to YOUR geometry (adaptive; ride minima).
  'all_earth'                   broadcast: worst case over all Moon-visible locations.
                                Depends only on (freq, epoch) -- TX and RX agree with no
                                shared location -- and it also BOUNDS any bistatic pair
                                (bistatic spread = TX term + RX term + libration, each term
                                <= the worst single-location term, so <= monostatic worst).
  [(lat,lon), ...]              coordinated: worst case over the named stations.

DEPENDENCY: skyfield + JPL kernels (de421.bsp, moon_080317.tf, pck00008.tpc,
moon_pa_de421_1900-2050.bpc), ~19 MB, downloaded ONCE from NAIF:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/   (spk/planets, fk/satellites, pck)
Point the kernel directory via the EVE_KERNELS env var, or set_kernel_dir().
"""
import os, math, datetime as _dt
import numpy as np

R_MOON_M  = 1737.4e3
C_M_S     = 299792458.0
SHAPE_W10 = 0.27                    # w10 / limb-to-limb (K1JT L-band). The one approximation.
_kernel_dir = os.environ.get("EVE_KERNELS", ".")

_frame = _ts = _earth = _moon = _wgs84 = None

def set_kernel_dir(path):
    global _kernel_dir, _frame
    _kernel_dir = path; _frame = None     # force reload

def _load():
    global _frame, _ts, _earth, _moon, _wgs84
    if _frame is not None:
        return
    from skyfield.api import Loader, PlanetaryConstants, wgs84
    L = Loader(_kernel_dir, verbose=False)
    _ts = L.timescale(); eph = L('de421.bsp')
    _earth, _moon = eph['earth'], eph['moon']
    pc = PlanetaryConstants()
    pc.read_text(L('moon_080317.tf')); pc.read_text(L('pck00008.tpc'))
    pc.read_binary(L('moon_pa_de421_1900-2050.bpc'))
    _frame = pc.build_frame_named('MOON_ME_DE421'); _wgs84 = wgs84

def _time(when):
    """Accept a datetime (assumed UTC if naive), an ISO-8601 string, or a Skyfield Time."""
    _load()
    if hasattr(when, "tt"):                         # already a Skyfield Time
        return when
    if isinstance(when, str):
        when = _dt.datetime.fromisoformat(when.replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    return _ts.from_datetime(when)

def _limb_at(freq_hz, t, lat, lon, dt=120.0):
    """limb-to-limb spread (Hz) and Moon altitude (deg) at one station, one Skyfield Time."""
    st = _earth + _wgs84.latlon(lat, lon, elevation_m=0)
    def ll(tt):
        p = st.at(tt).observe(_moon).apparent()
        a, o, _ = p.frame_latlon(_frame)
        return a.degrees, o.degrees, p.altaz()[0].degrees
    a0, o0, alt = ll(t)
    a1, o1, _   = ll(_ts.tt_jd(t.tt + dt/86400.0))
    dlo = (((o1 - o0 + 180) % 360) - 180) * (math.pi/180) / dt      # rad/s (unwrapped)
    dla = (a1 - a0) * (math.pi/180) / dt
    w = math.hypot(dlo*math.cos(math.radians(a0)), dla)             # rad/s perp to LOS
    return 4.0*freq_hz*R_MOON_M*w/C_M_S, alt

# ------------------------------------------------------------------ public API
def limb_spread_hz(freq_hz, when, lat, lon):
    """limb-to-limb Doppler spread (Hz) for one station/time."""
    B, _ = _limb_at(freq_hz, _time(when), lat, lon)
    return B

def moon_spread(freq_hz, when, rx="all_earth", station=None, grid_step_deg=15, min_alt_deg=3.0):
    """R_bw-relevant Moon Doppler spread for a receiver set. Returns a dict.
      rx='self'      -> station=(lat,lon) required; size to your own geometry.
      rx='all_earth' -> worst case over all Moon-visible locations (broadcast; bounds bistatic).
      rx=[(lat,lon)] -> worst case over the named stations.
    Keys: limb_hz, w10_hz (== R_bw), R_bw_hz, coherence_s, rx, where, provenance."""
    t = _time(when)
    if rx == "self":
        if station is None:
            raise ValueError("rx='self' requires station=(lat, lon)")
        B, alt = _limb_at(freq_hz, t, station[0], station[1])
        where = (station[0], station[1], alt); prov = "libration @ own station (monostatic)"
    elif rx == "all_earth":
        best = 0.0; where = None
        for lat in range(-80, 81, grid_step_deg):
            for lon in range(-180, 180, grid_step_deg):
                B, alt = _limb_at(freq_hz, t, lat, lon)
                if alt > min_alt_deg and B > best:
                    best = B; where = (lat, lon, alt)
        B = best; prov = "worst case over Earth (broadcast; bounds bistatic)"
        if where is None:
            raise RuntimeError("no Moon-visible grid point -- widen grid / check epoch")
    else:                                            # explicit list of (lat, lon)
        best = 0.0; where = None
        for lat, lon in rx:
            b, alt = _limb_at(freq_hz, t, lat, lon)
            if b > best: best = b; where = (lat, lon, alt)
        B = best; prov = "worst case over %d named stations (coordinated)" % len(rx)
    w10 = SHAPE_W10 * B
    return dict(limb_hz=B, w10_hz=w10, R_bw_hz=w10, coherence_s=(1.0/w10 if w10 > 0 else float("inf")),
                rx=rx, where=where,
                provenance="Moon libration (Skyfield/JPL, K1JT-validated); w10=%.2f*limb; %s" % (SHAPE_W10, prov))

def libration_forecast(freq_hz, start, hours, station, step_min=10, rx="self"):
    """Scan a window for scheduling. Returns list of (datetime_utc, R_bw_hz, coherence_s, alt).
    'self' uses the given station (fast); 'all_earth' recomputes the global worst per step (slow)."""
    _load()
    out = []; n = int(hours*60/step_min)
    t0 = _time(start)
    for i in range(n):
        t = _ts.tt_jd(t0.tt + i*step_min/1440.0)
        if rx == "self":
            B, alt = _limb_at(freq_hz, t, station[0], station[1])
            w10 = SHAPE_W10*B
        else:
            r = moon_spread(freq_hz, t, rx=rx); w10 = r["w10_hz"]; alt = r["where"][2]
        out.append((t.utc_datetime(), w10, (1.0/w10 if w10 > 0 else float("inf")), alt))
    return out

# ------------------------------------------------------------------ self-test / CLI
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Adaptive Moon libration Doppler spread.")
    ap.add_argument("--rf", type=float, default=2304e6)
    ap.add_argument("--when", default="2026-10-07T08:00:00Z")
    ap.add_argument("--lat", type=float, default=33.0); ap.add_argument("--lon", type=float, default=-117.25)
    ap.add_argument("--rx", default="all_earth", help="self | all_earth")
    ap.add_argument("--validate", action="store_true", help="reproduce K1JT 40N @1296")
    a = ap.parse_args()
    if a.validate:
        print("VALIDATION: 40N @1296 over 29 days vs K1JT (limb median~19, max~37; w10~10)")
        fc = libration_forecast(1296e6, "2026-10-01T00:00:00Z", 29*24, (40.0, 0.0), step_min=20)
        limb = np.array([w/SHAPE_W10 for (_,w,_,alt) in fc if alt > 0])
        print("  computed limb: median %.1f  max %.1f Hz   [K1JT 19 / 37]" % (np.median(limb), limb.max()))
    st = (a.lat, a.lon) if a.rx == "self" else None
    r = moon_spread(a.rf, a.when, rx=a.rx, station=st)
    print("\n%s @ %.4f MHz, rx=%s" % (a.when, a.rf/1e6, a.rx))
    print("  limb=%.1f Hz  R_bw(w10)=%.2f Hz  coherence=%.3f s  where(lat,lon,alt)=%s"
          % (r["limb_hz"], r["R_bw_hz"], r["coherence_s"], r["where"]))

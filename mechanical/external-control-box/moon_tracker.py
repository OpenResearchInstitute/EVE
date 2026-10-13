#!/usr/bin/env python3
"""
moon_tracker.py — Hello Giggy Moon tracking (ORI EVE)

Domain layers:
  MoonEphemeris  — where is the Moon (PyEphem, topocentric az/el)
  PointingPolicy — should we move (horizon mask, deadband, limits)
  Rotator        — how we move (rotctld text protocol, localhost:4533)

Chain: this script -> rotctld -> GS-232B serial -> K3NG Mega -> relays -> G-5500DC

Usage:
  python3 moon_tracker.py --dry-run       # compute + print, command nothing
  python3 moon_tracker.py                 # track
  ctrl-C exits cleanly (leaves rotator where it is)
"""

import argparse
import math
import socket
import sys
import time
from datetime import datetime, timezone

import ephem  # pip install ephem

# ---------------- Station configuration ----------------
# VERIFY: set to Hello Giggy's actual coordinates. Sign convention:
# west longitude is NEGATIVE. Decimal degrees.
STATION_LAT = 32.9  # SET_ME
STATION_LON = -117.2  # SET_ME
STATION_ELEV_M = 100  # SET_ME (meters ASL; small effect, get close)

# ---------------- Pointing policy ----------------
MIN_ELEVATION_DEG = 0.0   # don't command below the horizon
DEADBAND_DEG = 1.0        # don't command moves smaller than this
                          # (G-5500 honesty ~1 deg; relay life is finite)
UPDATE_PERIOD_S = 10      # recompute cadence; Moon moves ~0.25 deg/min,
                          # so commands actually fire every few minutes
AZ_MIN, AZ_MAX = 0.0, 450.0   # north-at-CCW-stop convention (\I0)
EL_MIN, EL_MAX = 0.0, 180.0

ROTCTLD_HOST = "127.0.0.1"
ROTCTLD_PORT = 4533


class MoonEphemeris:
    """Where the Moon is, from this station, right now."""

    def __init__(self, lat, lon, elev_m):
        self.obs = ephem.Observer()
        self.obs.lat = str(lat)
        self.obs.lon = str(lon)
        self.obs.elevation = elev_m
        self.moon = ephem.Moon()

    def azel_now(self):
        self.obs.date = ephem.now()  # UTC from system clock
        self.moon.compute(self.obs)
        return (math.degrees(self.moon.az), math.degrees(self.moon.alt))


class Rotator:
    """rotctld text protocol client. One command per connection keeps
    state simple and survives rotctld restarts."""

    def __init__(self, host, port, dry_run=False):
        self.host, self.port, self.dry_run = host, port, dry_run

    def _txn(self, line):
        with socket.create_connection((self.host, self.port), timeout=5) as s:
            s.sendall((line + "\n").encode())
            return s.recv(256).decode()

    def get_position(self):
        reply = self._txn("p").split()
        return (float(reply[0]), float(reply[1]))

    def set_position(self, az, el):
        if self.dry_run:
            return "DRY"
        reply = self._txn(f"P {az:.1f} {el:.1f}")
        if "RPRT 0" not in reply and reply.strip() != "":
            print(f"  ! rotctld replied: {reply.strip()}")
        return reply


class SerialRotator:
    """Direct GS-232B over the serial port — no rotctld. Opens the port
    ONCE per session (eating the one DTR reboot), then immediately sets
    the K3NG software clock from this machine's UTC, and holds the port
    so the clock survives for the whole session. The tracker is the
    clock source: every session begins synchronized."""

    def __init__(self, port, dry_run=False, baud=9600):
        import serial  # pip install pyserial
        self.dry_run = dry_run
        self.ser = serial.Serial(port, baud, timeout=2)
        print(f"Serial open on {port}; waiting out the DTR reboot...")
        time.sleep(4)  # K3NG boot after auto-reset
        self.ser.reset_input_buffer()
        self.sync_clock()

    def _txn(self, line):
        self.ser.reset_input_buffer()
        self.ser.write((line + "\r").encode())
        time.sleep(0.3)
        return self.ser.read(self.ser.in_waiting or 1).decode(errors="replace")

    def sync_clock(self):
        utc = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        reply = self._txn("\\O" + utc)
        print(f"K3NG clock sync -> {utc} UTC  ({reply.strip() or 'no echo'})")
        check = self._txn("\\C")
        if check.strip():
            print(f"K3NG clock readback: {check.strip()}")

    def get_position(self):
        reply = self._txn("C2")
        # GS-232B: 'AZ=aaaEL=eee' (formats vary slightly)
        digits = [int(s) for s in
                  reply.replace("AZ=", " ").replace("EL=", " ").split()
                  if s.lstrip("+-").isdigit()]
        if len(digits) >= 2:
            return (float(digits[0]), float(digits[1]))
        raise ValueError(f"unparseable C2 reply: {reply!r}")

    def set_position(self, az, el):
        if self.dry_run:
            return "DRY"
        return self._txn(f"W{az:03.0f} {el:03.0f}")


class PointingPolicy:
    """Decides whether a computed target becomes a command."""

    def __init__(self):
        self.last_commanded = None  # (az, el) or None

    def clamp(self, az, el):
        az = max(AZ_MIN, min(AZ_MAX, az))
        el = max(EL_MIN, min(EL_MAX, el))
        return az, el

    def should_move(self, az, el, el_raw):
        if el_raw < MIN_ELEVATION_DEG:
            return False, "below horizon"
        if self.last_commanded is None:
            return True, "initial acquisition"
        d_az = abs(az - self.last_commanded[0])
        d_el = abs(el - self.last_commanded[1])
        if max(d_az, d_el) < DEADBAND_DEG:
            return False, "within deadband"
        return True, f"delta az={d_az:.2f} el={d_el:.2f}"


def main():
    ap = argparse.ArgumentParser(description="Hello Giggy Moon tracker")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print, command nothing")
    ap.add_argument("--port", default=None, metavar="/dev/cu.usbmodemXXXX",
                    help="talk GS-232B directly on this serial port "
                         "(syncs the K3NG clock at startup); omit to use "
                         "rotctld at localhost:4533")
    args = ap.parse_args()

    eph = MoonEphemeris(STATION_LAT, STATION_LON, STATION_ELEV_M)
    if args.port:
        rot = SerialRotator(args.port, dry_run=args.dry_run)
    else:
        rot = Rotator(ROTCTLD_HOST, ROTCTLD_PORT, dry_run=args.dry_run)
    policy = PointingPolicy()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"Moon tracker [{mode}] station=({STATION_LAT},{STATION_LON}) "
          f"deadband={DEADBAND_DEG} min_el={MIN_ELEVATION_DEG}")

    try:
        while True:
            az, el = eph.azel_now()
            az_c, el_c = policy.clamp(az, el)
            move, why = policy.should_move(az_c, el_c, el)
            stamp = time.strftime("%H:%M:%S")
            if move:
                rot.set_position(az_c, el_c)
                policy.last_commanded = (az_c, el_c)
                print(f"{stamp}  moon az={az:6.1f} el={el:5.1f}  -> "
                      f"COMMAND P {az_c:.1f} {el_c:.1f}  ({why})")
            else:
                print(f"{stamp}  moon az={az:6.1f} el={el:5.1f}  -- hold ({why})")
            time.sleep(UPDATE_PERIOD_S)
    except KeyboardInterrupt:
        print("\nTracker stopped. Rotator holds last position.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

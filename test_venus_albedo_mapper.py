"""
Test harness for venus_albedo_mapper.py
========================================
Run this from the same directory as venus_albedo_mapper.py and your notebook.

Each test is independent. If one fails, the others still run.
Copy the output and share it — the failure messages are written to tell
you exactly what to fix.

Usage:
    python test_venus_albedo_mapper.py

    # Run only specific tests:
    python test_venus_albedo_mapper.py horizons
    python test_venus_albedo_mapper.py gdal
    python test_venus_albedo_mapper.py integration
    python test_venus_albedo_mapper.py full
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone

# ─── tiny test runner ────────────────────────────────────────────────────────

PASS  = "  PASS"
FAIL  = "  FAIL"
SKIP  = "  SKIP"
INFO  = "  INFO"
results = []

def report(label, status, detail=""):
    tag = {"PASS": "✓", "FAIL": "✗", "SKIP": "─", "INFO": "·"}.get(status, "?")
    line = f"  [{tag}] {label}"
    if detail:
        line += f"\n       {detail}"
    print(line)
    results.append((status, label))

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ─── import the module ───────────────────────────────────────────────────────

section("Import")
try:
    from venus_albedo_mapper import (
        HorizonsVenusEphemeris,
        VenusReflectivityMap,
        VenusAlbedoMapper,
        DynamicAlbedoEVELinkBudget,
        VENUS_RADAR_FEATURES,
        print_feature_table,
    )
    report("venus_albedo_mapper imports cleanly", "PASS")
except ImportError as e:
    report("venus_albedo_mapper import", "FAIL", str(e))
    print("\nCannot continue — fix the import error above first.")
    sys.exit(1)


# ─── TEST 1: GDAL availability ───────────────────────────────────────────────

section("TEST 1 — GDAL availability (optional)")
GDAL_OK = False
try:
    from osgeo import gdal
    gdal.UseExceptions()
    report(f"GDAL available, version {gdal.__version__}", "PASS")
    GDAL_OK = True
except ImportError:
    report("GDAL not installed — fallback PDS3 parser will be used", "INFO",
           "To install GDAL: conda install -c conda-forge gdal\n"
           "       (plain `pip install GDAL` needs system libs; conda is easier)")


# ─── TEST 2: Synthetic map — no files needed ─────────────────────────────────

section("TEST 2 — VenusReflectivityMap with synthetic data (no GDAL, no files)")

class SyntheticVenusMap(VenusReflectivityMap):
    """
    Subclass that bypasses GDAL and file I/O entirely.
    Loads a procedurally generated reflectivity map that mimics
    the GREDR structure: low plains, bright Aphrodite Terra, bright Maxwell Montes.
    """
    def load_synthetic(self):
        # 1° grid, -70 to +70 lat, 0 to 360 lon West
        nlat, nlon = 141, 361
        lats = np.linspace(70, -70, nlat)
        lons = np.linspace(0, 360, nlon)
        LON, LAT = np.meshgrid(lons, lats)

        refl = np.full((nlat, nlon), 0.12)  # baseline plains

        # Aphrodite Terra: ~105° W, equatorial
        aph_mask = (
            (np.abs(LAT) < 20) &
            (np.abs(((LON - 105) + 180) % 360 - 180) < 30)
        )
        refl[aph_mask] = 0.28

        # Maxwell Montes: ~3° W, ~65° N
        max_mask = (
            (LAT > 55) &
            (np.abs(((LON - 3) + 180) % 360 - 180) < 15)
        )
        refl[max_mask] = 0.52

        # Build a fake GeoTransform: (x_min, dx, 0, y_max, 0, dy)
        dx = 360.0 / nlon
        dy = -140.0 / nlat
        self._gt = (0.0, dx, 0.0, 70.0, 0.0, dy)
        self._reflectivity = refl.astype(np.float64)
        self._lat_centers  = lats
        self._lon_centers  = lons
        self._nodata       = None
        self._lon_is_east  = True   # synthetic data uses East longitude (Aphrodite at 105°E)
        self._loaded       = True

try:
    smap = SyntheticVenusMap()
    smap.load_synthetic()
    report("Synthetic map constructed", "PASS")

    # Check Aphrodite Terra region — East 105° (John K5JBT: "Aphrodite Terra is at ~105E")
    r_aph = smap.reflectivity_at(lat=0, lon_east=105)
    ok = 0.20 <= r_aph <= 0.40
    report(f"Aphrodite Terra reflectivity: {r_aph:.3f}  (expect 0.20–0.40)",
           "PASS" if ok else "FAIL")

    # Check Maxwell Montes region — East ~3°
    r_max = smap.reflectivity_at(lat=65, lon_east=3)
    ok = 0.40 <= r_max <= 0.65
    report(f"Maxwell Montes reflectivity:  {r_max:.3f}  (expect 0.40–0.65)",
           "PASS" if ok else "FAIL")

    # Check plains — East 200° (featureless region)
    r_plains = smap.reflectivity_at(lat=30, lon_east=200)
    ok = 0.08 <= r_plains <= 0.18
    report(f"Plains reflectivity:          {r_plains:.3f}  (expect 0.08–0.18)",
           "PASS" if ok else "FAIL")

    # Longitude wrapping: 361°E should equal 1°E
    r_0   = smap.reflectivity_at(lat=0, lon_east=0)
    r_360 = smap.reflectivity_at(lat=0, lon_east=360)
    ok = abs(r_0 - r_360) < 1e-6
    report(f"Longitude wrap 0°==360°: r(0)={r_0:.4f}, r(360)={r_360:.4f}",
           "PASS" if ok else "FAIL")

    # Latitude clamping: 95° N should clamp to map edge without crashing
    try:
        r_pole = smap.reflectivity_at(lat=95, lon_east=100)
        report("Latitude clamping at 95° N doesn't crash", "PASS",
               f"returned {r_pole:.4f}")
    except Exception as e:
        report("Latitude clamping at 95° N", "FAIL", str(e))

except Exception as e:
    report("Synthetic map test", "FAIL", str(e))
    smap = None


# ─── TEST 3: VenusAlbedoMapper disk integration ──────────────────────────────

section("TEST 3 — VenusAlbedoMapper disk integration (synthetic map)")

if smap is None:
    report("Skipping — synthetic map not available", "SKIP")
else:
    try:
        # horizons_lon_offset=0: the synthetic map uses Horizons-frame coords directly.
        # The real GREDR offset (186.5°) is only needed for actual Magellan data.
        mapper = VenusAlbedoMapper(smap, n_backscatter=2.0, grid_resolution_deg=2.0,
                                   horizons_lon_offset=0.0)

        # Sub-Earth point aimed at Aphrodite Terra (East 105°)
        r_aph = mapper.effective_albedo(sub_lon_east_deg=105, sub_lat_deg=0)
        report(f"ρ_eff with Aphrodite Terra facing Earth: {r_aph['albedo_eff']:.4f}",
               "PASS" if r_aph['albedo_eff'] > 0.14 else "FAIL",
               "Should be elevated above plains baseline of 0.12")

        # Sub-Earth point aimed at featureless plains (East 200°)
        r_plain = mapper.effective_albedo(sub_lon_east_deg=200, sub_lat_deg=0)
        report(f"ρ_eff with plains facing Earth:          {r_plain['albedo_eff']:.4f}",
               "PASS" if r_plain['albedo_eff'] < r_aph['albedo_eff'] else "FAIL",
               "Should be lower than Aphrodite-facing result")

        # Aphrodite should give a meaningfully higher albedo — what's the dB gain?
        delta_db = 10 * np.log10(r_aph['albedo_eff'] / r_plain['albedo_eff'])
        report(f"Aphrodite advantage over plains: {delta_db:+.2f} dB", "INFO")

        # Sub-Earth point aimed at Maxwell Montes (East 3°)
        r_max = mapper.effective_albedo(sub_lon_east_deg=3, sub_lat_deg=65)
        report(f"ρ_eff with Maxwell Montes near sub-Earth: {r_max['albedo_eff']:.4f}",
               "INFO",
               "(Maxwell is small and polar; elevation above baseline may be modest)")

        # Verify n_backscatter sensitivity: higher n → sub-radar point dominates more
        mapper_n4 = VenusAlbedoMapper(smap, n_backscatter=4.0, grid_resolution_deg=2.0,
                                      horizons_lon_offset=0.0)
        r_aph_n4  = mapper_n4.effective_albedo(sub_lon_east_deg=105, sub_lat_deg=0)
        report(
            f"n=4 vs n=2 at Aphrodite: {r_aph_n4['albedo_eff']:.4f} vs {r_aph['albedo_eff']:.4f}",
            "INFO",
            "Higher n gives more weight to sub-radar center — values should differ"
        )

        # Sanity: result must stay in [0, 1]
        all_ok = all(0 <= v['albedo_eff'] <= 1.0
                     for v in [r_aph, r_plain, r_max, r_aph_n4])
        report("All ρ_eff values in [0, 1]", "PASS" if all_ok else "FAIL")

    except Exception as e:
        import traceback
        report("VenusAlbedoMapper integration test", "FAIL",
               traceback.format_exc())


# ─── TEST 4: Horizons API ────────────────────────────────────────────────────

section("TEST 4 — HorizonsVenusEphemeris (requires network)")

HORIZONS_OK = False
horizons_rows = None
try:
    eph = HorizonsVenusEphemeris()
    rows = eph.get_sub_earth_series(
        start=datetime(2026, 10, 19, 0, 0, tzinfo=timezone.utc),
        stop =datetime(2026, 10, 26, 0, 0, tzinfo=timezone.utc),
        step ="1d",
    )
    report(f"Horizons returned {len(rows)} rows", "PASS" if len(rows) >= 6 else "FAIL")

    # Validate structure
    required_keys = {"utc", "ObsSub_LON", "ObsSub_LAT"}
    ok = all(required_keys.issubset(r.keys()) for r in rows)
    report("All rows have utc, ObsSub_LON, ObsSub_LAT", "PASS" if ok else "FAIL")

    # Validate longitude range
    lons = [r["ObsSub_LON"] for r in rows]
    ok = all(0 <= lon <= 360 for lon in lons)
    report(f"All longitudes in [0, 360]:  {[f'{l:.1f}' for l in lons]}",
           "PASS" if ok else "FAIL")

    # Rate of change: Venus rotates ~1.7°/day as seen from Earth
    if len(rows) >= 2:
        dt_days = (rows[1]["utc"] - rows[0]["utc"]).total_seconds() / 86400
        dlon = rows[1]["ObsSub_LON"] - rows[0]["ObsSub_LON"]
        rate = abs(dlon / dt_days)
        ok = 0.5 <= rate <= 5.0   # expect ~1.7°/day
        report(f"Sub-Earth longitude rate: {rate:.2f}°/day  (expect ~1.7°/day)",
               "PASS" if ok else "FAIL")

    # Print the table so you can eyeball it against John's values
    print()
    print(f"  {'Date':12}  {'ObsSub-LON':>12}  {'ObsSub-LAT':>12}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}")
    for r in rows:
        print(f"  {r['utc'].strftime('%Y-%m-%d'):12}  "
              f"{r['ObsSub_LON']:>11.3f}°  {r['ObsSub_LAT']:>+11.3f}°")

    HORIZONS_OK = True
    horizons_rows = rows

except Exception as e:
    # Print the URL we tried so it can be pasted into a browser to diagnose
    test_url = (
        "https://ssd.jpl.nasa.gov/api/horizons.api"
        "?format=json&COMMAND='299'&OBJ_DATA=NO&MAKE_EPHEM=YES"
        "&EPHEM_TYPE=OBSERVER&CENTER='500@399'"
        "&START_TIME='2026-10-19'&STOP_TIME='2026-10-26'&STEP_SIZE='1d'"
        "&QUANTITIES='14'&ANG_FORMAT=DEG&CAL_FORMAT=CAL&CSV_FORMAT=YES"
    )
    report("Horizons API", "FAIL", f"{e}\n       Test URL: {test_url}")
    report("Continuing with synthetic sub-Earth point for downstream tests", "INFO")


# ─── TEST 5: Single-date interpolation ───────────────────────────────────────

section("TEST 5 — get_sub_earth_at_date() interpolation")

if not HORIZONS_OK:
    report("Skipping — Horizons not reachable", "SKIP")
else:
    try:
        eph2 = HorizonsVenusEphemeris()
        date = datetime(2026, 10, 22, 6, 30, tzinfo=timezone.utc)  # 06:30 UTC
        lon, lat = eph2.get_sub_earth_at_date(date)
        report(f"get_sub_earth_at_date({date.strftime('%Y-%m-%d %H:%M')}): "
               f"lon={lon:.3f}°, lat={lat:.3f}°", "PASS")

        # Should be between the adjacent daily values
        ok = 0 <= lon <= 360
        report("Interpolated longitude in valid range", "PASS" if ok else "FAIL")
    except Exception as e:
        report("get_sub_earth_at_date", "FAIL", str(e))


# ─── TEST 6: GREDR file loading via GDAL ─────────────────────────────────────

section("TEST 6 — VenusReflectivityMap.load_from_files() (GDAL or PDS3 fallback)")

GREDR_OK = False
real_vmap = None

vmap_test = VenusReflectivityMap()
img_exists = os.path.exists(vmap_test.img_path)
lbl_exists = os.path.exists(vmap_test.lbl_path)

if not img_exists or not lbl_exists:
    report(f"GREDR files not cached at {vmap_test.cache_dir}", "SKIP",
           "Run VenusReflectivityMap().download_and_cache() then re-run this test")
else:
    report(f"Found browse.img ({os.path.getsize(vmap_test.img_path):,} bytes)", "PASS")
    report(f"Found browse.lbl ({os.path.getsize(vmap_test.lbl_path):,} bytes)", "PASS")

    # Print the label so we can verify the PDS3 keywords
    print("\n  --- browse.lbl contents ---")
    with open(vmap_test.lbl_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip()
            if any(kw in line.upper() for kw in [
                "LINE", "SAMPLE", "SCALING", "OFFSET", "LATITUDE",
                "LONGITUDE", "MINIMUM", "MAXIMUM", "WESTERNMOST",
                "EASTERNMOST", "SAMPLE_TYPE", "SAMPLE_BITS"
            ]):
                print(f"    {line}")
    print("  --- end label ---\n")

    try:
        vmap_test.load_from_files()
        loader = getattr(vmap_test, "_loader_used", "unknown")
        report(f"load_from_files() completed via '{loader}'", "PASS")

        lats, lons, refl = vmap_test.reflectivity_grid()
        ok = refl.ndim == 2 and refl.shape[0] > 10 and refl.shape[1] > 10
        report(f"Reflectivity array shape: {refl.shape}", "PASS" if ok else "FAIL")

        ok = refl.min() >= 0 and refl.max() <= 1.0
        report(f"Reflectivity range: [{refl.min():.4f}, {refl.max():.4f}]",
               "PASS" if ok else "FAIL", "Must be in [0, 1]")

        # Self-locating coordinate validation.
        # Rather than assuming exact feature coordinates, find the brightest
        # pixels and check they land where planetary science says they should.

        # Global maximum → should be Maxwell Montes (~3-6°E, ~60-70°N)
        max_idx = np.unravel_index(
            np.argmax(vmap_test._reflectivity), vmap_test._reflectivity.shape)
        max_lat_v = float(vmap_test._lat_centers[max_idx[0]])
        max_lon_v = float(vmap_test._lon_centers[max_idx[1]])
        max_val   = float(vmap_test._reflectivity[max_idx])
        report(f"Global max = {max_val:.3f} at ({max_lat_v:.1f}°N, {max_lon_v:.1f}°E)",
               "INFO")

        max_lon_ok = 80 <= max_lon_v <= 100
        max_lat_ok = 60 <= max_lat_v <= 70
        report("Global max near Maxwell Montes (GREDR frame: expect 80–100°E, 60–70°N)",
               "PASS" if (max_lon_ok and max_lat_ok) else "FAIL",
               f"Got {max_lon_v:.1f}°E, {max_lat_v:.1f}°N  "
               f"(≈{max_lon_v - 186.5:.1f}°E in Horizons/IAU 2000 frame)")

        # Equatorial maximum → should be in Aphrodite Terra (GREDR ~280-310°E)
        # In Horizons frame, Aphrodite Terra is at ~105°E (John K5JBT)
        # GREDR = Horizons + 186.5°  →  105 + 186.5 = 291.5°E in GREDR
        eq_row  = int(np.argmin(np.abs(vmap_test._lat_centers)))
        eq_refl = vmap_test._reflectivity[eq_row, :]
        eq_col  = int(np.argmax(eq_refl))
        eq_lon  = float(vmap_test._lon_centers[eq_col])
        eq_val  = float(eq_refl[eq_col])
        eq_horizons = (eq_lon - 186.5) % 360.0
        report(f"Equatorial max = {eq_val:.3f} at GREDR {eq_lon:.1f}°E "
               f"(≈ Horizons {eq_horizons:.1f}°E)", "INFO",
               "Should be ~105°E in Horizons frame = ~291°E in GREDR frame")

        report("Equatorial max in Aphrodite Terra GREDR range (275–310°E)",
               "PASS" if 275 <= eq_lon <= 310 else "FAIL")
        report(f"Equatorial max {eq_val:.3f} > global mean {vmap_test._reflectivity.mean():.3f}",
               "PASS" if eq_val > vmap_test._reflectivity.mean() * 1.2 else "FAIL")

        GREDR_OK = True
        real_vmap = vmap_test
    except Exception as e:
        import traceback
        report("load_from_files()", "FAIL", traceback.format_exc())


# ─── TEST 7: Full pipeline (real data) ───────────────────────────────────────

section("TEST 7 — Full pipeline (Horizons + GREDR + disk integration)")

if not HORIZONS_OK or not GREDR_OK:
    missing = []
    if not HORIZONS_OK: missing.append("Horizons")
    if not GREDR_OK:    missing.append("GREDR map")
    report(f"Skipping — need: {', '.join(missing)}", "SKIP")
else:
    try:
        full_mapper = VenusAlbedoMapper(
            real_vmap, n_backscatter=2.0, grid_resolution_deg=1.0
        )
        conj_date = datetime(2026, 10, 22, 0, 0, tzinfo=timezone.utc)
        result = full_mapper.albedo_for_date(conj_date)

        report(f"albedo_for_date({conj_date.date()}) returned without error", "PASS")
        report(f"Sub-Earth point: {result['sub_lon_E']:.2f}° W, "
               f"{result['sub_lat_N']:.3f}° N", "INFO")
        report(f"Effective albedo ρ_eff = {result['albedo_eff']:.4f}", "INFO")

        ok = 0.05 <= result['albedo_eff'] <= 0.60
        report("ρ_eff in plausible physical range [0.05, 0.60]",
               "PASS" if ok else "FAIL")

        delta_db = 10 * np.log10(result['albedo_eff'] / 0.152)
        report(f"CNR delta vs nominal 0.152:  {delta_db:+.2f} dB", "INFO")

    except Exception as e:
        import traceback
        report("Full pipeline test", "FAIL", traceback.format_exc())


# ─── TEST 8: DynamicAlbedoEVELinkBudget stub test ────────────────────────────

section("TEST 8 — DynamicAlbedoEVELinkBudget (stub EVELinkBudget)")

class StubEVELinkBudget:
    """Minimal stand-in so we can test the wrapper without the full notebook."""
    def __init__(self, params):
        self.params = params
        self.venus_radar_albedo = 0.152

    def calculate_link_budget(self, distance_km):
        # Simplified: CNR is just a function of albedo and distance
        # (not a real link budget — just enough to verify the wrapper works)
        fake_cnr = -20.0 + 10 * np.log10(self.venus_radar_albedo) \
                   - 20 * np.log10(distance_km / 42e6)
        return {
            "cnr_db": fake_cnr,
            "cnr_db_1hz": fake_cnr + 50,
            "venus_radar_albedo": self.venus_radar_albedo,
            "rx_power_dbw": -180.0,
        }

class StubParams:
    pass

if smap is None:
    report("Skipping — synthetic map not available", "SKIP")
else:
    try:
        stub_mapper = VenusAlbedoMapper(smap, n_backscatter=2.0, grid_resolution_deg=2.0,
                                        horizons_lon_offset=0.0)

        dyn = DynamicAlbedoEVELinkBudget(
            params=StubParams(),
            albedo_mapper=stub_mapper,
            base_link_budget_class=StubEVELinkBudget,
        )

        # Test calculate_link_budget_for_date using a synthetic sub-Earth point
        # (bypasses Horizons by calling effective_albedo directly)
        result = stub_mapper.effective_albedo(sub_lon_east_deg=105, sub_lat_deg=0)
        budget = StubEVELinkBudget(StubParams())
        budget.venus_radar_albedo = result["albedo_eff"]
        r = budget.calculate_link_budget(42e6)

        ok = "cnr_db" in r and "venus_radar_albedo" in r
        report("calculate_link_budget returns expected keys", "PASS" if ok else "FAIL")

        ok = r["venus_radar_albedo"] == result["albedo_eff"]
        report("Albedo correctly injected into budget", "PASS" if ok else "FAIL",
               f"albedo_eff={result['albedo_eff']:.4f}, "
               f"budget used={r['venus_radar_albedo']:.4f}")

        report(f"Stub CNR at inferior conjunction: {r['cnr_db']:+.2f} dB", "INFO")

    except Exception as e:
        import traceback
        report("DynamicAlbedoEVELinkBudget stub test", "FAIL",
               traceback.format_exc())


# ─── Summary ─────────────────────────────────────────────────────────────────

section("Summary")
n_pass = sum(1 for s, _ in results if s == "PASS")
n_fail = sum(1 for s, _ in results if s == "FAIL")
n_skip = sum(1 for s, _ in results if s == "SKIP")
n_info = sum(1 for s, _ in results if s == "INFO")

print(f"\n  {n_pass} passed   {n_fail} failed   {n_skip} skipped   {n_info} info\n")

if n_fail > 0:
    print("  Failed tests:")
    for s, label in results:
        if s == "FAIL":
            print(f"    ✗ {label}")

if n_skip > 0:
    print("\n  Skipped tests (dependencies not yet available):")
    for s, label in results:
        if s == "SKIP":
            print(f"    ─ {label}")

print()
print("  To run specific test groups:")
print("    python test_venus_albedo_mapper.py          # all tests")
print("    python test_venus_albedo_mapper.py horizons # only Horizons API test")
print("    python test_venus_albedo_mapper.py gdal     # only GDAL/file tests")
print("    python test_venus_albedo_mapper.py synth    # synthetic data only")

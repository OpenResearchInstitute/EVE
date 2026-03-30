"""
Venus Spatially-Resolved Radar Albedo Module
=============================================
Replaces the fixed `venus_radar_albedo` scalar in EVELinkBudget with a
physically accurate, date-dependent value derived from:

  1. JPL Horizons API  →  Venus sub-Earth longitude / latitude for any date
  2. Magellan GREDR    →  Global radar reflectivity map (PDS archive, Mercator)
  3. Disk integration  →  Weighted backscatter integral over the visible hemisphere

Data source for the reflectivity map:
  https://pds-geosciences.wustl.edu/mgn/mgn-v-gxdr-v1/mg_3002/gredr/merc/

Map loading uses GDAL (pip install GDAL), which reads PDS3 .img/.lbl files
natively and handles all projection/georeferencing metadata automatically.
Run `VenusReflectivityMap.download_and_cache()` once to save the files locally.
Afterward everything works offline.

Install:
    pip install GDAL

Authors:  ORI EVE Team
License:  MIT
"""

import numpy as np
import json
import re
import os
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1.  JPL Horizons ephemeris fetch
# ---------------------------------------------------------------------------

class HorizonsVenusEphemeris:
    """
    Queries the JPL Horizons API for Venus observer sub-longitude and
    sub-latitude (quantity 14), which tell you exactly which face of
    Venus is aimed at Earth at any moment.

    The ObsSub-LON column from Horizons is what John K5JBT identified as
    the right number to correlate against the Magellan reflectivity maps.

    Venus uses a West-longitude, IAU-right-hand coordinate system.
    Horizons ObsSub-LON is in West longitude (0-360°).
    The Magellan GREDR map is also in West longitude.
    No conversion needed — they speak the same language.
    """

    BASE_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

    def get_sub_earth_series(
        self,
        start: datetime,
        stop: datetime,
        step: str = "1d",
    ) -> list[dict]:
        """
        Returns a list of dicts:
          [{'utc': datetime, 'ObsSub_LON': float, 'ObsSub_LAT': float}, ...]

        Parameters
        ----------
        start, stop : datetime  (UTC; naive datetimes are treated as UTC)
        step        : Horizons step string e.g. '1d', '6h', '1h'
        """
        def _fmt(dt: datetime) -> str:
            # Horizons API requires single quotes around date strings with spaces
            return f"'{dt.strftime('%Y-%b-%d %H:%M')}'"

        # Build URL manually — urllib.parse.urlencode encodes @ as %40 which
        # some Horizons API versions reject in the CENTER parameter.
        start_str = start.strftime("%Y-%m-%d")
        stop_str  = stop.strftime("%Y-%m-%d")

        url = (
            f"{self.BASE_URL}"
            f"?format=json"
            f"&COMMAND='299'"
            f"&OBJ_DATA=NO"
            f"&MAKE_EPHEM=YES"
            f"&EPHEM_TYPE=OBSERVER"
            f"&CENTER='500@399'"
            f"&START_TIME='{start_str}'"
            f"&STOP_TIME='{stop_str}'"
            f"&STEP_SIZE='{step}'"
            f"&QUANTITIES='14'"
            f"&ANG_FORMAT=DEG"
            f"&CAL_FORMAT=CAL"
            f"&CSV_FORMAT=YES"
        )

        req = urllib.request.Request(
            url, headers={"User-Agent": "ORI-EVE-LinkBudget/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as urllib_err:
            # Fall back to curl — more reliable with some SSL/proxy configurations
            import subprocess
            try:
                result = subprocess.run(
                    ["curl", "-s", "--max-time", "30", url],
                    capture_output=True, text=True, check=True
                )
                payload = json.loads(result.stdout)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise urllib_err  # re-raise original error if curl also fails

        raw = payload.get("result", "")
        return self._parse_csv_result(raw)

    def get_sub_earth_at_date(self, date: datetime) -> Tuple[float, float]:
        """
        Returns (ObsSub_LON_deg, ObsSub_LAT_deg) for a single date.
        Fetches a 2-day window around the requested date so we always get
        at least one valid row.
        """
        rows = self.get_sub_earth_series(
            start=date - timedelta(hours=12),
            stop=date + timedelta(hours=12),
            step="1h",
        )
        if not rows:
            raise ValueError(f"No Horizons data returned for {date}")

        # Interpolate to the exact requested time
        if len(rows) == 1:
            return rows[0]["ObsSub_LON"], rows[0]["ObsSub_LAT"]

        # Find the bracketing rows
        t0 = rows[0]["utc"]
        for i in range(1, len(rows)):
            if rows[i]["utc"] >= date:
                t_lo, t_hi = rows[i - 1]["utc"], rows[i]["utc"]
                dt_total = (t_hi - t_lo).total_seconds()
                if dt_total == 0:
                    return rows[i]["ObsSub_LON"], rows[i]["ObsSub_LAT"]
                alpha = (date - t_lo).total_seconds() / dt_total
                lon = rows[i - 1]["ObsSub_LON"] + alpha * (
                    rows[i]["ObsSub_LON"] - rows[i - 1]["ObsSub_LON"]
                )
                lat = rows[i - 1]["ObsSub_LAT"] + alpha * (
                    rows[i]["ObsSub_LAT"] - rows[i - 1]["ObsSub_LAT"]
                )
                return lon, lat

        return rows[-1]["ObsSub_LON"], rows[-1]["ObsSub_LAT"]

    @staticmethod
    def _parse_csv_result(text: str) -> list[dict]:
        """
        Parse the Horizons CSV output between $$SOE / $$EOE markers.
        The CSV quantity-14 columns are:
          Date__(UT)__HR:MN  ... ObsSub-LON  ObsSub-LAT  ...
        """
        # Extract header and data block
        soe = text.find("$$SOE")
        eoe = text.find("$$EOE")
        if soe < 0 or eoe < 0:
            raise ValueError(
                "Could not find $$SOE/$$EOE markers in Horizons response. "
                "Check that the API call succeeded:\n" + text[:600]
            )

        header_block = text[:soe]
        data_block   = text[soe + 5: eoe].strip()

        # The CSV header line comes just before $$SOE
        # Find the last non-empty line in header_block
        header_lines = [l.strip() for l in header_block.splitlines() if l.strip()]
        # The last line starting with a date-like pattern is the column header
        col_header = next(
            (l for l in reversed(header_lines) if "ObsSub" in l or "Date" in l),
            None,
        )

        # Fallback: positional parsing based on known Horizons CSV quantity-14 layout
        # Columns (0-indexed after splitting on comma):
        #   0: Date__(UT)__HR:MN
        #   1: ObsSub-LON
        #   2: ObsSub-LAT
        # But column indices vary by query; parse by header name instead.

        if col_header is None:
            raise ValueError("Could not locate column header in Horizons output.")

        cols = [c.strip() for c in col_header.split(",")]

        def _col_idx(candidates):
            for name in candidates:
                for i, c in enumerate(cols):
                    if name in c:
                        return i
            return None

        idx_date = _col_idx(["Date"])
        idx_lon  = _col_idx(["ObsSub-LON", "Sub-LON", "Sub_LON"])
        idx_lat  = _col_idx(["ObsSub-LAT", "Sub-LAT", "Sub_LAT"])

        if idx_lon is None or idx_lat is None:
            raise ValueError(
                f"Could not find ObsSub-LON/LAT columns. "
                f"Found columns: {cols}"
            )

        results = []
        for line in data_block.splitlines():
            line = line.strip()
            if not line or line.startswith("*"):
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                date_str = parts[idx_date]          # e.g. "2026-Oct-19 00:00"
                lon      = float(parts[idx_lon])
                lat      = float(parts[idx_lat])
                # Parse date
                try:
                    utc = datetime.strptime(date_str, "%Y-%b-%d %H:%M")
                except ValueError:
                    utc = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
                utc = utc.replace(tzinfo=timezone.utc)
                results.append({"utc": utc, "ObsSub_LON": lon, "ObsSub_LAT": lat})
            except (IndexError, ValueError):
                continue

        return results


# ---------------------------------------------------------------------------
# 2.  Magellan GREDR reflectivity map
# ---------------------------------------------------------------------------

class VenusReflectivityMap:
    """
    Loads the Magellan Global Reflectivity Data Record (GREDR) from the
    PDS Geosciences Node.

    Coordinate convention: West longitude, 0-360°, matching Horizons ObsSub-LON.

    Typical reflectivity values:
      - Average plains:       0.11 – 0.13
      - Aphrodite Terra:      0.20 – 0.35   (~105° W lon, equatorial)
      - Maxwell Montes:       0.40 – 0.60   (~3° W lon, ~65° N lat)

    Loader strategy (automatic, no action required):
      1. GDAL  — used if `from osgeo import gdal` succeeds.
                 Reads PDS3 natively, honours all label metadata.
                 Install: conda install -c conda-forge gdal
                          (plain `pip install GDAL` needs system libs first;
                           see GDAL install note in the notebook)
      2. Direct PDS3 parser  — pure Python / NumPy fallback used automatically
                 when GDAL is not available. Reads the .lbl sidecar directly
                 and decodes the binary .img. Works without any extra installs.

    Both paths produce identical internal state, so all downstream code
    (VenusAlbedoMapper, plots, etc.) is unaffected by which loader ran.

    Usage
    -----
        vmap = VenusReflectivityMap()
        vmap.download_and_cache()      # once; saves browse.img + browse.lbl
        vmap.load_from_files()         # auto-selects GDAL or fallback
        r = vmap.reflectivity_at(lat=0, lon_west=105)  # Aphrodite Terra
    """

    PDS_LBL_URL = (
        "https://pds-geosciences.wustl.edu/mgn/mgn-v-gxdr-v1/"
        "mg_3002/gredr/merc/browse.lbl"
    )
    PDS_IMG_URL = (
        "https://pds-geosciences.wustl.edu/mgn/mgn-v-gxdr-v1/"
        "mg_3002/gredr/merc/browse.img"
    )

    DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/ori_eve")

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.lbl_path  = os.path.join(self.cache_dir, "browse.lbl")
        self.img_path  = os.path.join(self.cache_dir, "browse.img")

        # Set after load_from_files() — identical regardless of which loader ran
        self._gt           = None   # (x_min, dx, 0, y_max, 0, dy) GeoTransform tuple
        self._reflectivity = None   # 2-D float64 array, shape (n_lat, n_lon)
        self._lat_centers  = None   # 1-D array, degrees North, descending
        self._lon_centers  = None   # 1-D array, in map's native lon convention
        self._lon_is_east  = False  # True if map uses East longitude (0→360°E)
        self._loaded       = False
        self._loader_used  = None   # "gdal" or "pds3"

    # --- acquisition ---

    def download_and_cache(self, force: bool = False) -> None:
        """
        Download browse.lbl and browse.img from PDS and save locally.
        Only needed once; skips existing files unless force=True.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        for url, path in [(self.PDS_LBL_URL, self.lbl_path),
                          (self.PDS_IMG_URL,  self.img_path)]:
            if os.path.exists(path) and not force:
                print(f"  Already cached: {path}")
                continue
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, path)
            print(f"  Saved to {path}  ({os.path.getsize(path):,} bytes)")

    # --- public loader (auto-selects GDAL or fallback) ---

    def load_from_files(
        self,
        lbl_path: Optional[str] = None,
        img_path: Optional[str] = None,
    ) -> None:
        """
        Load the GREDR map into a NumPy array.

        Tries GDAL first (handles any exotic PDS3 binary encoding).
        Falls back to the built-in PDS3 parser if GDAL is not installed.
        Both paths produce identical internal state.

        Note: GDAL is used only for reading the raw pixel data. Geographic
        bounds and scaling are always read from the PDS3 label directly,
        because the GREDR browse label does not include the GeoTransform
        keywords that GDAL's PDS3 driver needs for automatic georeferencing.
        """
        img = img_path or self.img_path
        lbl = lbl_path or self.lbl_path

        for path, label in [(img, "browse.img"), (lbl, "browse.lbl")]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{label} not found at {path}\n"
                    "Run VenusReflectivityMap().download_and_cache() first."
                )

        try:
            from osgeo import gdal
            self._load_via_gdal(img, lbl, gdal)
            self._loader_used = "gdal"
        except ImportError:
            print("  GDAL not available — using built-in PDS3 parser.")
            print("  (To use GDAL: conda install -c conda-forge gdal)")
            self._load_via_pds3(img, lbl)
            self._loader_used = "pds3"

    # --- GDAL loader ---

    def _load_via_gdal(self, img_path: str, lbl_path: str, gdal) -> None:
        """
        Use GDAL to read the raw pixel array, then apply geographic bounds
        and scaling from the PDS3 label directly.

        GDAL's PDS3 driver does not always find the georeferencing keywords
        in browse-product labels (it gets pixel-space coordinates instead of
        degrees). We therefore parse the label ourselves for all metadata and
        use GDAL only for its binary decoding of the image data.
        """
        gdal.UseExceptions()
        ds = gdal.Open(img_path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"GDAL could not open {img_path}")

        band = ds.GetRasterBand(1)
        nx   = ds.RasterXSize
        ny   = ds.RasterYSize

        # Read raw pixel values — GDAL handles any PDS3 binary encoding
        raw = band.ReadAsArray().astype(np.float64)

        # Parse geographic bounds and scaling from the label (authoritative)
        meta = self._parse_pds3_label(lbl_path)

        # Apply PDS3 scaling: reflectivity = raw * SCALING_FACTOR + OFFSET
        raw = raw * meta["scale"] + meta["offset"]

        # Compute the true left-edge longitude.
        # X_AXIS_PROJECTION_OFFSET is the column (0-based) where CENTER_LONGITUDE falls.
        # If absent, fall back to the label's stated MINIMUM_LONGITUDE.
        lat_min = meta["lat_min"]
        lat_max = meta["lat_max"]
        if meta.get("x_proj_offset") is not None:
            deg_per_pixel = 360.0 / nx
            lon_start = meta["center_longitude"] - meta["x_proj_offset"] * deg_per_pixel
        else:
            lon_start = meta["lon_min"]
        dx =  360.0 / nx
        dy = -(lat_max - lat_min) / ny
        self._gt = (lon_start, dx, 0.0, lat_max, 0.0, dy)

        self._lon_centers  = np.linspace(lon_start + dx/2, lon_start + 360.0 - dx/2, nx)
        self._lat_centers  = np.linspace(lat_max + dy/2, lat_min - dy/2, ny)
        self._reflectivity = np.clip(raw, 0.0, 1.0)
        self._lon_is_east  = meta.get("lon_direction", "WEST").upper() == "EAST"
        self._loaded       = True

        print(f"  GREDR loaded via GDAL: {ny} lines × {nx} samples")
        print(f"  Column-0 longitude: {lon_start:.1f}°  "
              f"({'East' if self._lon_is_east else 'West'})")
        print(f"  Latitude range:  {lat_min:.1f}° – {lat_max:.1f}° North")
        self._print_stats()

    # --- Pure-Python / NumPy PDS3 fallback loader ---

    def _load_via_pds3(self, img_path: str, lbl_path: str) -> None:
        """
        Load using a direct PDS3 parser — no external dependencies beyond NumPy.

        Reads SCALING_FACTOR, OFFSET, and all spatial keywords from the .lbl
        sidecar, then decodes the raw binary .img accordingly.
        """
        meta = self._parse_pds3_label(lbl_path)

        nlines  = meta["lines"]
        nsamples= meta["line_samples"]

        # Dtype from label
        dtype_map = {
            "PC_REAL":         np.float32,
            "IEEE_REAL":       np.float32,
            "SUN_INTEGER":     np.int16,
            "PC_INTEGER":      np.int16,
            "LSB_INTEGER":     np.int16,
            "MSB_INTEGER":     np.int16,
            "UNSIGNED_INTEGER":np.uint8,
            "BYTE":            np.uint8,
        }
        dt = dtype_map.get(meta["sample_type"].upper(), np.uint8)

        raw = np.fromfile(img_path, dtype=dt)

        # Some PDS3 products have a label prefix embedded in the file itself
        # (RECORD_TYPE = FIXED_LENGTH with LABEL_RECORDS > 0).
        # If the flat read gives us more bytes than nlines*nsamples, strip the header.
        expected = nlines * nsamples
        if raw.size > expected:
            raw = raw[-expected:]   # take the last N elements (the image data)
        elif raw.size < expected:
            # Pad with zeros if the file is shorter than expected (shouldn't happen)
            raw = np.pad(raw, (0, expected - raw.size))

        raw = raw.reshape(nlines, nsamples).astype(np.float64)

        # Apply PDS3 scaling: physical = raw * SCALING_FACTOR + OFFSET
        raw = raw * meta["scale"] + meta["offset"]

        # Compute the true left-edge longitude using the projection offset keyword
        lat_min = meta["lat_min"]
        lat_max = meta["lat_max"]
        if meta.get("x_proj_offset") is not None:
            deg_per_pixel = 360.0 / nsamples
            lon_start = meta["center_longitude"] - meta["x_proj_offset"] * deg_per_pixel
        else:
            lon_start = meta["lon_min"]
        dx =  360.0 / nsamples
        dy = -(lat_max - lat_min) / nlines
        self._gt = (lon_start, dx, 0.0, lat_max, 0.0, dy)

        self._lon_centers  = np.linspace(lon_start + dx/2, lon_start + 360.0 - dx/2, nsamples)
        self._lat_centers  = np.linspace(lat_max + dy/2, lat_min - dy/2, nlines)
        self._reflectivity = np.clip(raw, 0.0, 1.0)
        self._lon_is_east  = meta.get("lon_direction", "WEST").upper() == "EAST"
        self._loaded       = True

        print(f"  GREDR loaded via PDS3 parser: {nlines} lines × {nsamples} samples")
        print(f"  Column-0 longitude: {lon_start:.1f}°  "
              f"({'East' if self._lon_is_east else 'West'})")
        print(f"  Latitude range:  {lat_min:.1f}° – {lat_max:.1f}° North")
        self._print_stats()

    @staticmethod
    def _parse_pds3_label(lbl_path: str) -> dict:
        """
        Extract geometry and scaling keywords from a PDS3 label file.
        Returns a dict with everything _load_via_pds3 needs.
        """
        with open(lbl_path, "r", encoding="latin-1") as f:
            text = f.read()

        def _kw(pattern, default=None, cast=float):
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return cast(m.group(1).strip().strip('"').split()[0])
                except Exception:
                    pass
            return default

        return {
            "lines":        int(_kw(r"LINES\s*=\s*(\d+)",           512)),
            "line_samples": int(_kw(r"LINE_SAMPLES\s*=\s*(\d+)",    1024)),
            "sample_bits":  int(_kw(r"SAMPLE_BITS\s*=\s*(\d+)",     8)),
            "sample_type":  _kw(r'SAMPLE_TYPE\s*=\s*"?(\w+)"?',
                                "UNSIGNED_INTEGER", cast=str),
            "lat_min":      _kw(r"MINIMUM_LATITUDE\s*=\s*([-\d.]+)",      -70.0),
            "lat_max":      _kw(r"MAXIMUM_LATITUDE\s*=\s*([-\d.]+)",       70.0),
            # Accept both WESTERNMOST/EASTERNMOST and MINIMUM/MAXIMUM_LONGITUDE
            "lon_min":      (_kw(r"WESTERNMOST_LONGITUDE\s*=\s*([-\d.]+)")
                             or _kw(r"MINIMUM_LONGITUDE\s*=\s*([-\d.]+)",   0.0)),
            "lon_max":      (_kw(r"EASTERNMOST_LONGITUDE\s*=\s*([-\d.]+)")
                             or _kw(r"MAXIMUM_LONGITUDE\s*=\s*([-\d.]+)", 360.0)),
            "scale":        _kw(r"SCALING_FACTOR\s*=\s*([-\d.eE+]+)",  1.0/255),
            "offset":       _kw(r"OFFSET\s*=\s*([-\d.eE+]+)",              0.0),
            # "EAST" means columns go 0→360° East; "WEST" means 0→360° West
            "lon_direction": _kw(r"POSITIVE_LONGITUDE_DIRECTION\s*=\s*(\w+)",
                                 "WEST", cast=str),
            # Projection origin keywords — needed to compute the correct column-0 longitude
            "center_longitude": _kw(r"CENTER_LONGITUDE\s*=\s*([-\d.]+)",    0.0),
            "x_proj_offset":    _kw(r"X_AXIS_PROJECTION_OFFSET\s*=\s*([-\d.]+)", None),
        }

    def _print_stats(self) -> None:
        r = self._reflectivity
        print(f"  Reflectivity:    "
              f"min={r.min():.3f}  mean={r.mean():.3f}  max={r.max():.3f}")

    # --- point sampling ---

    def reflectivity_at(self, lat: float, lon_east: float) -> float:
        """
        Return bilinearly interpolated reflectivity at (lat [°N], lon_east [°E]).

        Accepts East longitude (0-360°E), which is the convention used by:
          - JPL Horizons ObsSub-LON for Venus
          - Magellan GREDR (POSITIVE_LONGITUDE_DIRECTION = EAST)
          - Venus feature databases (USGS Gazetteer)

        If the underlying map happens to be West-positive (rare), the conversion
        is handled internally using the _lon_is_east flag set during loading.
        """
        if not self._loaded:
            raise RuntimeError("Call load_from_files() first.")

        lon_east = lon_east % 360.0   # wrap to [0, 360)

        # Map the query longitude to the map's native pixel coordinate
        if self._lon_is_east:
            lon_query = lon_east                       # map is East-positive, use directly
        else:
            lon_query = (360.0 - lon_east) % 360.0    # convert East→West for West-positive maps

        # Normalize query longitude to the image's actual starting longitude.
        # The image may start at a non-zero longitude (e.g. -30°E for this GREDR file).
        # We map the query into [lon_start, lon_start + 360) before computing the column.
        lon_start = self._gt[0]
        lon_query = lon_start + (lon_query - lon_start) % 360.0

        gt = self._gt
        col_f = (lon_query - gt[0]) / gt[1]
        row_f = (lat       - gt[3]) / gt[5]

        ny, nx = self._reflectivity.shape
        col_f = np.clip(col_f, 0, nx - 1.001)
        row_f = np.clip(row_f, 0, ny - 1.001)

        # Bilinear interpolation using the four surrounding pixels
        c0, r0 = int(col_f), int(row_f)
        c1, r1 = min(c0 + 1, nx - 1), min(r0 + 1, ny - 1)
        wc = col_f - c0   # fractional column weight
        wr = row_f - r0   # fractional row weight

        r = (
            self._reflectivity[r0, c0] * (1 - wr) * (1 - wc)
            + self._reflectivity[r1, c0] * wr       * (1 - wc)
            + self._reflectivity[r0, c1] * (1 - wr) * wc
            + self._reflectivity[r1, c1] * wr       * wc
        )
        return float(r)

    def reflectivity_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (lat_centers, lon_centers, reflectivity_2d_array).

        lat_centers : 1-D array, degrees North, descending (north to south)
        lon_centers : 1-D array, degrees West, ascending (0 → 360)
        reflectivity: 2-D float64 array, shape (n_lat, n_lon)

        These are derived directly from GDAL's GeoTransform so they are
        consistent with the map's actual projection metadata.
        """
        if not self._loaded:
            raise RuntimeError("Call load_from_files() first.")
        return self._lat_centers, self._lon_centers, self._reflectivity

    def plot_map(
        self,
        sub_lon_east: Optional[float] = None,
        sub_lat: Optional[float] = None,
        title: str = "Magellan GREDR Venus Radar Reflectivity",
    ) -> plt.Figure:
        """
        Plot the global reflectivity map.
        Optionally mark the sub-Earth point and the visible hemisphere boundary.
        """
        if not self._loaded:
            raise RuntimeError("Call load_from_files() first.")

        lats, lons, refl = self.reflectivity_grid()

        fig, ax = plt.subplots(figsize=(14, 7))
        im = ax.imshow(
            refl,
            extent=[lons[0], lons[-1], lats[-1], lats[0]],
            origin="upper",
            aspect="auto",
            cmap="inferno",
            vmin=0.0,
            vmax=0.5,
        )
        plt.colorbar(im, ax=ax, label="Radar Reflectivity")

        # Annotate known bright features
        ax.annotate("Maxwell\nMontes", xy=(90, 65), color="cyan", fontsize=8,
                    ha="center", fontweight="bold")
        ax.annotate("Aphrodite\nTerra", xy=(292, 5), color="cyan", fontsize=8,
                    ha="center", fontweight="bold")

        if sub_lon_east is not None:
            # Draw the limb circle (great circle 90° from sub-Earth point)
            # This marks the boundary of the hemisphere visible to Earth's radar
            phi_sub = np.radians(sub_lat or 0.0)
            lam_sub = np.radians(sub_lon_east)

            lam_plot = np.linspace(0, 360, 361)
            lat_plot = np.linspace(-90, 90, 181)
            LAM, PHI = np.meshgrid(np.radians(lam_plot), np.radians(lat_plot))

            cos_delta = (
                np.sin(phi_sub) * np.sin(PHI)
                + np.cos(phi_sub) * np.cos(PHI) * np.cos(LAM - lam_sub)
            )
            ax.contour(lam_plot, lat_plot, cos_delta, levels=[0.0],
                       colors="lime", linewidths=1.5, linestyles="--")

            ax.plot(sub_lon_east, sub_lat or 0, "g*", markersize=14,
                    label=f"Sub-Earth point\n({sub_lon_east:.1f}° E, {sub_lat:.1f}° N)")
            ax.legend(loc="lower right")

        ax.set_xlabel("GREDR East Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees N)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# 3.  Disk integrator: visible hemisphere → effective albedo
# ---------------------------------------------------------------------------

class VenusAlbedoMapper:
    """
    Computes the *disk-integrated* effective radar albedo seen from Earth
    for a given sub-Earth longitude and latitude.

    For dishes that over-illuminate Venus (Dwingeloo 25 m, DSES 18.3 m at
    2.3 GHz — both are well below the ~1.75° / 2 = 0.875° beamwidth needed
    to resolve Venus at inferior conjunction), the radar cross section is:

        σ = π R² · ρ_eff

    where ρ_eff is the power-weighted mean reflectivity over the visible
    hemisphere, weighted by the radar backscatter function.

    Backscatter weighting model
    ---------------------------
    Following Hagfors (1964) and the practice in planetary radar, we use:

        w(Δ) = cos^n(Δ)

    where Δ is the angle from the sub-radar point and n controls how
    specular the surface is (n = 2 for quasi-specular / Fresnel,
    n = 1 for Lambertian). Venus at decimeter wavelengths is quasi-specular
    with n ≈ 2–4 for the smooth plains.

    Coordinate system note
    ----------------------
    The Magellan GREDR uses the 1990s Magellan mission coordinate frame.
    JPL Horizons uses the IAU 2000 Venus frame. These differ by approximately
    180.7° theoretically (different W₀ in the rotation model), measured
    empirically as ~186.5° from Aphrodite Terra's known position:

        GREDR_lon = Horizons_ObsSub_LON + horizons_lon_offset

    This offset is applied automatically inside effective_albedo().
    If you load a different Venus reflectivity dataset that already uses
    the IAU 2000 frame, set horizons_lon_offset=0.

    Parameters
    ----------
    reflectivity_map : VenusReflectivityMap  (must already be loaded)
    n_backscatter    : float
        Hagfors exponent for the surface scattering law (default 2.0).
    grid_resolution_deg : float
        Angular step size for the integration grid (degrees).
    horizons_lon_offset : float
        Degrees to add to Horizons ObsSub-LON before looking up the GREDR map.
        Default 186.5° — the empirically measured Magellan↔IAU 2000 offset.
    """

    # Empirically measured offset between Horizons ObsSub-LON (IAU 2000 frame)
    # and Magellan GREDR East longitude.
    # Derived from: Aphrodite Terra at GREDR 291.5°E = Horizons 105°E
    # Consistent with theoretical ~180.7° difference in Venus rotation W₀.
    GREDR_HORIZONS_LON_OFFSET = 186.5

    def __init__(
        self,
        reflectivity_map: VenusReflectivityMap,
        n_backscatter: float = 2.0,
        grid_resolution_deg: float = 0.5,
        horizons_lon_offset: float = GREDR_HORIZONS_LON_OFFSET,
    ):
        self.vmap = reflectivity_map
        self.n = n_backscatter
        self.dres = grid_resolution_deg
        self.horizons_lon_offset = horizons_lon_offset
        self._ephemeris = None  # lazy-init

    @property
    def ephemeris(self) -> HorizonsVenusEphemeris:
        if self._ephemeris is None:
            self._ephemeris = HorizonsVenusEphemeris()
        return self._ephemeris

    def effective_albedo(
        self,
        sub_lon_east_deg: float,
        sub_lat_deg: float = 0.0,
    ) -> dict:
        """
        Compute ρ_eff for a given Venus orientation toward Earth.

        Parameters
        ----------
        sub_lon_east_deg : float
            Venus East longitude of the sub-Earth point.
            Pass Horizons ObsSub-LON directly — no conversion needed.
        sub_lat_deg : float
            Venus latitude of the sub-Earth point (Horizons ObsSub-LAT).

        Returns
        -------
        dict with keys:
            'albedo_eff'  : float  — disk-integrated effective albedo
            'sub_lon_E'   : float  — sub-Earth East longitude used
            'sub_lat_N'   : float  — sub-Earth latitude used
            'albedo_min'  : float  — 10th percentile of visible reflectivity
            'albedo_max'  : float  — 90th percentile
        """
        # Convert Horizons ObsSub-LON (IAU 2000 frame) to GREDR map longitude
        # by applying the Magellan↔IAU 2000 coordinate frame offset.
        map_lon_center = (sub_lon_east_deg + self.horizons_lon_offset) % 360.0

        # Build integration grid in GREDR map longitude
        lon_e = np.arange(0.0, 360.0, self.dres)
        lat   = np.arange(-90.0, 90.0 + self.dres, self.dres)
        LON, LAT = np.meshgrid(lon_e, lat)

        phi_sub = np.radians(sub_lat_deg)
        lam_sub = np.radians(map_lon_center)   # GREDR longitude in radians
        PHI     = np.radians(LAT)
        LAM     = np.radians(LON)                 # East longitude in radians

        # Cosine of angle from sub-Earth point (independent of lon convention)
        cos_delta = (
            np.sin(phi_sub) * np.sin(PHI)
            + np.cos(phi_sub) * np.cos(PHI) * np.cos(LAM - lam_sub)
        )

        visible = cos_delta > 0.0

        if not np.any(visible):
            return {
                "albedo_eff": 0.11,
                "sub_lon_E":  sub_lon_east_deg,
                "sub_lat_N":  sub_lat_deg,
                "albedo_min": 0.11,
                "albedo_max": 0.11,
            }

        weight       = np.zeros_like(cos_delta)
        weight[visible] = cos_delta[visible] ** self.n
        cos_lat      = np.cos(PHI)
        area_element = cos_lat

        # Reflectivity lookup — lon_e values are East longitude, matching map convention
        lat_flat  = LAT[visible].ravel()
        lon_flat  = LON[visible].ravel()   # East longitude
        refl_flat = np.array([
            self.vmap.reflectivity_at(la, lo)
            for la, lo in zip(lat_flat, lon_flat)
        ])

        w_flat    = weight[visible].ravel()
        area_flat = area_element[visible].ravel()
        combined  = w_flat * area_flat

        numerator   = np.sum(refl_flat * combined)
        denominator = np.sum(combined)
        albedo_eff  = numerator / denominator if denominator > 0 else 0.11

        return {
            "albedo_eff": float(albedo_eff),
            "sub_lon_E":  sub_lon_east_deg,
            "sub_lat_N":  sub_lat_deg,
            "albedo_min": float(np.percentile(refl_flat, 10)),
            "albedo_max": float(np.percentile(refl_flat, 90)),
        }

    def albedo_for_date(self, date: datetime) -> dict:
        """
        Full pipeline: date → Horizons ObsSub-LON (East) → disk integration → ρ_eff.
        Horizons ObsSub-LON for Venus is East longitude — passed directly, no conversion.
        """
        sub_lon, sub_lat = self.ephemeris.get_sub_earth_at_date(date)
        result = self.effective_albedo(sub_lon, sub_lat)   # sub_lon is already East
        result["utc_date"] = date
        return result

    def albedo_series(
        self,
        start: datetime,
        stop: datetime,
        step: str = "1d",
    ) -> list[dict]:
        """
        Compute albedo for a date range.
        Returns list of dicts with 'utc_date', 'albedo_eff', 'sub_lon_E', etc.
        """
        rows = self.ephemeris.get_sub_earth_series(start, stop, step)
        results = []
        n = len(rows)
        for i, row in enumerate(rows):
            if i % max(1, n // 10) == 0:
                print(f"  Integrating... {i}/{n}  "
                      f"(ObsSub-LON = {row['ObsSub_LON']:.1f}°E)")
            res = self.effective_albedo(row["ObsSub_LON"], row["ObsSub_LAT"])
            res["utc_date"] = row["utc"]
            results.append(res)
        return results

    def plot_albedo_series(
        self,
        results: list[dict],
        conjunction_date: Optional[datetime] = None,
        title: str = "Venus Effective Radar Albedo vs Date",
    ) -> plt.Figure:
        """
        Plot ρ_eff versus date, with optional inferior conjunction marker.

        Parameters
        ----------
        results : output of albedo_series()
        conjunction_date : datetime  — vertical line at inferior conjunction
        """
        dates   = [r["utc_date"] for r in results]
        albedos = [r["albedo_eff"] for r in results]
        lons    = [r["sub_lon_E"]  for r in results]

        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

        ax1 = axes[0]
        ax1.plot(dates, albedos, "b-o", markersize=3, linewidth=1.5,
                 label="ρ_eff (disk-integrated)")
        ax1.axhline(y=0.152, color="gray", linestyle="--", linewidth=1,
                    label="Nominal scalar (0.152)")
        ax1.axhspan(0.11, 0.13, alpha=0.12, color="gray",
                    label="Prior range 0.11–0.13")
        # Aphrodite Terra peak reference
        ax1.axhline(y=0.35, color="orange", linestyle=":", linewidth=1.0,
                    label="Aphrodite Terra peak (~0.35)")
        if conjunction_date:
            ax1.axvline(x=conjunction_date, color="red", linestyle="-",
                        linewidth=2, label="Inferior conjunction")
        ax1.set_ylabel("Effective Radar Albedo ρ_eff")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 0.55)

        ax2 = axes[1]
        ax2.plot(dates, lons, "g-", linewidth=1.5)
        ax2.axhline(y=105, color="orange", linestyle="--", linewidth=1,
                    label="Aphrodite Terra (~105° W)")
        ax2.axhline(y=3,   color="cyan",   linestyle="--", linewidth=1,
                    label="Maxwell Montes (~3° W)")
        if conjunction_date:
            ax2.axvline(x=conjunction_date, color="red", linestyle="-",
                        linewidth=2)
        ax2.set_ylabel("ObsSub-LON (degrees East)")
        ax2.set_xlabel("Date (UTC)")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.set_ylim(0, 360)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# 4.  Drop-in replacement for the fixed albedo in EVELinkBudget
# ---------------------------------------------------------------------------

class DynamicAlbedoEVELinkBudget:
    """
    Wraps the existing EVELinkBudget and overrides venus_radar_albedo
    with the date-resolved, map-integrated value from VenusAlbedoMapper.

    Usage
    -----
        # One-time setup
        vmap = VenusReflectivityMap()
        vmap.download_and_cache()   # downloads ~3 MB from PDS once
        vmap.load_from_files()

        mapper = VenusAlbedoMapper(vmap)

        # Create the enhanced budget
        params   = DSESLinkParameters()
        budget   = DynamicAlbedoEVELinkBudget(params, mapper)

        # October 2026 inferior conjunction
        date     = datetime(2026, 10, 22, 0, 0, tzinfo=timezone.utc)
        result   = budget.calculate_link_budget_for_date(distance_km=42_000_000, date=date)
        print(f"Albedo used: {result['venus_radar_albedo']:.4f}")
        print(f"CNR:         {result['cnr_db']:+.2f} dB")
    """

    def __init__(
        self,
        params,
        albedo_mapper: VenusAlbedoMapper,
        base_link_budget_class=None,
    ):
        self.params = params
        self.mapper = albedo_mapper

        # If user passes their existing EVELinkBudget class, we use it.
        # Otherwise import or define inline.
        if base_link_budget_class is not None:
            self._budget_class = base_link_budget_class
        else:
            # Fall back to importing from the notebook namespace — at runtime
            # this will resolve correctly when used inside the notebook.
            self._budget_class = None

    def calculate_link_budget_for_date(
        self,
        distance_km: float,
        date: datetime,
    ) -> dict:
        """
        Resolve Venus albedo from the map at the given date,
        then compute the full link budget.

        Returns the standard EVELinkBudget dict plus:
          'venus_radar_albedo'  : the spatially resolved ρ_eff used
          'sub_lon_E'           : sub-Earth longitude (Horizons ObsSub-LON)
          'sub_lat_N'           : sub-Earth latitude
          'albedo_source'       : 'VenusAlbedoMapper / Magellan GREDR'
        """
        albedo_info = self.mapper.albedo_for_date(date)
        eff_albedo  = albedo_info["albedo_eff"]

        # Create a fresh budget instance and override the albedo
        if self._budget_class is not None:
            budget = self._budget_class(self.params)
        else:
            raise RuntimeError(
                "No base link budget class provided. Pass EVELinkBudget as "
                "base_link_budget_class= when constructing DynamicAlbedoEVELinkBudget."
            )

        budget.venus_radar_albedo = eff_albedo
        result = budget.calculate_link_budget(distance_km)
        result["venus_radar_albedo"] = eff_albedo
        result["sub_lon_E"]         = albedo_info["sub_lon_E"]
        result["sub_lat_N"]         = albedo_info["sub_lat_N"]
        result["albedo_source"]     = "VenusAlbedoMapper / Magellan GREDR"
        return result

    def scan_conjunction_window(
        self,
        distance_km: float,
        start: datetime,
        stop: datetime,
        step: str = "1d",
    ) -> list[dict]:
        """
        Compute full link budgets for every day in a date range.
        Useful for finding the best day to attempt EVE during a conjunction window.

        Returns list of result dicts, each augmented with 'utc_date'.
        """
        albedo_series = self.mapper.albedo_series(start, stop, step)
        results = []
        for entry in albedo_series:
            budget = self._budget_class(self.params)
            budget.venus_radar_albedo = entry["albedo_eff"]
            r = budget.calculate_link_budget(distance_km)
            r.update({
                "utc_date":         entry["utc_date"],
                "venus_radar_albedo": entry["albedo_eff"],
                "sub_lon_E":        entry["sub_lon_E"],
                "sub_lat_N":        entry["sub_lat_N"],
            })
            results.append(r)
        return results

    def plot_cnr_vs_date(
        self,
        scan_results: list[dict],
        conjunction_date: Optional[datetime] = None,
    ) -> plt.Figure:
        """
        Dual-panel plot:
          Top:    CNR (dB) vs date for the dynamic-albedo budget
          Bottom: Effective albedo vs date
        """
        dates   = [r["utc_date"]          for r in scan_results]
        cnrs    = [r["cnr_db"]            for r in scan_results]
        albedos = [r["venus_radar_albedo"] for r in scan_results]

        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

        ax1 = axes[0]
        ax1.plot(dates, cnrs, "b-o", markersize=3, linewidth=1.5,
                 label="CNR (dynamic albedo)")
        if conjunction_date:
            ax1.axvline(x=conjunction_date, color="red", linestyle="-",
                        linewidth=2, label="Inferior conjunction")
        ax1.set_ylabel("CNR (dB)")
        ax1.set_title("EVE Link Budget CNR — Spatially Resolved Venus Albedo")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(dates, albedos, "g-o", markersize=3, linewidth=1.5,
                 label="ρ_eff")
        ax2.axhline(y=0.152, color="gray", linestyle="--", linewidth=1,
                    label="Nominal scalar (0.152)")
        if conjunction_date:
            ax2.axvline(x=conjunction_date, color="red", linestyle="-",
                        linewidth=2)
        ax2.set_ylabel("Effective Radar Albedo")
        ax2.set_xlabel("Date (UTC)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# 5.  Convenience: known Venus feature longitudes (West)
# ---------------------------------------------------------------------------

VENUS_RADAR_FEATURES = {
    # Longitudes in the GREDR (Magellan) East-positive frame.
    # To convert to Horizons/IAU 2000 frame: lon_horizons = (lon_GREDR - 186.5) % 360
    #
    # Feature                          GREDR lon_E   Horizons lon_E   lat
    "Aphrodite Terra (equatorial max)":{"lon_E": 292, "lat":  -5, "typ_refl": 0.45,
                                        "note": "Brightest equatorial feature; Horizons ~105°E"},
    "Aphrodite Terra (western lobe)":  {"lon_E": 250, "lat":  -5, "typ_refl": 0.22,
                                        "note": "Western edge; Horizons ~63°E"},
    "Maxwell Montes":                  {"lon_E":  90, "lat":  65, "typ_refl": 0.75,
                                        "note": "Global max; polar; Horizons ~264°E"},
    "Average lowland plains":          {"lon_E": 180, "lat":   0, "typ_refl": 0.11,
                                        "note": "Global baseline"},
}


def print_feature_table():
    """Print known bright Venus radar features with East longitudes."""
    print(f"{'Feature':<35} {'Lon E':>6}  {'Lat':>5}  {'Typ Refl':>9}  Note")
    print("-" * 80)
    for name, info in VENUS_RADAR_FEATURES.items():
        print(f"{name:<35} {info['lon_E']:>6.0f}°  {info['lat']:>+5.0f}°  {info['typ_refl']:>9.2f}  {info['note']}")

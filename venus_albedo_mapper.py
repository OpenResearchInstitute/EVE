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

    BASE_URL = "https://ssd-api.jpl.nasa.gov/horizons.api"

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
            return dt.strftime("%Y-%b-%d %H:%M")

        params = {
            "format": "json",
            "COMMAND": "299",          # Venus barycenter
            "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "OBSERVER",
            "CENTER": "500@399",       # Geocenter
            "START_TIME": _fmt(start),
            "STOP_TIME": _fmt(stop),
            "STEP_SIZE": step,
            "QUANTITIES": "14",        # ObsSub-LON, ObsSub-LAT
            "ANG_FORMAT": "DEG",
            "CAL_FORMAT": "CAL",
            "CSV_FORMAT": "YES",
        }

        url = self.BASE_URL + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode())

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
# 2.  Magellan GREDR reflectivity map  (GDAL-backed)
# ---------------------------------------------------------------------------

class VenusReflectivityMap:
    """
    Loads the Magellan Global Reflectivity Data Record (GREDR) from the
    PDS Geosciences Node using GDAL, which reads PDS3 .img files natively
    and handles all projection / georeferencing metadata automatically.

    Coordinate convention: West longitude, 0-360°, matching Horizons ObsSub-LON.

    Typical reflectivity values:
      - Average plains:       0.11 – 0.13
      - Aphrodite Terra:      0.20 – 0.35   (~105° W lon, equatorial)
      - Maxwell Montes:       0.40 – 0.60   (~3° W lon, ~65° N lat)

    Install GDAL:
        pip install GDAL
    or on conda:
        conda install -c conda-forge gdal

    Usage
    -----
        vmap = VenusReflectivityMap()
        vmap.download_and_cache()      # once; saves browse.img + browse.lbl
        vmap.load_from_files()
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

        # Set after load_from_files()
        self._ds           = None   # osgeo.gdal.Dataset (kept open for sampling)
        self._band         = None   # raster band 1
        self._gt           = None   # GDAL GeoTransform tuple
        self._reflectivity = None   # full 2-D float array (loaded on demand)
        self._lat_centers  = None
        self._lon_centers  = None
        self._nodata       = None
        self._loaded       = False

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

    # --- GDAL loader ---

    def load_from_files(
        self,
        lbl_path: Optional[str] = None,
        img_path: Optional[str] = None,
    ) -> None:
        """
        Open the PDS3 image with GDAL and read it into a NumPy array.

        GDAL reads the .lbl sidecar automatically when you open the .img file
        (it follows the ^IMAGE pointer in the label). The GeoTransform gives
        us pixel → (lon, lat) mapping with no manual label parsing needed.

        Parameters
        ----------
        lbl_path, img_path : optional paths; defaults to cached copies.
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise ImportError(
                "GDAL is required for VenusReflectivityMap.\n"
                "Install it with:  pip install GDAL\n"
                "or:               conda install -c conda-forge gdal"
            )

        img = img_path or self.img_path
        if not os.path.exists(img):
            raise FileNotFoundError(
                f"Image file not found: {img}\n"
                "Run VenusReflectivityMap().download_and_cache() first."
            )

        # GDAL opens PDS3 IMG files natively; it reads the paired .lbl
        # automatically via the ^IMAGE pointer if the label is a sidecar.
        gdal.UseExceptions()
        self._ds = gdal.Open(img, gdal.GA_ReadOnly)
        if self._ds is None:
            raise RuntimeError(f"GDAL could not open {img}")

        self._band = self._ds.GetRasterBand(1)

        # GeoTransform: (x_min, pixel_width, 0, y_max, 0, -pixel_height)
        #   x = longitude (West, 0-360), y = latitude (N positive)
        self._gt = self._ds.GetGeoTransform()
        nx = self._ds.RasterXSize
        ny = self._ds.RasterYSize

        x_min, dx, _, y_max, _, dy = self._gt   # dy is negative for north-up
        x_max = x_min + dx * nx
        y_min = y_max + dy * ny   # dy < 0 so this subtracts

        # Build coordinate arrays (pixel centres)
        self._lon_centers = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)
        self._lat_centers = np.linspace(y_max + dy / 2, y_min - dy / 2, ny)

        # NoData value (invalid pixels — some PDS products use 0 or 255)
        nd = self._band.GetNoDataValue()
        self._nodata = nd

        # Read the full band into a float64 array and apply scale/offset
        # GDAL applies SCALE and OFFSET from the label automatically when
        # you read via ReadAsArray() — the result is already in physical units.
        raw = self._band.ReadAsArray().astype(np.float64)

        # Mask nodata pixels; fill with the global median so they don't
        # bias the hemisphere integration
        if nd is not None:
            mask = (raw == nd)
            if mask.any():
                raw[mask] = np.nanmedian(raw[~mask])

        # Clip to [0, 1] — physical reflectivity must be in this range
        self._reflectivity = np.clip(raw, 0.0, 1.0)
        self._loaded = True

        print(f"  GREDR loaded via GDAL: {ny} lines × {nx} samples")
        print(f"  Longitude range: {x_min:.1f}° – {x_max:.1f}° West")
        print(f"  Latitude range:  {y_min:.1f}° – {y_max:.1f}° North")
        print(f"  Reflectivity:    "
              f"min={self._reflectivity.min():.3f}  "
              f"mean={self._reflectivity.mean():.3f}  "
              f"max={self._reflectivity.max():.3f}")

    # --- point sampling ---

    def reflectivity_at(self, lat: float, lon_west: float) -> float:
        """
        Return bilinearly interpolated reflectivity at (lat [°N], lon_west [°W]).

        Uses GDAL's pixel-coordinate transform so the interpolation is
        consistent with the map's actual projection metadata.
        """
        if not self._loaded:
            raise RuntimeError("Call load_from_files() first.")

        lon_west = lon_west % 360.0   # wrap to [0, 360)

        # Convert geographic → fractional pixel coordinates via inverse GeoTransform.
        # GeoTransform: X_geo = gt[0] + col*gt[1] + row*gt[2]
        #               Y_geo = gt[3] + col*gt[4] + row*gt[5]
        # For a north-up image (gt[2]=gt[4]=0):
        #   col = (X_geo - gt[0]) / gt[1]
        #   row = (Y_geo - gt[3]) / gt[5]
        gt = self._gt
        col_f = (lon_west - gt[0]) / gt[1]
        row_f = (lat      - gt[3]) / gt[5]

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
        sub_lon_west: Optional[float] = None,
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
        ax.annotate("Maxwell\nMontes", xy=(3, 65), color="cyan", fontsize=8,
                    ha="center", fontweight="bold")
        ax.annotate("Aphrodite\nTerra", xy=(105, 5), color="cyan", fontsize=8,
                    ha="center", fontweight="bold")

        if sub_lon_west is not None:
            # Draw the limb circle (great circle 90° from sub-Earth point)
            # This marks the boundary of the hemisphere visible to Earth's radar
            phi_sub = np.radians(sub_lat or 0.0)
            lam_sub = np.radians(sub_lon_west)

            lam_plot = np.linspace(0, 360, 361)
            lat_plot = np.linspace(-90, 90, 181)
            LAM, PHI = np.meshgrid(np.radians(lam_plot), np.radians(lat_plot))

            cos_delta = (
                np.sin(phi_sub) * np.sin(PHI)
                + np.cos(phi_sub) * np.cos(PHI) * np.cos(LAM - lam_sub)
            )
            ax.contour(lam_plot, lat_plot, cos_delta, levels=[0.0],
                       colors="lime", linewidths=1.5, linestyles="--")

            ax.plot(sub_lon_west, sub_lat or 0, "g*", markersize=14,
                    label=f"Sub-Earth point\n({sub_lon_west:.1f}° W, {sub_lat:.1f}° N)")
            ax.legend(loc="lower right")

        ax.set_xlabel("West Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees N)")
        ax.set_title(title)
        ax.invert_xaxis()   # West longitude increases right→left conventionally
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

    The effective albedo is:

        ρ_eff = ∫∫ r(λ,φ) · w(Δ) · cos(φ) dλ dφ
               ─────────────────────────────────────
                ∫∫ w(Δ) · cos(φ) dλ dφ

    (integral over the visible hemisphere, Δ < 90°)

    Parameters
    ----------
    reflectivity_map : VenusReflectivityMap  (must already be loaded)
    n_backscatter    : float
        Hagfors exponent for the surface scattering law (default 2.0).
        Higher n → more specular → sub-radar region dominates more.
    grid_resolution_deg : float
        Angular step size for the integration grid (degrees).
        0.5° gives good accuracy and runs in < 1 s.
    """

    def __init__(
        self,
        reflectivity_map: VenusReflectivityMap,
        n_backscatter: float = 2.0,
        grid_resolution_deg: float = 0.5,
    ):
        self.vmap = reflectivity_map
        self.n = n_backscatter
        self.dres = grid_resolution_deg
        self._ephemeris = None  # lazy-init

    @property
    def ephemeris(self) -> HorizonsVenusEphemeris:
        if self._ephemeris is None:
            self._ephemeris = HorizonsVenusEphemeris()
        return self._ephemeris

    def effective_albedo(
        self,
        sub_lon_west_deg: float,
        sub_lat_deg: float = 0.0,
    ) -> dict:
        """
        Compute ρ_eff for a given Venus orientation toward Earth.

        Parameters
        ----------
        sub_lon_west_deg : float
            Venus West longitude of the sub-Earth point (from Horizons ObsSub-LON).
        sub_lat_deg : float
            Venus latitude of the sub-Earth point (from Horizons ObsSub-LAT).

        Returns
        -------
        dict with keys:
            'albedo_eff'  : float  — the disk-integrated effective albedo (dimensionless)
            'sub_lon_W'   : float
            'sub_lat_N'   : float
            'albedo_min'  : float  — 10th percentile of visible reflectivity (context)
            'albedo_max'  : float  — 90th percentile
        """
        # Build integration grid over the full sphere in (lon_west, lat)
        lon_w = np.arange(0.0, 360.0, self.dres)
        lat   = np.arange(-90.0, 90.0 + self.dres, self.dres)
        LON, LAT = np.meshgrid(lon_w, lat)

        phi_sub = np.radians(sub_lat_deg)
        lam_sub = np.radians(sub_lon_west_deg)
        PHI     = np.radians(LAT)
        LAM     = np.radians(LON)

        # Cosine of angle from sub-Earth point
        cos_delta = (
            np.sin(phi_sub) * np.sin(PHI)
            + np.cos(phi_sub) * np.cos(PHI) * np.cos(LAM - lam_sub)
        )

        # Visibility mask: only the hemisphere facing Earth (cos_delta > 0)
        visible = cos_delta > 0.0

        if not np.any(visible):
            return {
                "albedo_eff": 0.11,  # fallback
                "sub_lon_W": sub_lon_west_deg,
                "sub_lat_N": sub_lat_deg,
                "albedo_min": 0.11,
                "albedo_max": 0.11,
            }

        # Backscatter weight: w(Δ) = cos^n(Δ)
        weight = np.zeros_like(cos_delta)
        weight[visible] = cos_delta[visible] ** self.n

        # Area element on the sphere: cos(φ) dφ dλ
        cos_lat = np.cos(PHI)
        area_element = cos_lat  # dλ and dφ are equal for our uniform grid

        # Reflectivity map lookup — vectorised over the integration grid
        # (Flatten → look up → reshape)
        lat_flat = LAT[visible].ravel()
        lon_flat = LON[visible].ravel()
        refl_flat = np.array([
            self.vmap.reflectivity_at(la, lo)
            for la, lo in zip(lat_flat, lon_flat)
        ])

        w_flat   = weight[visible].ravel()
        area_flat = area_element[visible].ravel()
        combined  = w_flat * area_flat

        numerator   = np.sum(refl_flat * combined)
        denominator = np.sum(combined)

        albedo_eff = numerator / denominator if denominator > 0 else 0.11

        # Percentile context (unweighted, for sanity check)
        p10 = float(np.percentile(refl_flat, 10))
        p90 = float(np.percentile(refl_flat, 90))

        return {
            "albedo_eff": float(albedo_eff),
            "sub_lon_W":  sub_lon_west_deg,
            "sub_lat_N":  sub_lat_deg,
            "albedo_min": p10,
            "albedo_max": p90,
        }

    def albedo_for_date(self, date: datetime) -> dict:
        """
        Full pipeline: date → Horizons sub-Earth point → disk integration → ρ_eff.

        Returns the same dict as effective_albedo(), plus 'utc_date'.
        """
        sub_lon, sub_lat = self.ephemeris.get_sub_earth_at_date(date)
        result = self.effective_albedo(sub_lon, sub_lat)
        result["utc_date"] = date
        return result

    def albedo_series(
        self,
        start: datetime,
        stop: datetime,
        step: str = "1d",
    ) -> list[dict]:
        """
        Compute albedo for a date range.  Returns list of dicts,
        each containing 'utc_date', 'albedo_eff', 'sub_lon_W', etc.

        step : Horizons step string ('1d', '6h', '12h', etc.)
        """
        # Fetch all sub-Earth positions in one Horizons call
        rows = self.ephemeris.get_sub_earth_series(start, stop, step)
        results = []
        n = len(rows)
        for i, row in enumerate(rows):
            if i % max(1, n // 10) == 0:
                print(f"  Integrating... {i}/{n}  "
                      f"(ObsSub-LON = {row['ObsSub_LON']:.1f}°)")
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
        lons    = [r["sub_lon_W"]  for r in results]

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
        ax2.set_ylabel("ObsSub-LON (degrees West)")
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
          'sub_lon_W'           : sub-Earth longitude (Horizons ObsSub-LON)
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
        result["sub_lon_W"]         = albedo_info["sub_lon_W"]
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
                "sub_lon_W":        entry["sub_lon_W"],
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
    "Aphrodite Terra (main lobe)": {"lon_W": 105, "lat":  -5, "typ_refl": 0.25, "note": "Equatorial; large area"},
    "Aphrodite Terra (eastern)":   {"lon_W":  60, "lat":  -5, "typ_refl": 0.22, "note": "Eastern lobe"},
    "Maxwell Montes":              {"lon_W":   3, "lat":  65, "typ_refl": 0.50, "note": "Polar; very bright; small"},
    "Alpha Regio":                 {"lon_W": 180, "lat": -25, "typ_refl": 0.18, "note": "Southern highlands"},
    "Beta Regio":                  {"lon_W": 283, "lat":  25, "typ_refl": 0.18, "note": "Volcanic rises"},
    "Average lowland plains":      {"lon_W":   0, "lat":   0, "typ_refl": 0.12, "note": "Global baseline"},
}


def print_feature_table():
    """Print the known bright Venus radar features and their West longitudes."""
    print(f"{'Feature':<35} {'Lon W':>6}  {'Lat':>5}  {'Typ Refl':>9}  Note")
    print("-" * 80)
    for name, info in VENUS_RADAR_FEATURES.items():
        print(f"{name:<35} {info['lon_W']:>6.0f}°  {info['lat']:>+5.0f}°  {info['typ_refl']:>9.2f}  {info['note']}")

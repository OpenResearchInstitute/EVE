#!/usr/bin/env python3
"""Complete dimensioned drawing set, septum feed 2304 MHz -- 5 sheets.
Shop-facing: final dimensions govern; vendor develops own flats for
formed parts. Regenerate after any SCAD change (numbers mirrored)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ---- parameters (mirror septum_feed_2304_v12.scad, t = 1.016) ----
t = 1.016; wg_id = 81.51; sept_t = 1.016
wg_od = wg_id + 2*t                    # 83.54
depth_i = (wg_id - sept_t)/2           # 40.25
wall_o = depth_i + t                   # 41.26
flange = 15.0; L = 354.0
flare_od = 184.20; slant = 194.46
cap_face = 84.14
cap_hole_dy = wg_id/4                  # 20.38
collar_hole_dy = wg_od/2 - flange      # 26.77
sept_L = 339.0; sept_W = 113.54; ear = 16.02
# septum steps, from REAR edge of blank: (start, end, height)
steps = [(0, 83.19, 81.51), (83.19, 95.34, 63.96), (95.34, 130.61, 39.21),
         (130.61, 164.36, 23.18), (164.36, 208.35, 10.41)]

def dim_h(ax, x0, x1, y, label, off=3, fs=8):
    ax.annotate('', (x0, y), (x1, y), arrowprops=dict(arrowstyle='<->', lw=0.8))
    ax.plot([x0]*2, [y-off, y+off], 'k', lw=0.5)
    ax.plot([x1]*2, [y-off, y+off], 'k', lw=0.5)
    ax.text((x0+x1)/2, y+1.5, label, ha='center', fontsize=fs)

def dim_v(ax, x, y0, y1, label, off=3, fs=8):
    ax.annotate('', (x, y0), (x, y1), arrowprops=dict(arrowstyle='<->', lw=0.8))
    ax.plot([x-off, x+off], [y0]*2, 'k', lw=0.5)
    ax.plot([x-off, x+off], [y1]*2, 'k', lw=0.5)
    ax.text(x+2, (y0+y1)/2, label, va='center', fontsize=fs, rotation=90)

def title(fig, part, sheet):
    fig.text(0.06, 0.965,
        f'OPEN RESEARCH INSTITUTE -- EVE / Hello Giggy | SEPTUM FEED 2304 MHz | {part} | sheet {sheet}/6 | v12.1 | dims mm',
        fontsize=9, weight='bold')

MAT = 'MATERIAL: 0.040" (1.016 mm) aluminum\n  3003-H14 or 5052-H32 acceptable'
TOL = 'TOLERANCE: +/-0.4 unless noted\n  holes DIA 3.40 THRU unless noted'

pdf = PdfPages('/mnt/user-data/outputs/septum_feed_drawing_set_v12.pdf')

# ================= SHEET 1: CLAMSHELL =================
fig = plt.figure(figsize=(11, 8.5)); title(fig, 'CLAMSHELL HALF', 1)
ax = fig.add_axes([0.06, 0.42, 0.55, 0.50]); ax.set_aspect('equal'); ax.axis('off')
lw = 1.4
xs = [-(wg_od/2+flange), -wg_od/2, -wg_od/2, wg_od/2, wg_od/2, wg_od/2+flange]
ys = [wall_o, wall_o, 0, 0, wall_o, wall_o]
ax.plot(xs, ys, 'k', lw=lw)
xi = [-(wg_od/2+flange), -(wg_od/2-t), -(wg_od/2-t), wg_od/2-t, wg_od/2-t, wg_od/2+flange]
yi = [wall_o-t, wall_o-t, t, t, wall_o-t, wall_o-t]
ax.plot(xi, yi, 'k', lw=lw)
ax.plot([-(wg_od/2+flange)]*2, [wall_o-t, wall_o], 'k', lw=lw)
ax.plot([(wg_od/2+flange)]*2, [wall_o-t, wall_o], 'k', lw=lw)
dim_h(ax, -(wg_od/2-t), wg_od/2-t, -12, '81.51 INSIDE *CRITICAL*')
dim_h(ax, -wg_od/2, wg_od/2, -22, '83.54 OUTSIDE')
dim_h(ax, -(wg_od/2+flange), wg_od/2+flange, 58, '113.54 ACROSS FLANGES')
dim_h(ax, wg_od/2, wg_od/2+flange, 48, '15.00')
dim_v(ax, -(wg_od/2+flange+10), t, wall_o, '40.25 INSIDE *CRITICAL*')
dim_v(ax, wg_od/2+flange+10, 0, wall_o, '41.26')
ax.text(0, wall_o/2, 'SECTION A-A (formed)\ninside bend R1.0 nom.\n(R0.8-1.6 OK if final\ndims held)', ha='center', va='center', fontsize=8)
ax.set_xlim(-95, 95); ax.set_ylim(-30, 70)
ax2 = fig.add_axes([0.06, 0.05, 0.88, 0.32]); ax2.set_aspect('equal'); ax2.axis('off')
ax2.add_patch(plt.Rectangle((0, 0), L, flange, fill=False, lw=1.2))
for x0 in (0, L-15):
    ax2.add_patch(plt.Rectangle((x0, 0), 15, flange, fill=False, ls='--', lw=0.8))
for z in np.arange(40, 335, 15):
    ax2.add_patch(plt.Circle((z, 7.5), 1.7, fill=False, lw=0.8))
ax2.text(L/2, flange+23, 'FLANGE RAIL (each of 2): 20x DIA 3.40 THRU, pitch 15.00, first 40.00 from FRONT edge, 7.50 from flange edge', ha='center', fontsize=8)
dim_h(ax2, 0, L, -10, '354.00', off=2)
dim_h(ax2, 0, 40, 22, '40.00', off=2)
ax2.text(7.5, -6, '15x15 corner\nnotches (4x)', fontsize=7, ha='center')
ax2.set_xlim(-15, 380); ax2.set_ylim(-18, 46)
ax3 = fig.add_axes([0.64, 0.42, 0.34, 0.50]); ax3.axis('off')
ax3.text(0, 1, f"""CLAMSHELL HALF -- QTY 2 (identical)

{MAT}

CRITICAL (formed):
 * 81.51 inside floor width +/-0.20
 * 40.25 inside wall depth  +/-0.20
 bends 90.0 +/-0.5 deg
{TOL}

FLOOR HOLES (per flat DXF):
 SMA: 1x DIA 4.40 + 4x DIA 2.30
  on 8.64 sq pattern (verified)
 cap: 2x DIA 3.40 at z=346.5,
  +/-20.38 off centerline
 collar: 2x DIA 3.40 at z=7.5,
  +/-26.77 off centerline

FLAT PATTERN: vendor develops
 own flat and bend allowances
 from formed dimensions on
 this sheet.

Deburr all. No scratches across
bend lines on convex faces.""", fontsize=7.4, family='monospace', va='top')
pdf.savefig(fig); plt.close(fig)


# ================= SHEET 2: CLAMSHELL FLAT PATTERN (REFERENCE) =================
fig = plt.figure(figsize=(11, 8.5)); title(fig, 'CLAMSHELL FLAT PATTERN (REFERENCE)', 2)
ax = fig.add_axes([0.05, 0.30, 0.9, 0.58]); ax.set_aspect('equal'); ax.axis('off')
W = 188.90                       # blank width at customer nominal BA (REF)
bends = [14.10, 53.57, 135.32, 174.79]   # customer nominal (REF)
yc2 = W/2                        # floor centerline == blank centerline (exact)
ax.add_patch(plt.Rectangle((0, 0), L, W, fill=False, lw=1.3))
for b in bends:
    ax.plot([0, L], [b, b], 'r', lw=0.8, ls='--')
ax.text(L+4, bends[0], 'B1', fontsize=7, color='r', va='center')
ax.text(L+4, bends[1], 'B2', fontsize=7, color='r', va='center')
ax.text(L+4, bends[2], 'B3', fontsize=7, color='r', va='center')
ax.text(L+4, bends[3], 'B4', fontsize=7, color='r', va='center')
ax.plot([0, L], [yc2, yc2], 'k', lw=0.5, ls='-.')
ax.text(L+4, yc2, 'CL', fontsize=7, va='center')
# corner notches 15x15 in both flange strips at both ends
for x0 in (0, L-15):
    for y0 in (0, W-15):
        ax.add_patch(plt.Rectangle((x0, y0), 15, 15, fill=False, ls='--', lw=0.8))
# rail holes both rails
for z in np.arange(40, 335, 15):
    ax.add_patch(plt.Circle((z, 7.5), 1.7, fill=False, lw=0.6))
    ax.add_patch(plt.Circle((z, W-7.5), 1.7, fill=False, lw=0.6))
# floor group: SMA (center + 4 corners), cap pair, collar pair
zs_sma = 354 - 24.19
ax.add_patch(plt.Circle((zs_sma, yc2), 2.2, fill=False, lw=0.8))
for sx in (-1, 1):
    for sy in (-1, 1):
        ax.add_patch(plt.Circle((zs_sma + sx*4.32, yc2 + sy*4.32), 1.15, fill=False, lw=0.6))
for sgn in (-1, 1):
    ax.add_patch(plt.Circle((346.5, yc2 + sgn*cap_hole_dy), 1.7, fill=False, lw=0.7))
    ax.add_patch(plt.Circle((7.5,   yc2 + sgn*collar_hole_dy), 1.7, fill=False, lw=0.7))
dim_h(ax, 0, L, -14, '354.00', off=3)
dim_v(ax, -14, 0, W, 'blank width per\nvendor development', off=3, fs=6.5)
dim_h(ax, 0, 40, W+12, '40.00', off=2, fs=7)
dim_h(ax, 340, zs_sma, W+12, '', off=2, fs=7)
ax.text(zs_sma, W+16, 'SMA ctr 24.19\nfrom REAR edge', fontsize=6.5, ha='center')
dim_v(ax, 346.5+16, yc2, yc2+cap_hole_dy, '20.38', off=2, fs=6.5)
dim_v(ax, 30, yc2, yc2+collar_hole_dy, '26.77', off=2, fs=6.5)
ax.set_xlim(-25, 395); ax.set_ylim(-28, 225)
ax3 = fig.add_axes([0.08, 0.03, 0.86, 0.22]); ax3.axis('off')
ax3.text(0, 1, f"""FLAT PATTERN -- REFERENCE ONLY. Formed dimensions (sheet 1) GOVERN; vendor develops own flat and bend allowances.
Bend lines B1-B4 shown SCHEMATICALLY -- positions and blank width per vendor development.
ALL HOLE POSITIONS are development-independent: lengthwise from cut edges (first rail hole 40.00 from
FRONT edge, pitch 15.00, 7.50 from long edges); floor group symmetric about blank centerline CL
(collar +/-26.77 at 7.50 from FRONT edge; cap +/-20.38 at 7.50 from REAR edge; SMA on CL, 24.19 from
REAR edge, 1x DIA 4.40 + 4x DIA 2.30 on 8.64 sq (verified)). Corner notches 15 x 15 (4x).
{TOL}""", fontsize=7.4, family='monospace', va='top')
pdf.savefig(fig); plt.close(fig)

# ================= SHEET 3: SEPTUM =================
fig = plt.figure(figsize=(11, 8.5)); title(fig, 'SEPTUM PLATE (FLAT -- DO NOT FORM)', 3)
ax = fig.add_axes([0.05, 0.30, 0.9, 0.58]); ax.set_aspect('equal'); ax.axis('off')
# draw: x = distance from REAR edge (right-to-left visual: put rear at right)
# outline in blank coords: x 0..339 front->rear? use front at left, rear at right.
FR = sept_L  # rear edge x
# rails top and bottom full length; web steps at rear.
# bottom rail: y -16.02..0 ; web y 0..h ; top rail y 81.51..97.53
yb0, yb1 = -ear, 0.0
yt0, yt1 = wg_id, wg_id + ear
# perimeter (simplified representative outline):
ax.plot([0, FR], [yb0, yb0], 'k', lw=1.2)             # bottom edge
ax.plot([0, FR], [yt1, yt1], 'k', lw=1.2)             # top edge
ax.plot([0, 0], [yb0, yb1], 'k', lw=1.2)              # front bottom rail end
ax.plot([0, 0], [yt0, yt1], 'k', lw=1.2)              # front top rail end
ax.plot([FR, FR], [yb0, yt1], 'k', lw=1.2)            # rear edge (full height)
ax.plot([0, 0], [yb1, yt0], 'k', lw=0.6, ls=':')      # front web absent marker
# rail inner edges forward of web
tipx = FR - 208.35
ax.plot([0, tipx], [0, 0], 'k', lw=1.2)
ax.plot([0, FR-83.19], [wg_id, wg_id], "k", lw=1.2)
# stepped web profile (top of web steps, rising toward rear) -- draw step
# boundary from tip: heights ascend rear-ward. Steps measured from rear.
prof_x, prof_y = [tipx], [0]
for (s0, s1, h) in reversed(steps):     # tip-side first
    x1, x0 = FR - s0, FR - s1           # blank coords
    prof_x += [x0, x0, x1]; prof_y += [prof_y[-1], h, h]
ax.plot(prof_x, prof_y, 'k', lw=1.2)
ax.plot([FR- s for s in [0]], [0], alpha=0)  # noop
# mirror note: steps rise from BOTTOM wall only (single-sided) -- correct.
# rear notches
for y0 in (yb0, yt1-15):
    ax.add_patch(plt.Rectangle((FR-15, y0), 15, 15, fill=False, ls='--', lw=0.8))
# rail holes
for z in np.arange(25, 320, 15):
    ax.add_patch(plt.Circle((z, yb0+7.5), 1.7, fill=False, lw=0.6))
    ax.add_patch(plt.Circle((z, yt1-7.5), 1.7, fill=False, lw=0.6))
dim_h(ax, 0, FR, yb0-14, '339.00', off=3)
dim_v(ax, -12, yb0, yt1, '113.54', off=3)
dim_v(ax, FR+12, 0, wg_id, '81.51 *CRITICAL*', off=3)
dim_h(ax, tipx, FR, yt1+12, '208.35 (stepped web)', off=3)
for (s0, s1, h) in steps:
    ax.text(FR-(s0+s1)/2, h+3 if h < 75 else h-6, f'{h:.2f}', fontsize=6.5, ha='center')
ax.set_xlim(-25, 385); ax.set_ylim(-40, 120)
ax3 = fig.add_axes([0.08, 0.03, 0.86, 0.22]); ax3.axis('off')
ax3.text(0, 1, f"""SEPTUM -- QTY 1. FLAT LASER PART, NO BENDS. Profile per supplied DXF (governs over this sketch).
{MAT}
CRITICAL: full-height web = 81.51 +0.00/-0.10.
Step heights from bottom rail: 10.41 / 23.18 / 39.21 / 63.96 / 81.51; boundaries from REAR edge: 208.35 / 164.36 / 130.61 / 95.34 / 83.19.
Rails 16.02 wide, full length. Holes: 2 rails x 20x DIA 3.40, pitch 15.00, first 25.00 from FRONT edge, 7.50 from rail edge.
Rear corner notches 15 x 15 (2x). {TOL}""", fontsize=7.4, family='monospace', va='top')
pdf.savefig(fig); plt.close(fig)

# ================= SHEET 3: BACKSHORT CAP =================
fig = plt.figure(figsize=(11, 8.5)); title(fig, 'BACKSHORT CAP (PAN)', 4)
ax = fig.add_axes([0.05, 0.25, 0.52, 0.62]); ax.set_aspect('equal'); ax.axis('off')
f = cap_face; fl = flange
# flat blank: plus shape
ax.add_patch(plt.Rectangle((fl, fl), f, f, fill=False, lw=1.3))
for (x0, y0, w, h) in [(fl, 0, f, fl), (fl, fl+f, f, fl), (0, fl, fl, f), (fl+f, fl, fl, f)]:
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, lw=1.1))
for s in (-1, 1):
    for (hx, hy) in [(fl+f/2+s*cap_hole_dy, 7.5), (fl+f/2+s*cap_hole_dy, 2*fl+f-7.5),
                     (7.5, fl+f/2+s*cap_hole_dy), (2*fl+f-7.5, fl+f/2+s*cap_hole_dy)]:
        ax.add_patch(plt.Circle((hx, hy), 1.7, fill=False, lw=0.7))
for (rx, ry) in [(fl, fl), (fl+f, fl), (fl, fl+f), (fl+f, fl+f)]:
    ax.add_patch(plt.Circle((rx, ry), 2.5, fill=False, lw=0.9))
dim_h(ax, fl, fl+f, -10, '84.14 FACE *CRITICAL* (see note)', off=3)
dim_h(ax, fl+f/2-cap_hole_dy, fl+f/2+cap_hole_dy, 2*fl+f+10, '2x 20.38 off ctr', off=2, fs=7)
dim_v(ax, -10, fl, fl+f, '84.14', off=3)
dim_v(ax, 2*fl+f+10, 0, fl, '15.00 skirt (4x)', off=2, fs=7)
ax.text(fl+f/2, fl+f/2, 'FLAT BLANK (plus shape)\nfold 4 skirts 90 deg\ncorners OPEN -- no seam', ha='center', fontsize=8)
ax.set_xlim(-22, 140); ax.set_ylim(-20, 140)
ax3 = fig.add_axes([0.60, 0.25, 0.37, 0.62]); ax3.axis('off')
ax3.text(0, 1, f"""BACKSHORT CAP -- QTY 1

{MAT}

FORM: pan, 4x 90 deg bends,
 skirt 15.00 deep, corners open.

CORNER RELIEF: 4x R2.5 circles at
 bend-line intersections --
 REQUIRED cut feature (perpen-
 dicular flange clearance),
 per cut DXF.

CRITICAL: inside face-to-face
 (between opposite skirts)
 84.14 +0.30/-0.00
 = slip fit over 83.54 tube.

HOLES: 8x DIA 3.40 (2 per skirt),
 7.50 from skirt edge,
 +/-20.38 about face center.

{TOL}
Deburr all.""", fontsize=7.6, family='monospace', va='top')
pdf.savefig(fig); plt.close(fig)

# ================= SHEET 4: FLARE PANEL =================
fig = plt.figure(figsize=(11, 8.5)); title(fig, 'FLARE PANEL', 5)
ax = fig.add_axes([0.05, 0.22, 0.55, 0.66]); ax.set_aspect('equal'); ax.axis('off')
ws, wl = wg_od, flare_od
poly_x = [0, 0, slant, slant, 0]
poly_y = [-ws/2, ws/2, wl/2, -wl/2, -ws/2]
ax.plot(poly_x, poly_y, 'k', lw=1.3)
ax.add_patch(plt.Rectangle((-flange, -ws/2), flange, ws, fill=False, lw=1.1))
ax.plot([0, 0], [-ws/2, ws/2], 'r', lw=0.8, ls='--')
for s in (-1, 1):
    ax.add_patch(plt.Circle((-flange/2, s*collar_hole_dy), 1.7, fill=False, lw=0.7))
dim_v(ax, -flange-12, -ws/2, ws/2, '83.54 (throat)', off=3, fs=7.5)
dim_v(ax, slant+12, -wl/2, wl/2, '184.20 (mouth)', off=3, fs=7.5)
dim_h(ax, 0, slant, -wl/2-14, '194.46 (slant)', off=3)
dim_h(ax, -flange, 0, wl/2+12, '15.00 tab', off=2, fs=7)
ax.text(slant/2, 0, 'trapezoid, symmetric\ntab bend 15 deg (dashed),\ntab folds toward viewer', ha='center', fontsize=8)
ax.set_xlim(-45, 245); ax.set_ylim(-115, 115)
ax3 = fig.add_axes([0.63, 0.22, 0.34, 0.66]); ax3.axis('off')
ax3.text(0, 1, f"""FLARE PANEL -- QTY 4 (identical)

{MAT}

FORM: single bend, 15.0 +/-1 deg,
 at tab/panel junction
 (dihedral 165 deg).

HOLES: 2x DIA 3.40 in tab,
 +/-26.77 about centerline,
 7.50 from tab edge.

CRITICAL: mouth width 184.20
 +/-0.4; symmetry about
 centerline +/-0.3.
{TOL}""", fontsize=7.6, family='monospace', va='top')
pdf.savefig(fig); plt.close(fig)

# ================= SHEET 5: CORNER BRACKET =================
fig = plt.figure(figsize=(11, 8.5)); title(fig, 'FLARE CORNER BRACKET', 6)
ax = fig.add_axes([0.05, 0.40, 0.9, 0.42]); ax.set_aspect('equal'); ax.axis('off')
ax.add_patch(plt.Rectangle((0, 0), slant, 25, fill=False, lw=1.3))
ax.plot([0, slant], [12.5, 12.5], 'r', lw=0.8, ls='--')
for x in np.arange(25, slant-24, 50):
    ax.add_patch(plt.Circle((x, 6.25), 1.7, fill=False, lw=0.7))
    ax.add_patch(plt.Circle((x, 18.75), 1.7, fill=False, lw=0.7))
dim_h(ax, 0, slant, -12, '194.46', off=3)
dim_h(ax, 0, 25, 34, '25.00 first hole', off=2, fs=7)
dim_h(ax, 25, 75, 42, '50.00 pitch', off=2, fs=7)
dim_v(ax, slant+12, 0, 25, '25.00', off=2, fs=7)
ax.text(slant/2, -24, 'bend 86.16 deg from flat along centerline (dashed) -- pyramid corner dihedral 93.84 deg', ha='center', fontsize=8)
ax.set_xlim(-15, 230); ax.set_ylim(-34, 52)
ax3 = fig.add_axes([0.08, 0.06, 0.86, 0.26]); ax3.axis('off')
ax3.text(0, 1, f"""CORNER BRACKET -- QTY 4 (identical)
{MAT}
FORM: single bend 86.2 +/-1.5 deg along centerline (12.50 from either edge). Loose tolerance -- bracket
HOLES: 6x DIA 3.40 (2 rows at 6.25 / 18.75 from edge; x = 25, 75, 125 from one end).
{TOL}""", fontsize=7.6, family='monospace', va='top')
pdf.savefig(fig); plt.close(fig)

pdf.close()
print('6-sheet drawing set written')

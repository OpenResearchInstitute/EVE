#!/usr/bin/env python3
"""DXF edition of the septum-feed drawing set -- six files, one per sheet.
Real DXF entities (LINE/CIRCLE/TEXT) on layers OUTLINE/HOLES/BEND/DIM/NOTES,
modelspace, 1 unit = 1 mm. Mirrors make_drawing_set.py content."""
import ezdxf, numpy as np, os

# ---- parameters (mirror septum_feed_2304_v12.scad, t = 1.016) ----
t = 1.016; wg_id = 81.51; sept_t = 1.016
wg_od = wg_id + 2*t; depth_i = (wg_id - sept_t)/2; wall_o = depth_i + t
flange = 15.0; L = 354.0
flare_od = 184.20; slant = 194.46
cap_face = 84.14; cap_hole_dy = wg_id/4; collar_hole_dy = wg_od/2 - flange
sept_L = 339.0; sept_W = 113.54; ear = 16.02
steps = [(0, 83.19, 81.51), (83.19, 95.34, 63.96), (95.34, 130.61, 39.21),
         (130.61, 164.36, 23.18), (164.36, 208.35, 10.41)]

OUT = '/mnt/user-data/outputs'

def newdoc():
    doc = ezdxf.new('R2010', setup=True)
    for name, color, lt in [('OUTLINE', 7, 'CONTINUOUS'), ('HOLES', 5, 'CONTINUOUS'),
                            ('BEND', 1, 'DASHED'), ('DIM', 3, 'CONTINUOUS'),
                            ('NOTES', 7, 'CONTINUOUS'), ('CL', 4, 'CENTER')]:
        doc.layers.add(name, color=color, linetype=lt)
    return doc, doc.modelspace()

def pl(ms, pts, layer='OUTLINE'):
    ms.add_lwpolyline(pts, dxfattribs={'layer': layer})

def circ(ms, c, r, layer='HOLES'):
    ms.add_circle(c, r, dxfattribs={'layer': layer})

def txt(ms, p, s, h=2.5, layer='NOTES', align='LEFT'):
    e = ms.add_text(s, dxfattribs={'layer': layer, 'height': h})
    e.set_placement(p)

def dimh(ms, x0, x1, y, label, h=2.5):
    ms.add_line((x0, y), (x1, y), dxfattribs={'layer': 'DIM'})
    for x in (x0, x1):
        ms.add_line((x, y-2), (x, y+2), dxfattribs={'layer': 'DIM'})
    txt(ms, ((x0+x1)/2 - len(label)*h*0.4, y+1.5), label, h, 'DIM')

def dimv(ms, x, y0, y1, label, h=2.5):
    ms.add_line((x, y0), (x, y1), dxfattribs={'layer': 'DIM'})
    for y in (y0, y1):
        ms.add_line((x-2, y), (x+2, y), dxfattribs={'layer': 'DIM'})
    txt(ms, (x+2, (y0+y1)/2), label, h, 'DIM')

def notes(ms, x, y, lines, h=2.8):
    for i, ln in enumerate(lines):
        txt(ms, (x, y - i*(h+1.6)), ln, h, 'NOTES')

MAT = ['MATERIAL: 0.040" (1.016 mm) ALUMINUM, 3003-H14 OR 5052-H32',
       'TOLERANCE: +/-0.4 UNLESS NOTED. HOLES DIA 3.40 THRU UNLESS NOTED.',
       'DEBURR ALL. NO SCRATCHES ACROSS BEND LINES ON CONVEX FACES.']

# ============ SHEET 1: CLAMSHELL (formed section + rail) ============
doc, ms = newdoc()
xs = [(-(wg_od/2+flange), wall_o), (-wg_od/2, wall_o), (-wg_od/2, 0),
      (wg_od/2, 0), (wg_od/2, wall_o), (wg_od/2+flange, wall_o)]
pl(ms, xs)
xi = [(-(wg_od/2+flange), wall_o-t), (-(wg_od/2-t), wall_o-t), (-(wg_od/2-t), t),
      (wg_od/2-t, t), (wg_od/2-t, wall_o-t), (wg_od/2+flange, wall_o-t)]
pl(ms, xi)
ms.add_line((-(wg_od/2+flange), wall_o-t), (-(wg_od/2+flange), wall_o), dxfattribs={'layer': 'OUTLINE'})
ms.add_line((wg_od/2+flange, wall_o-t), (wg_od/2+flange, wall_o), dxfattribs={'layer': 'OUTLINE'})
dimh(ms, -(wg_od/2-t), wg_od/2-t, -12, '81.51 INSIDE *CRITICAL* +/-0.20')
dimh(ms, -wg_od/2, wg_od/2, -22, '83.54 OUTSIDE')
dimh(ms, -(wg_od/2+flange), wg_od/2+flange, 58, '113.54 ACROSS FLANGES')
dimh(ms, wg_od/2, wg_od/2+flange, 48, '15.00')
dimv(ms, -(wg_od/2+flange+12), t, wall_o, '40.25 INSIDE *CRITICAL* +/-0.20')
dimv(ms, wg_od/2+flange+12, 0, wall_o, '41.26')
txt(ms, (-40, 20), 'SECTION A-A (FORMED). BENDS 90.0 +/-0.5 DEG.', 3)
txt(ms, (-40, 14), 'INSIDE BEND R1.0 NOM (R0.8-1.6 OK IF FINAL DIMS HELD)', 2.5)
# rail elevation below
oy = -120
pl(ms, [(0, oy), (L, oy), (L, oy+flange), (0, oy+flange), (0, oy)])
for x0 in (0, L-15):
    pl(ms, [(x0, oy), (x0+15, oy), (x0+15, oy+flange), (x0, oy+flange), (x0, oy)], 'BEND')
for z in np.arange(40, 335, 15):
    circ(ms, (z, oy+7.5), 1.7)
dimh(ms, 0, L, oy-10, '354.00')
dimh(ms, 0, 40, oy+flange+8, '40.00')
txt(ms, (0, oy+flange+16), 'FLANGE RAIL (EACH OF 2): 20X DIA 3.40 THRU, PITCH 15.00,', 2.8)
txt(ms, (0, oy+flange+11), 'FIRST 40.00 FROM FRONT EDGE, 7.50 FROM FLANGE EDGE. 15X15 CORNER NOTCHES (4X).', 2.8)
notes(ms, -110, -160, ['CLAMSHELL HALF -- QTY 2 (IDENTICAL)'] + MAT + [
 'FLOOR HOLES PER SHEET 2: SMA 1X DIA 4.40 + 4X DIA 2.30 ON 8.64 SQ (VERIFIED);',
 ' CAP 2X DIA 3.40; COLLAR 2X DIA 3.40.',
 'FLAT PATTERN: VENDOR DEVELOPS OWN FLAT AND BEND ALLOWANCES FROM FORMED DIMS.'])
doc.saveas(f'{OUT}/drawing_1_clamshell.dxf')

# ============ SHEET 2: CLAMSHELL FLAT (REFERENCE) ============
doc, ms = newdoc()
W = 188.90; yc = W/2
pl(ms, [(0, 0), (L, 0), (L, W), (0, W), (0, 0)])
for b in [14.10, 53.57, 135.32, 174.79]:
    ms.add_line((0, b), (L, b), dxfattribs={'layer': 'BEND'})
for i, b in enumerate([14.10, 53.57, 135.32, 174.79]):
    txt(ms, (L+4, b-1), f'B{i+1} (SCHEMATIC)', 2.2, 'BEND')
ms.add_line((0, yc), (L, yc), dxfattribs={'layer': 'CL'})
txt(ms, (L+4, yc-1), 'CL', 2.5, 'CL')
for x0 in (0, L-15):
    for y0 in (0, W-15):
        pl(ms, [(x0, y0), (x0+15, y0), (x0+15, y0+15), (x0, y0+15), (x0, y0)], 'BEND')
for z in np.arange(40, 335, 15):
    circ(ms, (z, 7.5), 1.7); circ(ms, (z, W-7.5), 1.7)
zs = 354 - 24.19
circ(ms, (zs, yc), 2.2)
for sx in (-1, 1):
    for sy in (-1, 1):
        circ(ms, (zs + sx*4.32, yc + sy*4.32), 1.15)
for sg in (-1, 1):
    circ(ms, (346.5, yc + sg*cap_hole_dy), 1.7)
    circ(ms, (7.5, yc + sg*collar_hole_dy), 1.7)
dimh(ms, 0, L, -12, '354.00')
dimh(ms, 0, 40, W+10, '40.00')
dimh(ms, zs, L, W+10, '24.19 (SMA CTR FROM REAR EDGE)')
dimv(ms, 30, yc, yc+collar_hole_dy, '26.77')
dimv(ms, 330, yc, yc+cap_hole_dy, '20.38')
notes(ms, 0, -24, ['CLAMSHELL FLAT -- REFERENCE ONLY. FORMED DIMS (SHEET 1) GOVERN.',
 'BEND LINES SCHEMATIC; POSITIONS AND BLANK WIDTH PER VENDOR DEVELOPMENT.',
 'ALL HOLES DEVELOPMENT-INDEPENDENT: LENGTHWISE FROM CUT EDGES; FLOOR GROUP',
 ' SYMMETRIC ABOUT CL. SMA: 1X DIA 4.40 + 4X DIA 2.30 ON 8.64 SQ (VERIFIED).',
 'COLLAR PAIR 7.50 FROM FRONT EDGE; CAP PAIR 7.50 FROM REAR EDGE.'] + MAT)
doc.saveas(f'{OUT}/drawing_2_clamshell_flat_ref.dxf')

# ============ SHEET 3: SEPTUM ============
doc, ms = newdoc()
FR = sept_L; yb0 = -ear; yt1 = wg_id + ear
pl(ms, [(0, yb0), (FR, yb0)]); pl(ms, [(0, yt1), (FR, yt1)])
pl(ms, [(0, yb0), (0, 0)]); pl(ms, [(0, wg_id), (0, yt1)])
pl(ms, [(FR, yb0), (FR, yt1)])
tipx = FR - 208.35
pl(ms, [(0, 0), (tipx, 0)])
pl(ms, [(0, wg_id), (FR-83.19, wg_id)])
prof = [(tipx, 0)]
for (s0, s1, h) in reversed(steps):
    x1, x0 = FR - s0, FR - s1
    prof += [(x0, prof[-1][1]), (x0, h), (x1, h)]
pl(ms, prof)
for y0 in (yb0, yt1-15):
    pl(ms, [(FR-15, y0), (FR, y0), (FR, y0+15), (FR-15, y0+15), (FR-15, y0)], 'BEND')
for z in np.arange(25, 320, 15):
    circ(ms, (z, yb0+7.5), 1.7); circ(ms, (z, yt1-7.5), 1.7)
dimh(ms, 0, FR, yb0-12, '339.00')
dimv(ms, -12, yb0, yt1, '113.54')
dimv(ms, FR+12, 0, wg_id, '81.51 *CRITICAL* +0.00/-0.10')
dimh(ms, tipx, FR, yt1+10, '208.35 STEPPED WEB')
for (s0, s1, h) in steps:
    txt(ms, (FR-(s0+s1)/2 - 8, min(h+3, 88)), f'{h:.2f}', 2.2, 'DIM')
notes(ms, 0, yb0-24, ['SEPTUM -- QTY 1. FLAT LASER PART. DO NOT FORM. PROFILE PER SUPPLIED CUT DXF (GOVERNS).',
 'STEP BOUNDARIES FROM REAR EDGE: 208.35 / 164.36 / 130.61 / 95.34 / 83.19.',
 'RAILS 16.02 WIDE FULL LENGTH. 2 RAILS X 20X DIA 3.40, PITCH 15.00, FIRST 25.00 FROM FRONT EDGE,',
 ' 7.50 FROM RAIL EDGE. REAR CORNER NOTCHES 15X15 (2X).'] + MAT)
doc.saveas(f'{OUT}/drawing_3_septum.dxf')

# ============ SHEET 4: BACKSHORT CAP ============
doc, ms = newdoc()
f = cap_face; fl = flange
pl(ms, [(fl, fl), (fl+f, fl), (fl+f, fl+f), (fl, fl+f), (fl, fl)])
for (x0, y0, w, h) in [(fl, 0, f, fl), (fl, fl+f, f, fl), (0, fl, fl, f), (fl+f, fl, fl, f)]:
    pl(ms, [(x0, y0), (x0+w, y0), (x0+w, y0+h), (x0, y0+h), (x0, y0)])
for s in (-1, 1):
    for (hx, hy) in [(fl+f/2+s*cap_hole_dy, 7.5), (fl+f/2+s*cap_hole_dy, 2*fl+f-7.5),
                     (7.5, fl+f/2+s*cap_hole_dy), (2*fl+f-7.5, fl+f/2+s*cap_hole_dy)]:
        circ(ms, (hx, hy), 1.7)
for (rx, ry) in [(fl, fl), (fl+f, fl), (fl, fl+f), (fl+f, fl+f)]:
    circ(ms, (rx, ry), 2.5)          # corner reliefs (required cut feature)
dimh(ms, fl, fl+f, -10, '84.14 FACE *CRITICAL* +0.30/-0.00 INSIDE')
dimv(ms, -10, fl, fl+f, '84.14')
dimv(ms, 2*fl+f+10, 0, fl, '15.00 SKIRT (4X)')
notes(ms, 0, -22, ['BACKSHORT CAP -- QTY 1. FORM: PAN, 4X 90 DEG BENDS, CORNERS OPEN (NO SEAM).',
 'CORNER RELIEF: 4X R2.5 CIRCLES AT BEND-LINE INTERSECTIONS -- REQUIRED CUT FEATURE.',
 'CRITICAL: INSIDE FACE-TO-FACE BETWEEN OPPOSITE SKIRTS 84.14 +0.30/-0.00.',
 'HOLES: 8X DIA 3.40 (2 PER SKIRT), 7.50 FROM SKIRT EDGE, +/-20.38 ABOUT FACE CENTER.'] + MAT)
doc.saveas(f'{OUT}/drawing_4_backshort_cap.dxf')

# ============ SHEET 5: FLARE PANEL ============
doc, ms = newdoc()
ws, wl = wg_od, flare_od
pl(ms, [(0, -ws/2), (0, ws/2), (slant, wl/2), (slant, -wl/2), (0, -ws/2)])
pl(ms, [(-flange, -ws/2), (0, -ws/2), (0, ws/2), (-flange, ws/2), (-flange, -ws/2)])
ms.add_line((0, -ws/2), (0, ws/2), dxfattribs={'layer': 'BEND'})
for s in (-1, 1):
    circ(ms, (-flange/2, s*collar_hole_dy), 1.7)
dimv(ms, -flange-12, -ws/2, ws/2, '83.54 THROAT')
dimv(ms, slant+12, -wl/2, wl/2, '184.20 MOUTH +/-0.4')
dimh(ms, 0, slant, -wl/2-12, '194.46 SLANT')
dimh(ms, -flange, 0, wl/2+10, '15.00 TAB')
notes(ms, 0, -wl/2-26, ['FLARE PANEL -- QTY 4 (IDENTICAL). SYMMETRIC TRAPEZOID +/-0.3 ABOUT CL.',
 'FORM: SINGLE BEND 15.0 +/-1 DEG AT TAB/PANEL JUNCTION (DASHED). DIHEDRAL 165 DEG.',
 'TAB HOLES: 2X DIA 3.40, +/-26.77 ABOUT CL, 7.50 FROM TAB EDGE. SLANT EDGES: NO HOLES.'] + MAT)
doc.saveas(f'{OUT}/drawing_5_flare_panel.dxf')

# ============ SHEET 6: CORNER BRACKET ============
doc, ms = newdoc()
pl(ms, [(0, 0), (slant, 0), (slant, 25), (0, 25), (0, 0)])
ms.add_line((0, 12.5), (slant, 12.5), dxfattribs={'layer': 'BEND'})
for x in np.arange(25, slant-24, 50):
    circ(ms, (x, 6.25), 1.7); circ(ms, (x, 18.75), 1.7)
dimh(ms, 0, slant, -10, '194.46')
dimh(ms, 0, 25, 32, '25.00 FIRST HOLE')
dimv(ms, slant+10, 0, 25, '25.00')
notes(ms, 0, -22, ['CORNER BRACKET -- QTY 4 (IDENTICAL).',
 'FORM: SINGLE BEND 86.2 +/-1.5 DEG ALONG CENTERLINE (12.50 FROM EITHER EDGE).',
 'HOLES: 6X DIA 3.40, 2 ROWS AT 6.25 / 18.75 FROM EDGE, X = 25 / 75 / 125 FROM ONE END.'] + MAT)
doc.saveas(f'{OUT}/drawing_6_corner_bracket.dxf')

print('6 DXF drawings written')

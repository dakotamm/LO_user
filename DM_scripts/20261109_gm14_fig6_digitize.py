"""
Digitize Geyer & MacCready (2014, Annu. Rev. Fluid Mech. 46:175-197) Figure 6, the Fr_f - M
estuarine parameter space, straight from the vector drawing commands in the PDF (page 18), so
the regime shading, boundary lines and estuary rectangles are exact rather than eyeballed.

AXES (from the figure's own tick marks, PDF points)
    M     log axis, ticks 0.2 / 0.5 / 1.0 / 2.0 at x = 228.2 / 327.4 / 402.4 / 477.2
          -> 249.1 pt per decade; frame spans M = 0.150 - 3.00
    Fr_f  log axis, 1e-4 ... 1e0 at y = 407.17 ... 678.42 -> 67.81 pt per decade

WHAT IS EXTRACTED
    fills   every filled polygon inside the frame (regime shading, incl. the transparency-flattening
            slivers where regimes overlap), with its CMYK colour converted to RGB
    lines   the dashed / dotted red regime boundaries
    rects   the 18 estuary rectangles (names assigned by position, checked against the rendered page)
    labels  regime label positions (read off the rendered page, approximate; for annotation only)

THE MIXING LINE (solid red diagonal)
    The figure's line is M = 1.36 Fr_f^(+1/6). Eq. 22 as printed, alpha^(1/2) Fr_f^(1/3) M^2 = 1,
    gives M ~ Fr_f^(-1/6) -- the wrong slope. Re-deriving from Eq. 13 with N_inf = N_0 (ds/s_ocn)^(1/2)
    and ds/s_ocn = alpha Fr_f^(2/3) (alpha = 3.4) gives alpha^(-1/2) Fr_f^(-1/3) M^2 = 1, i.e.
    M = alpha^(1/4) Fr_f^(1/6) = 1.358 Fr_f^(1/6), which matches the drawn line. So the printed Eq. 22
    has its exponent signs flipped; the figure is right.

INPUT  the GM14 PDF (EndNote library on the Mac; -pdf to override)
OUTPUT LO_output/DM_outs/20261109_gm14_fig6_digitize/
    gm14_fig6.p          dict: fills, lines, rects, labels, mixing-line parameters
    gm14_fig6_redraw.png the digitized figure redrawn in data coordinates, to check against the PDF

run 20261109_gm14_fig6_digitize.py
"""
import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pypdf
from pypdf.generic import ContentStream

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-pdf', type=str, default='/Users/dakotamascarenas/Desktop/My EndNote Library.Data'
                    '/PDF/2729488158/Geyer_MacCready_2014.pdf')
args = parser.parse_args()

PAGE = 17                                  # 0-based; printed page 192, Figure 6

# ------------------------------------------------------------------ axis calibration (PDF pt -> data)
X_TICK = {0.2: 228.2, 0.5: 327.4, 1.0: 402.4, 2.0: 477.2}
Y_TICK = {1e-4: 407.17, 1e-3: 474.98, 1e-2: 542.79, 1e-1: 610.59, 1e0: 678.42}
FRAME = dict(x0=197.1, x1=521.3, y0=407.17, y1=678.42)

# least-squares fit of log10(value) vs position over the ticks; residuals printed as a check
xa, xb = np.polyfit(list(X_TICK.values()), np.log10(list(X_TICK.keys())), 1)
ya, yb = np.polyfit(list(Y_TICK.values()), np.log10(list(Y_TICK.keys())), 1)
to_M = lambda x: 10 ** (xa * np.asarray(x) + xb)
to_F = lambda y: 10 ** (ya * np.asarray(y) + yb)
for v, x in X_TICK.items():
    print('M tick %.1f -> %.4f' % (v, to_M(x)))
for v, y in Y_TICK.items():
    print('Fr tick %.0e -> %.4e' % (v, to_F(y)))
M_LIM = (float(to_M(FRAME['x0'])), float(to_M(FRAME['x1'])))
F_LIM = (float(to_F(FRAME['y0'])), float(to_F(FRAME['y1'])))
print('frame: M %.3f - %.3f, Fr_f %.1e - %.1e' % (M_LIM + F_LIM))

ALPHA = 3.4                                # GM14 / Geyer (2010): ds/s_ocn = alpha Fr_f^(2/3)
MIX_C, MIX_P = ALPHA ** 0.25, 1 / 6        # mixing line M = MIX_C Fr_f^MIX_P (see docstring)

Ldir = Lfun.Lstart()
out_dir = Ldir['LOo'] / 'DM_outs' / '20261109_gm14_fig6_digitize'
out_dir.mkdir(parents=True, exist_ok=True)


def cmyk2rgb(c, m, y, k):
    return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))


def in_frame(pts, tol=2.0):
    p = np.asarray(pts)
    return ((p[:, 0] >= FRAME['x0'] - tol).all() and (p[:, 0] <= FRAME['x1'] + tol).all()
            and (p[:, 1] >= FRAME['y0'] - tol).all() and (p[:, 1] <= FRAME['y1'] + tol).all())


def to_data(pts):
    p = np.asarray(pts)
    return np.column_stack([to_M(p[:, 0]), to_F(p[:, 1])])


# ------------------------------------------------------------------ walk the page's drawing operators
reader = pypdf.PdfReader(args.pdf)
cs = ContentStream(reader.pages[PAGE].get_contents(), reader)

gs = dict(fill=(0, 0, 0, 1), stroke=(0, 0, 0, 1), dash=[], lw=1.0)
stack, subpaths, cur = [], [], None
fills, strokes, rects_raw = [], [], []
for ops, op in cs.operations:
    op = op.decode()
    if op == 'q':
        stack.append(dict(gs))
    elif op == 'Q':
        gs = stack.pop()
    elif op == 'k':
        gs['fill'] = tuple(float(v) for v in ops)
    elif op == 'K':
        gs['stroke'] = tuple(float(v) for v in ops)
    elif op == 'd':
        gs['dash'] = [float(v) for v in ops[0]]
    elif op == 'w':
        gs['lw'] = float(ops[0])
    elif op == 'm':
        cur = [(float(ops[0]), float(ops[1]))]
        subpaths.append(cur)
    elif op == 'l':
        cur.append((float(ops[0]), float(ops[1])))
    elif op in ('c', 'v', 'y'):                    # Bezier: keep the end point (curves here are ~straight)
        cur.append((float(ops[-2]), float(ops[-1])))
    elif op == 're':
        x, y, w, h = (float(v) for v in ops)
        rects_raw.append((x, y, w, h, gs['stroke']))
        subpaths.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
    elif op in ('f', 'f*', 'S', 's', 'B', 'b', 'n'):
        for sp in subpaths:
            if len(sp) < 2 or not in_frame(sp):
                continue
            if op in ('f', 'f*', 'B', 'b'):
                if len(sp) >= 3:
                    fills.append(dict(xy=to_data(sp), rgb=cmyk2rgb(*gs['fill'])))
            elif op in ('S', 's') and gs['stroke'] != (0, 0, 0, 1):   # black = frame and ticks
                strokes.append(dict(xy=to_data(sp), rgb=cmyk2rgb(*gs['stroke']),
                                    dash=list(gs['dash']), lw=gs['lw'], cmyk=gs['stroke']))
        subpaths, cur = [], None
print('%d fills, %d coloured strokes, %d re-rectangles' % (len(fills), len(strokes), len(rects_raw)))

# ------------------------------------------------------------------ estuary rectangles, boundaries, mixing line
# approximate (M, Fr_f) box centres read off the rendered page, used only to attach names
NAMES = {'Hudson River': (0.63, 2.7e-2), 'San Francisco Bay': (1.0, 1.5e-2), 'Willapa Bay': (1.08, 1.2e-3),
         'Mississippi River': (0.85, 4.5e-1), 'Ebro River': (0.29, 1.5e-1), 'James River': (0.73, 6.7e-3),
         'Chesapeake Bay': (0.41, 4.4e-3), 'Narragansett Bay': (0.62, 1.2e-3),
         'Long Island Sound': (0.25, 5.4e-4), 'Puget Sound': (0.22, 1.9e-4), 'Tamar River': (1.2, 1.4e-2),
         'Columbia River': (1.4, 7.1e-2), 'Amazon River': (0.99, 2.2e-1), 'Chang Jiang River': (1.22, 2.0e-1),
         'Merrimack River': (1.54, 3.1e-1), 'Fraser River': (1.27, 3.9e-1), 'Conwy River': (1.48, 3.1e-2),
         'Baltic Sea': (0.26, 1.6e-3)}
CLASS = {(0.75, 0.0, 1.0, 0.25): 'strongly stratified', (0.5, 0.0, 1.0, 0.0): 'partially mixed',
         (0.0, 0.15, 1.0, 0.43): 'SIPS', (0.0, 1.0, 1.0, 0.0): 'salt wedge',
         (0.0, 0.53, 0.3, 0.0): 'time-dependent salt wedge', (1.0, 0.4, 0.0, 0.0): 'fjord / bay'}

rects, lines = {}, []
for s in strokes:
    xy = s['xy']
    closed_box = len(xy) == 5 and np.allclose(xy[0], xy[-1]) and len(np.unique(np.round(xy[:, 0], 6))) == 2
    if closed_box:
        m0, m1 = xy[:, 0].min(), xy[:, 0].max()
        f0, f1 = xy[:, 1].min(), xy[:, 1].max()
        c = np.array([np.log10(np.sqrt(m0 * m1)), np.log10(np.sqrt(f0 * f1))])
        name = min(NAMES, key=lambda n: np.hypot(*(c - np.log10(NAMES[n]))))
        assert name not in rects, 'two boxes matched ' + name
        key = tuple(round(v, 2) for v in s['cmyk'])
        rects[name] = dict(M=(m0, m1), Fr=(f0, f1), rgb=s['rgb'], cls=CLASS.get(key, '?'))
    elif s['lw'] > 0.9 and s['rgb'][0] > 0.9 and s['rgb'][1] < 0.2:   # red boundaries (dashed / dotted)
        lines.append(dict(xy=xy, rgb=s['rgb'], dash=s['dash'], lw=s['lw']))
assert len(rects) == 18, len(rects)
print('\n%-20s %-26s %-15s %s' % ('estuary', 'class (box colour)', 'M', 'Fr_f'))
for n, r in sorted(rects.items(), key=lambda kv: kv[1]['M'][0]):
    print('%-20s %-26s %.3f-%.3f    %.1e-%.1e' % ((n, r['cls']) + r['M'] + r['Fr']))
print('\n%d boundary lines:' % len(lines))
for ln in lines:
    print('   dash %-12s M %.3f -> %.3f   Fr %.1e -> %.1e' % (ln['dash'], ln['xy'][0, 0], ln['xy'][-1, 0],
                                                          ln['xy'][0, 1], ln['xy'][-1, 1]))

# the drawn mixing line is a thin filled red quadrilateral; its long axis is the line
red = [f for f in fills if np.allclose(f['rgb'], (1, 0, 0)) and len(f['xy']) == 5]
assert len(red) == 1, len(red)
q = red[0]['xy']
pM, cM = np.polyfit(np.log10(q[:, 1]), np.log10(q[:, 0]), 1)
print('\ndrawn mixing line: M = %.3f Fr_f^%.4f   (alpha^(1/4) Fr_f^(1/6) = %.3f Fr_f^%.4f)'
      % (10 ** cM, pM, MIX_C, MIX_P))

# regime labels, positions read off the rendered page (for annotation only)
LABELS = {'Salt wedge': (0.50, 0.29), 'Time-dependent\nsalt wedge': (2.4, 0.54),
          'Strongly\nstratified': (0.31, 0.024), 'Partially\nmixed': (1.0, 0.053),
          'Fjord': (0.18, 7.5e-4), 'Bay': (0.49, 3.3e-4), 'SIPS': (0.80, 4.4e-4),
          'Well\nmixed': (1.62, 8.6e-4), 'Bu': (1.36, 2.0e-3)}

fig6 = dict(fills=fills, lines=lines, rects=rects, labels=LABELS, M_lim=M_LIM, F_lim=F_LIM,
            alpha=ALPHA, mix_c=MIX_C, mix_p=MIX_P, mix_drawn=(10 ** cM, pM), src=args.pdf)
pickle.dump(fig6, open(out_dir / 'gm14_fig6.p', 'wb'))
print('saved ' + str(out_dir / 'gm14_fig6.p'))

# ------------------------------------------------------------------ check figure: redraw in data coordinates
fig, ax = plt.subplots(figsize=(7, 6.5))
for f in fills:
    ax.fill(f['xy'][:, 0], f['xy'][:, 1], color=f['rgb'], lw=0)
for ln in lines:
    ax.plot(ln['xy'][:, 0], ln['xy'][:, 1], color=ln['rgb'], lw=1.5,
            ls=(0, tuple(ln['dash'])) if ln['dash'] else '-')
ff = np.logspace(-4, 0, 50)
ax.plot(MIX_C * ff ** MIX_P, ff, color='r', lw=1.2, label=r'$M=\alpha^{1/4}Fr_f^{1/6}$')
for n, r in rects.items():
    (m0, m1), (f0, f1) = r['M'], r['Fr']
    ax.add_patch(plt.Rectangle((m0, f0), m1 - m0, f1 - f0, fill=False, ec=r['rgb'], lw=1.2))
    ax.text(np.sqrt(m0 * m1), np.sqrt(f0 * f1), n.replace(' ', '\n', 1), ha='center', va='center',
            fontsize=6, color=r['rgb'])
for t, (m, f) in LABELS.items():
    ax.text(m, f, t, ha='center', va='center', fontsize=8, weight='bold', color='0.25')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(M_LIM)
ax.set_ylim(F_LIM)
ax.set_xticks([0.2, 0.5, 1, 2])
ax.set_xticklabels(['0.2', '0.5', '1.0', '2.0'])
ax.set_xlabel('M')
ax.set_ylabel(r'$Fr_f$')
ax.legend(loc='lower right', fontsize=8)
ax.set_title('GM14 Fig. 6, digitized from PDF vectors', fontsize=10)
fig.tight_layout()
fig.savefig(out_dir / 'gm14_fig6_redraw.png', dpi=200, transparent=True)
plt.close(fig)
print('saved ' + str(out_dir / 'gm14_fig6_redraw.png'))

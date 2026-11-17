"""
TEF at pc_lj (Lovejoy), wb1_t0_xn11abbur00, 2024-2025: volume, salt and oxygen transport.
Built step by step.

STEP 1  load the bulk TEF output for pc_lj (daily, Godin-filtered, multi-layer) and form the
        two-layer quantities, + = into the cove:
            Q_in  = sum of layers with q > 0        Q_out = sum of layers with q < 0
            F_in  = sum_in q s                      s_in  = F_in / Q_in      (salt)
            G_in  = sum_in q O                      O_in  = G_in / Q_in      (oxygen)
        and the same for out; net = in + out. Kept for YEAR (2024) only. As LO
        tef_fun.get_two_layer, but WITHOUT its "Q < mean/100 -> NaN" mask, so no days are dropped.
        -bin salt    bulk_avg     layers are salinity classes (standard TEF)      (default)
        -bin oxygen  bulk_avg_DO  layers are oxygen classes
        Both files are in the raw section frame (+ = east = out of the cove), so SIGN = -1.

STEP 2  daily time series, 3 x 3: columns volume / salt / oxygen; rows
            gross in and out (signed)      Q_in, Q_out      F_in, F_out      G_in, G_out
            what the layers carry          qprism           s_in, s_out      O_in, O_out
            net                            Q_net (+ qnet)   F_net            G_net
        The daily bulk values are plotted as they are (already Godin-filtered, one per day).

Units: Q m3/s; salt g/kg, salt flux g/kg m3/s; oxygen mmol m-3 (x 0.032 = mg/L), oxygen flux
mmol/s (x 0.032 = g/s).

run 20261116_pc_lj_tef.py
run 20261116_pc_lj_tef.py -bin oxygen
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-bin', choices=['salt', 'oxygen'], default='salt')
args = parser.parse_args()

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SN = 'pc_lj'
YEAR = 2024              # the bulk file covers 2024-2025; keep one year
SIGN = -1                 # section-positive (east) is out of the cove
K = 0.032                 # mmol O2 -> g, and mmol m-3 -> mg/L

Ldir = Lfun.Lstart(gridname='wb1')
tef_dir = Ldir['LOo'] / 'extract' / GTAGEX / 'tef2'
bulk_name = {'salt': 'bulk_avg_', 'oxygen': 'bulk_avg_DO_'}[args.bin] + DS0 + '_' + DS1
out_dir = Ldir['LOo'] / 'DM_outs' / '20261116_pc_lj_tef'
out_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ STEP 1 load
ds = xr.open_dataset(tef_dir / bulk_name / (SN + '.nc'))
time = pd.to_datetime(ds.time.values)
q = SIGN * ds.q.to_numpy()                # (t, layer), NaN-padded past the last layer
s = ds.salt.to_numpy()
O = ds.oxygen.to_numpy()
qnet = SIGN * ds.qnet.to_numpy()          # Godin(section-total q)
qprism = ds.qprism.to_numpy()
ds.close()

qin = np.where(q > 0, q, 0.)
qout = np.where(q < 0, q, 0.)

tef = pd.DataFrame(index=time)
tef['Q_in'] = qin.sum(axis=1)
tef['Q_out'] = qout.sum(axis=1)
tef['F_in'] = np.nansum(qin * s, axis=1)
tef['F_out'] = np.nansum(qout * s, axis=1)
tef['G_in'] = np.nansum(qin * O, axis=1)
tef['G_out'] = np.nansum(qout * O, axis=1)
for v in ['Q', 'F', 'G']:
    tef[v + '_net'] = tef[v + '_in'] + tef[v + '_out']
tef['s_in'], tef['s_out'] = tef.F_in / tef.Q_in, tef.F_out / tef.Q_out
tef['O_in'], tef['O_out'] = tef.G_in / tef.Q_in, tef.G_out / tef.Q_out
tef['qnet'] = qnet
tef['qprism'] = qprism
tef = tef[tef.index.year == YEAR]

print('%s  %s-binned  %d days  %s to %s' % (SN, args.bin, len(tef), tef.index[0].date(), tef.index[-1].date()))
print('%d means (+ = into cove):' % YEAR)
print('  Q_in %7.1f   Q_out %7.1f   Q_net %6.2f   (qnet %6.2f)   qprism %7.1f  m3/s'
      % (tef.Q_in.mean(), tef.Q_out.mean(), tef.Q_net.mean(), tef.qnet.mean(), tef.qprism.mean()))
print('  s_in %7.3f   s_out %7.3f   F_net %7.2f  g/kg m3/s'
      % (tef.s_in.mean(), tef.s_out.mean(), tef.F_net.mean()))
print('  O_in %7.2f   O_out %7.2f  mg/L    G_net %7.1f  g/s'
      % (K * tef.O_in.mean(), K * tef.O_out.mean(), K * tef.G_net.mean()))


# ------------------------------------------------------------------ STEP 2 time series
C_IN, C_OUT, C_NET = 'tab:red', 'tab:blue', 'k'
t = tef.index

fig, axs = plt.subplots(3, 3, figsize=(16, 9), sharex=True, layout='constrained')

# volume
ax = axs[0, 0]
ax.plot(t, tef.Q_in, color=C_IN, lw=0.8, label='$Q_{in}$')
ax.plot(t, tef.Q_out, color=C_OUT, lw=0.8, label='$Q_{out}$')
ax.set_ylabel('m$^3$ s$^{-1}$')
ax.set_title('Volume')
ax = axs[1, 0]
ax.plot(t, tef.qprism, color='0.4', lw=0.8, label='$Q_{prism}$')
ax.set_ylabel('m$^3$ s$^{-1}$')
ax = axs[2, 0]
ax.plot(t, tef.Q_net, color=C_NET, lw=0.8, label='$Q_{net}$')
ax.plot(t, tef.qnet, color='tab:orange', lw=0.6, ls='--', label='qnet (bulk file)')
ax.set_ylabel('m$^3$ s$^{-1}$')

# salt
ax = axs[0, 1]
ax.plot(t, tef.F_in, color=C_IN, lw=0.8, label='$F_{in}$')
ax.plot(t, tef.F_out, color=C_OUT, lw=0.8, label='$F_{out}$')
ax.set_ylabel('g kg$^{-1}$ m$^3$ s$^{-1}$')
ax.set_title('Salt')
ax = axs[1, 1]
ax.plot(t, tef.s_in, color=C_IN, lw=0.8, label='$s_{in}$')
ax.plot(t, tef.s_out, color=C_OUT, lw=0.8, label='$s_{out}$')
ax.set_ylabel('g kg$^{-1}$')
ax = axs[2, 1]
ax.plot(t, tef.F_net, color=C_NET, lw=0.8, label='$F_{net}$')
ax.set_ylabel('g kg$^{-1}$ m$^3$ s$^{-1}$')

# oxygen
ax = axs[0, 2]
ax.plot(t, K * tef.G_in, color=C_IN, lw=0.8, label='$G_{in}$')
ax.plot(t, K * tef.G_out, color=C_OUT, lw=0.8, label='$G_{out}$')
ax.set_ylabel('g O$_2$ s$^{-1}$')
ax.set_title('Oxygen')
ax = axs[1, 2]
ax.plot(t, K * tef.O_in, color=C_IN, lw=0.8, label='$O_{in}$')
ax.plot(t, K * tef.O_out, color=C_OUT, lw=0.8, label='$O_{out}$')
ax.set_ylabel('mg L$^{-1}$')
ax = axs[2, 2]
ax.plot(t, K * tef.G_net, color=C_NET, lw=0.8, label='$G_{net}$')
ax.set_ylabel('g O$_2$ s$^{-1}$')

for ax in axs.flat:
    ax.grid(True, color='0.85', lw=0.5)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
for ax in list(axs[0, :]) + list(axs[2, :]):
    ax.axhline(0, color='0.4', lw=0.5)
loc = mdates.AutoDateLocator()
for ax in axs[-1, :]:
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
fig.suptitle('%s TEF, %s-binned, %s, %d   (+ = into Penn Cove)' % (SN, args.bin, GTAGEX, YEAR))

fig.savefig(out_dir / ('%s_tef_ts_%s_%d.png' % (SN, args.bin, YEAR)), dpi=200, transparent=True)
plt.close(fig)
print('saved', out_dir / ('%s_tef_ts_%s_%d.png' % (SN, args.bin, YEAR)))

"""
Plot bulk fluxes in OXYGEN coordinates as a time series, from bulk_avg_DO_*.

This is bulk_plot_avg.py pointed at the oxygen-coordinate bulk files. Note that
tef_fun.get_two_layer needs no changes at all: it splits the layers on the sign
of q and flux weights every other layer variable it finds, so it does not care
what coordinate the division was made in. It returns oxygen_p / oxygen_m here
in place of salt_p / salt_m.

Output: bulk_avg_DO_plots_[ds0]_[ds1]/[sn].png

WHAT THE THIRD PANEL IS
bulk_plot_avg.py's third panel is Qin*DS, and it picks which limb is "in" by
taking the one with the higher mean salinity. That heuristic is meaningful for
salt -- the saltier limb is the one coming from the ocean -- and meaningless
for oxygen, where the more oxygenated limb is just the one nearer the surface.
So it is not ported. The panel instead shows two things that need no in/out
call at all, both in mol s-1:

    F_net   = sum over layers of q*O, the net oxygen flux through the section
    F_exch  = Q_p*(O_p - O_m), the part the exchange circulation carries

F_net is summed over the multi-layer bulk values rather than rebuilt from the
two-layer collapse, so it does not inherit get_two_layer's small-transport
masking.

SIGN CONVENTION
Raw section frame, positive in the section's own direction (pm = +1), as in
bulk_avg_DO_*. For the wb1 pc sections that points OUT of Penn Cove, so red
(positive) is the EASTWARD limb and a positive F_net is oxygen leaving the
cove. The red '+' on the map shows the positive direction. The INFLOW_SIGN flip
lives in DM_scripts/20260916_exchange_fun.py, not here.

FIGURES ARE SAVED, NEVER SHOWN
bulk_plot_avg.py calls plt.show() when testing. This one always writes a png.

To test on mac (first section only):
run bulk_plot_avg_DO.py -gtx wb1_t0_xn11abbur00 -ctag pc1 -0 2024.01.01 -1 2025.12.31 -test True

For real:
run bulk_plot_avg_DO.py -gtx wb1_t0_xn11abbur00 -ctag pc1 -0 2024.01.01 -1 2025.12.31
"""
import sys
import matplotlib
matplotlib.use('Agg')  # figures are saved, never shown
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun
import extract.tef2.archive.tef_fun as tef_fun

from lo_tools import extract_argfun as exfun
Ldir = exfun.intro() # this handles the argument passing

gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'

sect_df_fn = tef2_dir / ('sect_df_' + gctag + '.p')
sect_df = pd.read_pickle(sect_df_fn)

out_dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
in_dir = out_dir0 / ('bulk_avg_DO_' + Ldir['ds0'] + '_' + Ldir['ds1'])
out_dir = out_dir0 / ('bulk_avg_DO_plots_' + Ldir['ds0'] + '_' + Ldir['ds1'])

if not in_dir.is_dir():
    print('ERROR: no ' + str(in_dir))
    print('Run process_sections_avg_DO.py then bulk_calc_avg_DO.py first.')
    sys.exit(1)

# clean only on a real run, so a -test pass does not wipe a full set of plots
Lfun.make_dir(out_dir, clean=not Ldir['testing'])

sect_list = [item.name.replace('.nc','') for item in in_dir.glob('*.nc')]
sect_list.sort()
if Ldir['testing']:
    sect_list = sect_list[:1]
    print('testing: only ' + sect_list[0])

# grid info
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
h = g.h.values
h[g.mask_rho.values==0] = np.nan
xrho = g.lon_rho.values
yrho = g.lat_rho.values
xp, yp = pfun.get_plon_plat(xrho,yrho)
xu = g.lon_u.values
yu = g.lat_u.values
xv = g.lon_v.values
yv = g.lat_v.values

# PLOTTING
fs = 12
plt.close('all')
if Ldir['testing']:
    figsize = (12,8)
else:
    figsize = (21,12)
pfun.start_plot(fs=fs, figsize=figsize)

for sect_name in sect_list:

    bulk = xr.open_dataset(in_dir / (sect_name + '.nc'))
    O_units = bulk.attrs.get('oxygen_units', 'mmol m-3')

    tef_df, vn_list, vec_list = tef_fun.get_two_layer(in_dir, sect_name)

    # adjust units
    tef_df['Q_p'] = tef_df['q_p']/1000
    tef_df['Q_m'] = tef_df['q_m']/1000
    tef_df['Q_prism'] = tef_df['qprism']/1000

    # labels and colors
    ylab_dict = {'Q': r'Transport $[10^{3}\ m^{3}s^{-1}]$',
                'oxygen': r'Oxygen $[mmol\ m^{-3}]$',
                'F': r'Oxygen flux $[mol\ s^{-1}]$'}
    p_color = 'r'
    m_color = 'b'
    lw = 2

    fig = plt.figure()

    ax1 = plt.subplot2grid((3,3), (0,0), colspan=2) # Q+, Q-
    ax2 = plt.subplot2grid((3,3), (1,0), colspan=2) # O+, O-
    ax3 = plt.subplot2grid((3,3), (2,0), colspan=2) # oxygen flux and Qprism
    axmap = plt.subplot2grid((1,3), (0,2)) # map

    ot = bulk.time.values

    def add_qprism(ax):
        # add Qprism
        axqp = ax.twinx()
        axqp.plot(ot,tef_df['Q_prism'].to_numpy(),'-',
            color='c',linewidth=3,alpha=.4)
        axqp.text(.95,.9,r'$Q_{prism}\ [10^{3}\ m^{3}s^{-1}]$', color='c',
            transform=ax.transAxes, ha='right',
            bbox=pfun.bbox)
        axqp.set_ylim(bottom=0)
        axqp.xaxis.label.set_color('c')
        axqp.tick_params(axis='y', colors='c')

    # ---------------------------------------------------------------- panel 1
    ax1.plot(ot,tef_df['Q_p'].to_numpy(), color=p_color, linewidth=lw, zorder=5)
    ax1.plot(ot,tef_df['Q_m'].to_numpy(), color=m_color, linewidth=lw, zorder=5)
    ax1.grid(True)
    ax1.set_ylabel(ylab_dict['Q'])
    ax1.set_xlim(ot[0],ot[-1])

    qp = bulk['q'].values/1000
    qp[qp<0] = np.nan
    qm = bulk['q'].values/1000
    qm[qm>0]=np.nan
    op = bulk['oxygen'].values.copy()
    op[np.isnan(qp)] = np.nan
    om = bulk['oxygen'].values.copy()
    om[np.isnan(qm)]=np.nan

    # The multi-layer dots are the raw division; the heavy lines are the
    # two-layer collapse. In the salinity version the layers cluster tightly
    # and big dots are fine. Oxygen layers spread over the whole 0-430 range,
    # so the dots have to be small and faint or they bury the lines.
    alpha = .12
    ms = 2
    ax1.plot(ot,qp,'o',color=p_color,alpha=alpha,ms=ms,mew=0,zorder=1)
    ax1.plot(ot,qm,'o',color=m_color,alpha=alpha,ms=ms,mew=0,zorder=1)

    # ---------------------------------------------------------------- panel 2
    ax2.plot(ot,op,'o',color=p_color,alpha=alpha,ms=ms,mew=0,zorder=1)
    ax2.plot(ot,om,'o',color=m_color,alpha=alpha,ms=ms,mew=0,zorder=1)

    ax2.plot(ot,tef_df['oxygen_p'].to_numpy(), color=p_color, linewidth=lw,
             zorder=5, path_effects=[pe.Stroke(linewidth=lw+1.5, foreground='w'), pe.Normal()])
    ax2.plot(ot,tef_df['oxygen_m'].to_numpy(), color=m_color, linewidth=lw,
             zorder=5, path_effects=[pe.Stroke(linewidth=lw+1.5, foreground='w'), pe.Normal()])
    ax2.grid(True)
    ax2.set_ylabel(ylab_dict['oxygen'])
    ax2.set_xlim(ot[0],ot[-1])

    # ---------------------------------------------------------------- panel 3
    # net oxygen flux, summed over the multi-layer bulk values [mol s-1]
    F_net = np.nansum(bulk['q'].values * bulk['oxygen'].values, axis=1)/1000
    # the part the exchange circulation carries [mol s-1]
    F_exch = (tef_df['q_p'].to_numpy()
              * (tef_df['oxygen_p'].to_numpy() - tef_df['oxygen_m'].to_numpy()))/1000

    ax3.plot(ot, F_net, color='k', linewidth=lw, label=r'$F_{net}$')
    ax3.plot(ot, F_exch, color='0.5', linewidth=lw, linestyle='--',
             label=r'$Q_{+}\Delta O$')
    ax3.axhline(0, color='k', linewidth=.8, alpha=.5)
    ax3.grid(True)
    ax3.set_ylabel(ylab_dict['F'])
    ax3.set_xlim(ot[0],ot[-1])
    ax3.legend(loc='upper left', framealpha=.6)
    add_qprism(ax3)
    ax3.text(.05,.05,'positive is the section + direction', transform=ax3.transAxes,
             bbox=pfun.bbox)

    # -------------------------------------------------------------------- map
    sn = sect_name.replace('.p','')
    sinfo = sect_df.loc[sect_df.sn==sn,:]
    i0 = sinfo.iloc[0,:].i
    j0 = sinfo.iloc[0,:].j
    uv0 = sinfo.iloc[0,:].uv
    i1 = sinfo.iloc[-1,:].i
    j1 = sinfo.iloc[-1,:].j
    uv1 = sinfo.iloc[-1,:].uv
    if uv0=='u':
        x0 = xu[j0,i0]
        y0 = yu[j0,i0]
    elif uv0=='v':
        x0 = xv[j0,i0]
        y0 = yv[j0,i0]
    if uv1=='u':
        x1 = xu[j1,i1]
        y1 = yu[j1,i1]
    elif uv1=='v':
        x1 = xv[j1,i1]
        y1 = yv[j1,i1]
    axmap.plot([x0,x1],[y0,y1],'-c', lw=3)
    axmap.plot(x0,y0,'og', ms=10)
    pfun.add_coast(axmap)
    pfun.dar(axmap)
    axmap.pcolormesh(xp, yp, -h, vmin=-100, vmax=100,
        cmap='jet', alpha=.4)

    dx = x1-x0; dy = y1-y0
    xmid = x0 + dx/2; ymid = y0 + dy/2
    pad = np.max((np.sqrt(dx**2 + dy**2)*2,.1))
    axmap.axis([x0-pad, x1+pad, y0-pad, y1+pad])
    axmap.xaxis.set_major_locator(MaxNLocator(4))
    axmap.yaxis.set_major_locator(MaxNLocator(5))
    axmap.set_xlabel('Longitude [deg]')
    axmap.set_ylabel('Latitude [deg]')
    axmap.set_title(sn + ' (oxygen coordinates)')
    # indicate which direction is positive with a red plus
    xpos = xmid - dy/2
    ypos = ymid + dx/2
    axmap.text(xpos, ypos, '+', fontweight='bold', c='r', fontsize=20,
        ha='center',va='center')

    bulk.close()

    fig_fn = out_dir / (sect_name.replace('.p','') + '.png')
    plt.savefig(fig_fn, transparent=True)
    plt.close()
    print('saved ' + str(fig_fn))
    sys.stdout.flush()

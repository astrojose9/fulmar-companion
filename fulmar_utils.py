#!/usr/bin/env python3
"""
fulmar_utils.py — helper functions for FULMAR.
Imported by fulmar.py.
"""

import json
import os
import warnings

import astropy.units as u
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import wotan as wt
import yaml

from astropy.coordinates import Angle, SkyCoord
from astropy.io.ascii import convert_numpy
from astropy.stats import mad_std
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.timeseries import TimeSeries

from astroquery.gaia import Gaia
from astroquery.mast import Catalogs

from collections import OrderedDict

from ldtk import LDPSetCreator, SVOFilter

from transitleastsquares import transit_mask, transitleastsquares  # Talens fork: https://github.com/talensgj/tls

from pytransit import BaseLPF


# ---- JSON encoder for numpy types -------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# ---- Time utilities ---------------------------------------------------------

def to_btjd(t):
    """
    Convert a date string to BTJD (BJD - 2457000, scale='tdb').
    Accepts BTJD float string, BJD float string, ISO/ISOT UTC string, or None.
    Returns None if t is None.
    """
    if t is None:
        return None
    try:
        val = float(t)
        if val > 2457000:
            return Time(val, format='jd', scale='tdb').btjd
        else:
            return val
    except (ValueError, TypeError):
        return Time(t, scale='utc').tdb.btjd


def sectors_in_date_range(lc_col, date_start, date_end):
    """
    Return indices into lc_col for sectors overlapping the given date range.
    date_start or date_end can be None to indicate no lower/upper bound.
    """
    t_start = to_btjd(date_start)
    t_end   = to_btjd(date_end)
    return [i for i, lc in enumerate(lc_col)
            if (t_start is None or lc.time.btjd.max() >= t_start)
            and (t_end   is None or lc.time.btjd.min() <= t_end)]


# ---- Light curve utilities --------------------------------------------------

def find_longest_consecutive_sequence(arr, tolerance=0):
    """
    Find the longest consecutive sequence in an array with a given gap tolerance.
    Used to select TESS sectors.

    Parameters
    ----------
    arr : array-like
        Observed sector numbers.
    tolerance : int
        Maximum allowed gap between consecutive sectors.

    Returns
    -------
    consecutive_array, start_idx, end_idx
    """
    if len(arr) == 0:
        return None, None, None
    if len(arr) == 1:
        return arr, 0, 0

    diffs = np.diff(arr) - 1
    is_consecutive = diffs <= tolerance
    is_consecutive_padded = np.concatenate(([False], is_consecutive, [False]))
    transitions = np.diff(is_consecutive_padded.astype(int))
    starts = np.where(transitions == 1)[0]
    ends   = np.where(transitions == -1)[0]

    if len(starts) == 0:
        return arr[0:1], 0, 0

    lengths = ends - starts + 1
    max_idx = np.argmax(lengths)
    start_idx = starts[max_idx]
    end_idx   = ends[max_idx]
    return arr[start_idx:end_idx + 1], start_idx, end_idx


def sigma_clip_mask(flux, sigma_upper=5, sigma_lower=5):
    """
    Return a boolean mask removing outliers while preserving sequences of
    3+ consecutive low points (which may be genuine transit signals).

    Parameters
    ----------
    flux : array-like
    sigma_upper, sigma_lower : float

    Returns
    -------
    mask : boolean array — True for points to keep.
    """
    flux   = np.array(flux)
    median = np.median(flux)
    sigma  = mad_std(flux)

    upper_outliers = flux > (median + sigma_upper * sigma)
    lower_outliers = flux < (median - sigma_lower * sigma)

    lower_padded = np.concatenate(([False], lower_outliers, [False]))
    diff   = np.diff(lower_padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    keep_lower = np.zeros_like(lower_outliers)
    for start, end in zip(starts, ends):
        if (end - start) >= 3:
            keep_lower[start:end] = True

    return ~upper_outliers & (~lower_outliers | keep_lower)


def time_flux_err(timeseries, flux_kw='flux', flux_err_kw='flux_err',
                  replace_nan_err=True):
    """
    Extract (time, flux, flux_err) arrays from a TimeSeries or LightCurve.

    Parameters
    ----------
    timeseries : astropy TimeSeries or lightkurve LightCurve
    flux_kw : str
    flux_err_kw : str
    replace_nan_err : bool
        If True and flux_err is all NaN, replace with mad_std(flux).

    Returns
    -------
    t, flux, flux_err : np.ndarray
    """
    t        = np.ascontiguousarray(timeseries.time.value)
    flux     = np.ascontiguousarray(np.array(timeseries[flux_kw],     dtype=np.float64))
    flux_err = np.ascontiguousarray(np.array(timeseries[flux_err_kw], dtype=np.float64))

    if np.isnan(np.sum(flux_err)):
        flux_err = np.full_like(flux, mad_std(flux)) if replace_nan_err else np.nan * flux

    return t, flux, flux_err


def bin_lc(timeseries, bin_size, flux_kw='flux', flux_err_kw='flux_err'):
    """
    Bin a light curve. Flux is medianed; errors are combined in quadrature.
    Returns an object of the same type as the input.

    Parameters
    ----------
    timeseries : astropy TimeSeries or lightkurve LightCurve
    bin_size : float or astropy Quantity
        Bin size in days, or an astropy Quantity (e.g. 4*u.h).
    flux_kw, flux_err_kw : str

    Returns
    -------
    Binned object of the same type as input.
    """
    bin_size_days = bin_size.to(u.d).value if hasattr(bin_size, 'to') else float(bin_size)

    t, flux, err = time_flux_err(timeseries, flux_kw=flux_kw, flux_err_kw=flux_err_kw)

    bins = np.arange(t.min(), t.max() + bin_size_days, bin_size_days)
    idx  = np.digitize(t, bins)

    t_bin, f_bin, e_bin = [], [], []
    for i in range(1, len(bins)):
        mask = idx == i
        if mask.sum() == 0:
            continue
        t_bin.append(np.nanmedian(t[mask]))
        f_bin.append(np.nanmedian(flux[mask]))
        n = np.sum(np.isfinite(err[mask]))
        e_bin.append(np.sqrt(np.nansum(err[mask]**2)) / n if n > 0 else np.nan)

    t_bin = np.array(t_bin)
    f_bin = np.array(f_bin)
    e_bin = np.array(e_bin)

    time_obj = Time(t_bin, format=timeseries.time.format, scale=timeseries.time.scale)

    if isinstance(timeseries, lk.LightCurve):
        result = lk.LightCurve(time=time_obj, data={flux_kw: f_bin, flux_err_kw: e_bin})
        result.flux     = result[flux_kw]
        result.flux_err = result[flux_err_kw]
    else:
        result = TimeSeries(time=time_obj, data={flux_kw: f_bin, flux_err_kw: e_bin})

    return result


# ---- Plotting utilities -----------------------------------------------------

def tls_plotter(tls_result, maxper=None, title=None, savefig=False, outdir=None):
    """Plot a TLS periodogram, highlighting aliases."""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.axvline(tls_result['period'], alpha=0.4, lw=3, color='xkcd:green')
    plt.xlim(np.min(tls_result['periods']), np.max(tls_result['periods']))
    for n in range(2, 30):
        ax.axvline(n * tls_result['period'], alpha=0.4, lw=1, ls='dashed', color='xkcd:vibrant green')
        ax.axvline(tls_result['period'] / n, alpha=0.4, lw=1, ls='dashed', color='xkcd:vibrant green')
    plt.ylabel(r'SDE', fontsize=14)
    plt.xlabel('Period (days)', fontsize=14)
    ax.text(.02, .95,
            f"SDE = {tls_result['SDE']:.2f}, best period = {tls_result['period']:.5f} days",
            transform=ax.transAxes, fontsize=12, backgroundcolor='white')
    plt.plot(tls_result['periods'], tls_result['power'], color='black', lw=0.5)
    plt.ylim(0, min(tls_result['SDE'] + 5, 1.2 * tls_result['SDE']))
    if maxper is not None:
        plt.xlim(0, maxper)
    if title is not None:
        plt.title(title, fontsize=16)
    if savefig:
        fname = str(title).replace(' ', '_').replace('.', '-') + '.png'
        path  = os.path.join(outdir, fname) if outdir else fname
        os.makedirs(outdir, exist_ok=True) if outdir else None
        plt.savefig(path, facecolor='white', dpi=240)


def fold(time, period, origo=0.0, shift=0.0, normalize=True, clip_range=None):
    """Fold time array over a given period."""
    tf = ((time - origo) / period + shift) % 1.
    if not normalize:
        tf *= period
    if clip_range is not None:
        mask = np.logical_and(clip_range[0] < tf, tf < clip_range[1])
        tf = tf[mask], mask
    return tf


def downsample_time(time, vals, inttime=1.):
    """Time binning for phase-folded data."""
    duration = time.max() - time.min()
    nbins = int(np.ceil(duration / inttime))
    bins  = np.arange(nbins)
    edges = time[0] + bins * inttime
    bids  = np.digitize(time, edges) - 1
    bt, bv, be = np.full(nbins, np.nan), np.zeros(nbins), np.zeros(nbins)
    for i, bid in enumerate(bins):
        bmask = bid == bids
        if bmask.sum() > 0:
            bt[i] = time[bmask].mean()
            bv[i] = vals[bmask].mean()
            be[i] = vals[bmask].std() / np.sqrt(bmask.sum()) if bmask.sum() > 2 else np.nan
    m = np.isfinite(be)
    return bt[m], bv[m], be[m]


def plot_folded_transit(lpf, method='median', figsize=(8, 6), ylim=None,
                        xlim=None, binwidth=20, offset=0):
    """Plot a phase-folded transit from a PyTransit LPF object."""
    SMALL_SIZE  = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    plt.rc('font',   size=SMALL_SIZE)
    plt.rc('axes',   titlesize=SMALL_SIZE)
    plt.rc('axes',   labelsize=MEDIUM_SIZE)
    plt.rc('xtick',  labelsize=SMALL_SIZE)
    plt.rc('ytick',  labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    if method == 'de':
        pv = lpf.de.minimum_location
        tc, p = pv[[0, 1]]
    elif method == 'median':
        pv = lpf.posterior_samples().median()
        tc, p = pv[['tc', 'p']]
    else:
        raise NotImplementedError

    phase    = p * fold(lpf.timea, p, tc, 0.5)
    bw       = binwidth / 24 / 60
    sids     = np.argsort(phase)
    tm       = lpf.transit_model(pv)
    bp, bfo, beo = downsample_time(phase[sids], lpf.ofluxa[sids], bw)

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(phase - 0.5 * p, lpf.ofluxa, lpf.errora,
                fmt='o', color='xkcd:grey', ecolor='xkcd:silver',
                markersize=2, alpha=1, lw=0.5, zorder=1)
    ax.errorbar(bp - 0.5 * p, bfo, beo,
                fmt='o', ms=6, color='xkcd:tree green', zorder=3)
    ax.plot(phase[sids] - 0.5 * p, tm[sids],
            color='xkcd:black', lw=2.5, alpha=1, zorder=2)

    xlim = xlim if xlim is not None else 1.01 * (bp[np.isfinite(bp)][[0, -1]] - 0.5 * p)
    ylim = ylim if ylim is not None else (min(lpf.ofluxa), max(lpf.ofluxa))
    plt.setp(ax, ylim=ylim, xlim=xlim,
             xlabel='Time - Tc [d]', ylabel='Normalised flux')
    plt.title(f"{lpf.name} phase folded at {p:.3f} d")
    fig.tight_layout()
    return fig


# ---- Planet parameter estimators --------------------------------------------

def estimate_planet_mass(R_p, R_p_err=None, rho_p='rocky'):
    """
    Estimate planet mass from radius and assumed bulk density.

    Parameters
    ----------
    R_p : float or astropy Quantity (Earth radii)
    R_p_err : float or astropy Quantity, optional
    rho_p : str, float, or astropy Quantity
        'iron', 'rocky', 'neptune', 'jupiter', 'puffy', or kg/m^3.

    Returns
    -------
    M_planet, M_planet_err : astropy Quantity
    """
    dens_dic = {
        'iron':    7874 * (u.kg / u.m**3),
        'rocky':   5500 * (u.kg / u.m**3),
        'neptune': 1640 * (u.kg / u.m**3),
        'jupiter': 1330 * (u.kg / u.m**3),
        'puffy':    300 * (u.kg / u.m**3),
    }

    if isinstance(R_p, (int, float)):
        R_p = R_p * u.earthRad
    elif isinstance(R_p, u.Quantity):
        R_p = R_p.to(u.earthRad)
    else:
        raise TypeError('R_p should be astropy Quantity or float')

    if R_p_err is not None:
        R_p_err = (R_p_err * u.earthRad if isinstance(R_p_err, (int, float))
                   else R_p_err.to(u.earthRad))

    if isinstance(rho_p, str):
        if rho_p.lower() not in dens_dic:
            raise ValueError(f'Accepted str values for rho_p: {list(dens_dic.keys())}')
        rho_p = dens_dic[rho_p.lower()]
    elif isinstance(rho_p, (int, float)):
        rho_p = rho_p * (u.kg / u.m**3)

    earth_density = 5514 * (u.kg / u.m**3)
    M_planet = (R_p.value**3 * rho_p / earth_density) * u.earthMass

    if R_p_err is not None:
        M_planet_err = M_planet.value * 3 * (R_p_err.value / R_p.value) * u.earthMass
        return M_planet, M_planet_err
    return M_planet, None


def estimate_semi_amplitude(period, M_star=1*u.solMass, period_err=None,
                             M_star_err=None, M_planet=None, M_planet_err=None,
                             R_planet=None, R_planet_err=None, rho_planet='rocky',
                             inc=90*u.deg, inc_err=None, ecc=0):
    """
    Estimate the RV semi-amplitude K for a planet.

    Uses Eq. 14 from Lovis & Fischer (2010).
    """
    def to_quantity(val, unit):
        if isinstance(val, (int, float)):
            return val * unit
        return val

    period   = to_quantity(period, u.d)
    M_star   = to_quantity(M_star, u.solMass)
    inc      = to_quantity(inc, u.deg)
    if period_err  is not None: period_err  = to_quantity(period_err,  u.d)
    if M_star_err  is not None: M_star_err  = to_quantity(M_star_err,  u.solMass)
    if inc_err     is not None: inc_err     = to_quantity(inc_err,     u.deg)

    if M_planet is None:
        if R_planet is None:
            raise ValueError('R_planet required when M_planet is not given')
        M_planet, M_planet_err = estimate_planet_mass(R_planet, R_planet_err, rho_planet)
    else:
        if R_planet is not None:
            warnings.warn('M_planet overrides R_planet and rho_planet')
        M_planet = to_quantity(M_planet, u.earthMass)
        if M_planet_err is not None:
            M_planet_err = to_quantity(M_planet_err, u.earthMass)

    M_p_jov   = M_planet.to(u.jupiterMass).value
    M_tot_sol = (M_star + M_planet).to(u.solMass).value
    inc_rad   = inc.to(u.rad).value

    K = (28.4329 * (u.m / u.s) * M_p_jov * np.sin(inc_rad)
         * M_tot_sol**(-2/3) * period.to(u.year).value**(-1/3)
         / np.sqrt(1 - ecc))

    if any(e is not None for e in [M_planet_err, M_star_err, period_err, inc_err]):
        rel_err_sq = 0.0
        if M_planet_err is not None:
            rel_err_sq += (M_planet_err.to(u.jupiterMass).value / M_p_jov)**2
        if M_star_err is not None:
            rel_err_sq += (2/3 * M_star_err.to(u.solMass).value / M_star.to(u.solMass).value)**2
        if period_err is not None:
            rel_err_sq += (1/3 * period_err.to(u.year).value / period.to(u.year).value)**2
        if inc_err is not None and np.sin(inc_rad) > 0:
            rel_err_sq += (np.cos(inc_rad) * inc_err.to(u.rad).value / np.sin(inc_rad))**2
        return K, K.value * np.sqrt(rel_err_sq) * (u.m / u.s)

    return K, None


# ---- False-positive vetting -------------------------------------------------

def vet_tls_result(tls_result, odd_even_sigma=5.0,
                   depth_scatter_nsig=3.0, secondary_depth=0.5):
    """
    Run false-positive vetting checks on a TLS result.

    The minimum transit count is handled upstream via n_transits_min in TLS.
    Visual inspection of the phase-folded plot is strongly recommended.

    Returns
    -------
    flags : dict — True means the check FAILED.
    values : dict — computed diagnostic values.
    passed : bool
    """
    flags  = {}
    values = {}

    n_transits = int(tls_result['transit_count'])
    values['n_transits'] = n_transits

    # 1. Odd/even depth mismatch
    if n_transits >= 2:
        odd_even = float(tls_result['odd_even_mismatch'])
        values['odd_even_sigma'] = odd_even
        flags['odd_even_mismatch'] = odd_even > odd_even_sigma
    else:
        values['odd_even_sigma'] = np.nan
        flags['odd_even_mismatch'] = False

    # 2. Transit depth consistency
    if n_transits >= 3:
        depths     = np.array(tls_result['transit_depths'])
        depths_err = np.array(tls_result['transit_depths_uncertainties'])
        median_d   = np.nanmedian(depths)
        with np.errstate(invalid='ignore'):
            deviations = np.abs(depths - median_d) / np.where(depths_err > 0, depths_err, np.nan)
        max_dev = float(np.nanmax(deviations))
        values['max_depth_deviation_sigma'] = max_dev
        flags['depth_inconsistency'] = max_dev > depth_scatter_nsig
    else:
        values['max_depth_deviation_sigma'] = np.nan
        flags['depth_inconsistency'] = False

    # 3. Secondary eclipse
    try:
        phases        = tls_result['folded_phase']
        flux_fold     = tls_result['folded_y']
        primary_depth = float(tls_result['depth'])
        near_sec = np.abs(phases - 0.5) < 0.05
        if near_sec.sum() > 0 and primary_depth > 0:
            sec_frac = (1.0 - float(np.nanmedian(flux_fold[near_sec]))) / primary_depth
            values['secondary_depth_fraction'] = sec_frac
            flags['secondary_eclipse'] = sec_frac > secondary_depth
        else:
            values['secondary_depth_fraction'] = np.nan
            flags['secondary_eclipse'] = False
    except (KeyError, TypeError):
        values['secondary_depth_fraction'] = np.nan
        flags['secondary_eclipse'] = False

    passed = not any(flags.values())
    return flags, values, passed


def print_vetting_report(flags, values, passed):
    """Print a human-readable vetting report."""
    status = 'PASSED' if passed else 'FAILED'
    print(f"\n{'='*40}")
    print(f"  Vetting report       : {status}")
    print(f"{'='*40}")
    print(f"  N transits           : {values.get('n_transits', 'N/A')}")
    print(f"  Odd/even mismatch    : {values.get('odd_even_sigma', np.nan):.2f} sigma  {'[!]' if flags.get('odd_even_mismatch') else ''}")
    print(f"  Max depth deviation  : {values.get('max_depth_deviation_sigma', np.nan):.2f} sigma  {'[!]' if flags.get('depth_inconsistency') else ''}")
    print(f"  Secondary depth      : {values.get('secondary_depth_fraction', np.nan):.3f} x primary  {'[!]' if flags.get('secondary_eclipse') else ''}")
    print(f"{'='*40}")
    print("Don't forget to visually inspect the phase-folded plot :)\n")


# ---- Iterative planet search ------------------------------------------------

def planet_search(lc, stellar_params, max_planets=10, sde_threshold=9, snr_threshold=4,
                  period_max=None, mask_mult=1.4, min_transits=2,
                  odd_even_sigma=5.0, depth_scatter_nsig=3.0, secondary_depth=0.5,
                  target="", outdir="", outdir_plots="", outdir_tls=""):
    """
    Iterative planet search with TLS and PyTransit.

    Parameters
    ----------
    lc : lightkurve LightCurve
    stellar_params : dict
    max_planets : int
    sde_threshold, snr_threshold : float
    period_max : float or None
    mask_mult : float
    min_transits : int — passed to TLS via n_transits_min.
    odd_even_sigma, depth_scatter_nsig, secondary_depth : float — vetting thresholds.
    target : str
    outdir, outdir_plots, outdir_tls : str

    Returns
    -------
    all_results : list of OrderedDict
    """
    for d in [outdir, outdir_plots, outdir_tls]:
        os.makedirs(d, exist_ok=True)

    all_results = []
    t, y, yerr  = time_flux_err(lc)
    intransit   = np.full(len(t), False)

    for iteration in range(max_planets):

        tls_kwargs = dict(
            R_star    =stellar_params['Rstar'],
            R_star_min=stellar_params['Rstar'] - stellar_params['Rstar_err'],
            R_star_max=stellar_params['Rstar'] + stellar_params['Rstar_err'],
            M_star    =stellar_params['Mstar'],
            M_star_min=stellar_params['Mstar'] - stellar_params['Mstar_err'],
            M_star_max=stellar_params['Mstar'] + stellar_params['Mstar_err'],
            u=[stellar_params['u1'], stellar_params['u2']],
            n_transits_min=min_transits,
        )
        if period_max is not None:
            tls_kwargs['period_max'] = period_max

        tls_result = transitleastsquares(
            t[~intransit], y[~intransit], yerr[~intransit]
        ).power(**tls_kwargs)

        with open(os.path.join(outdir_tls, f'tls_{target.replace(" ", "_")}_{iteration+1}.json'), 'w') as f:
            json.dump(json.dumps(tls_result, cls=NumpyEncoder), f)

        if tls_result['SDE'] < sde_threshold or tls_result['snr'] < snr_threshold:
            print(f"Iteration {iteration+1}: SDE={tls_result['SDE']:.2f}, "
                  f"SNR={tls_result['snr']:.1f}. Below threshold, stopping.\n")
            tls_plotter(tls_result, title=f"{target} non-detection TLS Periodogram",
                        savefig=True, maxper=period_max, outdir=outdir_plots)
            break

        print(f"\nIteration {iteration+1}: SDE={tls_result['SDE']:.2f}, "
              f"P={tls_result['period']:.3f} d\n")

        tls_plotter(tls_result, title=f"{target}.{iteration+1:0>2} TLS Periodogram",
                    savefig=True, maxper=period_max, outdir=outdir_plots)
        if period_max is None or period_max > 40:
            tls_plotter(tls_result, title=f"{target}.{iteration+1:0>2} TLS Periodogram_40d",
                        savefig=True, maxper=40, outdir=outdir_plots)

        flags, values, passed = vet_tls_result(
            tls_result,
            odd_even_sigma=odd_even_sigma,
            depth_scatter_nsig=depth_scatter_nsig,
            secondary_depth=secondary_depth,
        )
        print_vetting_report(flags, values, passed)
        if not passed:
            print(f"Candidate at P={tls_result['period']:.4f} d failed vetting. Stopping.\n")
            break

        print("Starting optimisation.\n")
        transit_window = transit_mask(t, tls_result['period'],
                                      0.5 * tls_result['period'], tls_result['T0'])
        fit_window = ~intransit & transit_window

        lpf = BaseLPF(f"{target}.{iteration+1:0>2}", ['TESS'],
                      times=t[fit_window], fluxes=y[fit_window],
                      exptimes=2 * u.min.to(u.d), errors=[yerr[fit_window]])

        lpf.set_prior('tc',      'NP', tls_result['T0'],              0.05)
        lpf.set_prior('p',       'NP', tls_result['period'],          0.2)
        lpf.set_prior('rho',     'UP', 0.5,                           2.0)
        lpf.set_prior('k2',      'UP', (1 - tls_result['depth']),     0.2**2)
        lpf.set_prior('q1_TESS', 'NP', stellar_params['q1'],          stellar_params['q1_err'])
        lpf.set_prior('q2_TESS', 'NP', stellar_params['q2'],          stellar_params['q2_err'])

        lpf.optimize_global(1000)
        lpf.sample_mcmc(2500, thin=5)

        samples = lpf.posterior_samples()
        medians = np.median(samples, axis=0)
        stds    = np.std(samples, axis=0)

        pffig = plot_folded_transit(lpf, method='median', xlim=(-.2, .2),
                                    ylim=None, binwidth=20)
        pffig.savefig(os.path.join(outdir_plots,
                                   f"{target}-{iteration+1:0>2}_phasefold.png"),
                      facecolor='white', dpi=240)
        plt.close("all")

        results = OrderedDict()
        results['iteration'] = iteration
        results['tls_sde']   = tls_result['SDE']
        results['tls_snr']   = tls_result['snr']

        for name, med, std in zip(samples.columns, medians, stds):
            if name == 'tc':
                results['T0']     = med
                results['T0_err'] = std
            elif name == 'inc':
                results['inc']     = med * u.rad.to(u.deg)
                results['inc_err'] = std * u.rad.to(u.deg)
            else:
                results[name]          = med
                results[f'{name}_err'] = std

        k, k_err      = results['k'], results['k_err']
        Rstar, Rs_err = stellar_params['Rstar'], stellar_params['Rstar_err']
        Rp     = k * Rstar * u.solRad.to(u.earthRad)
        Rp_err = Rp * np.sqrt((k_err / k)**2 + (Rs_err / Rstar)**2)
        results['Rp']     = Rp
        results['Rp_err'] = Rp_err

        if   Rp < 1.5: regimes = [('earth', 'rocky')]
        elif Rp < 6.0: regimes = [('earth', 'rocky'), ('nep', 'neptune')]
        else:          regimes = [('jup', 'jupiter'),  ('puffy', 'puffy')]

        for label, regime in regimes:
            Mp, Mp_err = estimate_planet_mass(Rp, Rp_err, regime)
            # K,  K_err  = estimate_semi_amplitude(
            #     results['p'], period_err=results['p_err'],s
            #     M_star=Rstar, M_star_err=Rs_err,
            #     M_planet=Mp, M_planet_err=Mp_err,
            #     inc=results['inc'], inc_err=results['inc_err'], ecc=0
            # )
            K,  K_err  = estimate_semi_amplitude(
                results['p'], period_err=results['p_err'], M_star=stellar_params['Mstar'],
                M_star_err=stellar_params['Mstar_err'], M_planet=Mp, M_planet_err=Mp_err,
                inc=results['inc'], inc_err=results['inc_err'], ecc=0
                )
            results[f'Mp_{label}']     = Mp
            results[f'Mp_{label}_err'] = Mp_err
            results[f'K_{label}']      = K
            results[f'K_{label}_err']  = K_err

        all_results.append(results)
        plt.close()

        intransitb = transit_mask(t, results['p'],
                                  mask_mult * results['t14'], results['T0'])
        intransit  = np.logical_or(intransit, intransitb)

    else:
        print(f"Reached maximum of {max_planets} planets.\n")
        if max_planets > 8:
            print("You either made a great discovery or something went wrong.")

    return all_results


# ---- Output functions -------------------------------------------------------

def human_readable_table(data):
    """Create a human-readable summary table from FULMAR results."""
    k_columns = [col for col in data.colnames
                 if col.startswith('K_') and not col.endswith('_err')]
    summary = Table()
    summary['#']      = data['iteration']
    summary['Period'] = [f"{p:.3f}"   for p   in data['p']]
    summary['T0']     = [f"{t0:.3f}"  for t0  in data['T0']]
    summary['SDE']    = [f"{sde:.2f}" for sde in data['tls_sde']]
    summary['Rp']     = [f"{rp:.2f} +- {rp_err:.2f}"
                         for rp, rp_err in zip(data['Rp'], data['Rp_err'])]
    for k_col in k_columns:
        comp = k_col.replace('K_', '')
        summary[f'K_{comp}'] = [f"{k:.2f} +- {k_err:.2f}"
                                 for k, k_err in zip(data[k_col], data[f'{k_col}_err'])]
        summary[f'K_{comp}'].unit = 'm/s'
    summary['Period'].unit = u.d
    summary['T0'].unit     = data['T0'].unit
    summary['Rp'].unit     = u.earthRad
    return summary


def lc_to_dat(lc, time_col, flux_col, err_col, output_file,
              jitter_flag=0, offset_flag=-1):
    """Write a light curve column subset to a PyOrbit-compatible .dat file."""
    t = Table()
    t['epoch']       = lc.time.value if time_col == 'time' else np.array(lc[time_col])
    t['value']       = np.array(lc[flux_col])
    t['error']       = np.array(lc[err_col])
    t['jitter_flag'] = np.full(len(t), jitter_flag, dtype=int)
    t['offset_flag'] = np.full(len(t), offset_flag, dtype=int)
    t.write(output_file, format='ascii.commented_header', overwrite=True)
    print(f"Exported {len(t)} epochs to {output_file}")


def make_pyorbit_yaml(results, stellar_params, target, outdir):
    """
    Generate a PyOrbit YAML configuration file from FULMAR results.
    Planets are labelled b, c, d, ... in order of detection.
    """
    letters = 'bcdefghij'
    planets = {}
    for i, res in enumerate(results):
        p, p_err   = float(res['p']),  float(res['p_err'])
        t0, t0_err = float(res['T0']), float(res['T0_err'])
        planets[letters[i]] = {
            'orbit': 'circular',
            'use_time_inferior_conjunction': True,
            'boundaries': {
                'P':  [round(p - 1, 6),   round(p + 1, 6)],
                'K':  [0.001, 20.0],
                'Tc': [round(t0 - 10, 6), round(t0 + 10, 6)],
            },
            'priors': {
                'P':  ['Gaussian', round(p,  6), round(p_err,  6)],
                'Tc': ['Gaussian', round(t0, 6), round(t0_err, 6)],
            },
            'spaces': {'K': 'Linear'},
        }
    outpath = os.path.join(outdir, f'{target.replace(" ", "_")}_pyorbit.yaml')
    with open(outpath, 'w') as f:
        yaml.dump({'planets': planets}, f, default_flow_style=False, sort_keys=False)
    print(f'PyOrbit YAML written to {outpath}')
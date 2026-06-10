#!/usr/bin/env python3
# Author: José Rodrigues - ORCID 0000-0001-5164-3602
"""
FULMAR - Follow-Up Lightcurves Multitool Assisting Radial-velocities

Usage:
    python fulmar.py -t TOI-512
    python fulmar.py --target TOI-512
    python fulmar.py targets.txt
    python fulmar.py targets.yaml

targets.txt: one target per line; lines starting with # are ignored.

targets.yaml format:
    targets:
      - TOI-512
      - TOI-174
    parameters:          # optional, overrides fulmar_config.py
      MODE: FAST
      SDE_THRESHOLD: 8.0
      PERIOD_MAX: 40
"""

import argparse
import importlib.util
import os
import sys
import traceback

import numpy as np
import astropy.units as u
import lightkurve as lk
import wotan as wt

from astropy.stats import mad_std
from astropy.table import Table, vstack
from astropy.io.ascii import convert_numpy

from fulmar_utils import (
    find_longest_consecutive_sequence,
    sectors_in_date_range,
    sigma_clip_mask,
    time_flux_err,
    bin_lc,
    planet_search,
    human_readable_table,
    lc_to_dat,
    make_pyorbit_yaml,
)

# ---- Parse arguments --------------------------------------------------------
parser = argparse.ArgumentParser(
    description='FULMAR: Follow-Up Lightcurves Multitool Assisting Radial-velocities',
    usage='fulmar.py -t TARGET | fulmar.py FILE'
)
parser.add_argument('-t', '--target', default=None,
                    help='Single target name (TIC ID, TOI number, or common name)')
parser.add_argument('file', nargs='?', default=None,
                    help='Target list (.txt) or run file (.yaml)')
args = parser.parse_args()

if args.target is None and args.file is None:
    parser.print_help()
    print('\nError: provide either -t TARGET or a target file (.txt or .yaml).')
    sys.exit(1)

# ---- Load default config ----------------------------------------------------
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fulmar_config.py')
if not os.path.exists(config_path):
    print(f'Error: fulmar_config.py not found at {config_path}')
    sys.exit(1)

spec = importlib.util.spec_from_file_location('config', config_path)
cfg  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

params = {k: getattr(cfg, k) for k in [
    'MODE', 'GAP_TOLERANCE', 'MAX_SECTORS', 'DATE_START', 'DATE_END',
    'PERIOD_MAX', 'SDE_THRESHOLD', 'SNR_THRESHOLD', 'MIN_TRANSITS',
    'ODD_EVEN_SIGMA', 'DEPTH_SCATTER_NSIG', 'SECONDARY_DEPTH',
]}

# ---- Build target list ------------------------------------------------------
targets   = []
overrides = {}

sweetCat_table_url = "https://sweetcat.iastro.pt/catalog/SWEETCAT_Dataframe.csv"
converters = {'gaia_dr2': [convert_numpy(np.int64)], 'gaia_dr3': [convert_numpy(np.int64)]}
sweet_cat  = Table.read(sweetCat_table_url, encoding='UTF-8', format='csv', converters=converters)

if args.target:
    targets = [args.target]

elif args.file:
    if not os.path.exists(args.file):
        print(f'Error: file not found: {args.file}')
        sys.exit(1)

    if args.file.endswith(('.yaml', '.yml')):
        import yaml
        with open(args.file) as f:
            run = yaml.safe_load(f)
        targets   = run.get('targets', [])
        overrides = run.get('parameters', {})
        if not targets:
            print(f'Error: no targets found in {args.file}')
            sys.exit(1)

    elif args.file.endswith('.txt'):
        with open(args.file) as f:
            targets = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        if not targets:
            print(f'Error: no targets found in {args.file}')
            sys.exit(1)

    else:
        print('Error: unsupported file format. Use .txt or .yaml.')
        sys.exit(1)

params.update(overrides)

# ---- Run for each target ----------------------------------------------------
def run_target(target, params):
    """Run the full FULMAR workflow for a single target."""

    # Output directories
    outdir           = os.path.join('fulmar_output', target.replace(' ', '_'))
    outdir_plots     = os.path.join(outdir, 'plots')
    outdir_by_sector = os.path.join(outdir_plots, 'lc_by_sector')
    outdir_tls       = os.path.join(outdir, 'tls')
    outdir_pyorbit   = os.path.join(outdir, 'pyorbit')
    for d in [outdir, outdir_plots, outdir_by_sector, outdir_tls, outdir_pyorbit]:
        os.makedirs(d, exist_ok=True)

    # ---- 1. Stellar parameters ----------------------------------------------
    from astroquery.mast import Catalogs
    from astroquery.gaia import Gaia
    from astropy.coordinates import Angle, SkyCoord
    from astropy.io.ascii import convert_numpy
    from ldtk import LDPSetCreator, SVOFilter
    import warnings

    print('Retrieving TIC data\n')
    tic_data = Catalogs.query_object(target, catalog="TIC", radius=0.001)
    tic_ra   = tic_data['ra'][0]
    tic_dec  = tic_data['dec'][0]
    tic_gmag = tic_data['GAIAmag'][0]

    coord = SkyCoord(ra=tic_ra * u.deg, dec=tic_dec * u.deg, frame='icrs')

    print('Retrieving Gaia data\n')
    gaia_query = f"""
    SELECT g.source_id, g.ra, g.dec, g.parallax, g.parallax_error,
           g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
           ap.teff_gspphot, ap.teff_gspphot_lower, ap.teff_gspphot_upper,
           ap.logg_gspphot, ap.logg_gspphot_lower, ap.logg_gspphot_upper,
           ap.mh_gspphot, ap.mh_gspphot_lower, ap.mh_gspphot_upper,
           ap.radius_gspphot, ap.radius_gspphot_lower, ap.radius_gspphot_upper,
           ap.mass_flame, ap.mass_flame_lower, ap.mass_flame_upper,
           ap.azero_gspphot, ap.ebpminrp_gspphot, ap.ag_gspphot, ap.mg_gspphot,
           ap.libname_gspphot
    FROM gaiadr3.gaia_source AS g
    LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id
    WHERE 1=CONTAINS(POINT(g.ra, g.dec), CIRCLE({coord.ra.deg}, {coord.dec.deg}, {5/3600.}))
    """
    job          = Gaia.launch_job_async(gaia_query)
    gaia_result  = job.get_results()

    ok_mask = gaia_result['phot_g_mean_mag'].round(1) == tic_gmag.round(1)
    if sum(ok_mask) == 0:
        ok_mask = gaia_result['phot_g_mean_mag'].round(0) == tic_gmag.round(0)
        if sum(ok_mask) == 0:
            print(f"No magnitude match (G={tic_gmag:.1f}), defaulting to brightest target")
        gaia_source = gaia_result[0]
    elif ok_mask.sum() == 1:
        gaia_source = gaia_result[ok_mask]
    else:
        match = gaia_result[ok_mask]
        seps  = coord.separation(SkyCoord(ra=match['ra'], dec=match['dec'], frame='icrs'))
        gaia_source = match[seps.argmin()]
        print(f"Multiple Gaia matches, taking closest source. Check contamination.")

    def gaia_param(col, lower, upper, floor):
        val = gaia_source[col][0]
        err = max((gaia_source[upper][0] - gaia_source[lower][0]) / 2, floor)
        return val, err

    Teff_gaia,  Teff_gaia_err  = gaia_param('teff_gspphot',   'teff_gspphot_lower',   'teff_gspphot_upper',   70)
    logg_gaia,  logg_gaia_err  = gaia_param('logg_gspphot',   'logg_gspphot_lower',   'logg_gspphot_upper',   0.03)
    Rstar_gaia, Rstar_gaia_err = gaia_param('radius_gspphot', 'radius_gspphot_lower', 'radius_gspphot_upper', 0.05)

    stellar_params = {
        'tic':          int(tic_data['ID'][0]),
        'gaia_dr3':     gaia_source['SOURCE_ID'][0],
        'ra_gaia':      Angle(gaia_source['ra']).to_string(unit=u.hour)[0],
        'dec_gaia':     Angle(gaia_source['dec']).to_string()[0],
        'tmag':         tic_data['Tmag'][0],
        'gmag':         gaia_source['phot_g_mean_mag'][0],
        'parallax':     gaia_source['parallax'][0],
        'parallax_err': gaia_source['parallax_error'][0],
        'logg_gaia':    logg_gaia,
        'logg_gaia_err':logg_gaia_err,
    }

    print('Checking SWEET-Cat\n')
    # SWEET-Cat
    #sweetCat_table_url = "https://sweetcat.iastro.pt/catalog/SWEETCAT_Dataframe.csv"
    #converters = {'gaia_dr2': [convert_numpy(np.int64)], 'gaia_dr3': [convert_numpy(np.int64)]}
    #sweet_cat  = Table.read(sweetCat_table_url, encoding='UTF-8', format='csv', converters=converters)
    SC_check   = sweet_cat['gaia_dr3'] == stellar_params['gaia_dr3']

    if sum(SC_check) == 0:
        print(f"{target} not in SWEET-Cat. Falling back to TIC v8 / Gaia DR3.")
        stellar_params['Teff']        = tic_data['Teff'][0] if Teff_gaia < 4000 else Teff_gaia
        stellar_params['Teff_err']    = tic_data['e_Teff'][0] if Teff_gaia < 4000 else Teff_gaia_err
        stellar_params['Teff_source'] = 'TIC' if Teff_gaia < 4000 else 'Gaia DR3'
        stellar_params['logg']        = tic_data['logg'][0]
        stellar_params['logg_err']    = tic_data['e_logg'][0]
        stellar_params['logg_source'] = 'TIC'
        stellar_params['Mstar']       = tic_data['mass'][0]
        stellar_params['Mstar_err']   = tic_data['e_mass'][0]
        stellar_params['Mstar_source']= 'TIC'
        stellar_params['Rstar']       = Rstar_gaia
        stellar_params['Rstar_err']   = Rstar_gaia_err
        stellar_params['Rstar_source']= 'Gaia DR3'
        if np.isnan(tic_data['MH'][0]):
            stellar_params['[Fe/H]']       = gaia_source['mh_gspphot'][0]
            stellar_params['[Fe/H]_err']   = 0.25
            stellar_params['[Fe/H]_source']= 'Gaia DR3'
        else:
            stellar_params['[Fe/H]']       = tic_data['MH'][0]
            stellar_params['[Fe/H]_err']   = 0.25
            stellar_params['[Fe/H]_source']= 'TIC'
    else:
        print(f"{target} found in SWEET-Cat.")
        sc = sweet_cat[SC_check]
        for key, col in [('Teff','Teff'), ('logg','Logg'), ('Mstar','Mass_t'),
                         ('Rstar','Radius_t'), ('[Fe/H]','[Fe/H]')]:
            stellar_params[key]         = float(sc[col][0])
            stellar_params[f'{key}_err']= float(sc[f'e{col}'][0])
            stellar_params[f'{key}_source'] = 'SWEET-Cat'

    print('Computing limb darkening parameters\n')        
    # Limb darkening
    teff = (stellar_params['Teff'],       stellar_params['Teff_err'])
    logg = (stellar_params['logg_gaia'],  stellar_params['logg_gaia_err'])
    z    = (stellar_params['[Fe/H]'],     stellar_params['[Fe/H]_err'])

    sc_ld = LDPSetCreator(teff, logg, z, filters=[SVOFilter('TESS')], dataset='vis')
    ps    = sc_ld.create_profiles(10000)
    ps.resample_linear_z(300)

    qd, qd_err = ps.coeffs_qd(do_mc=True, n_mc_samples=10000)
    tq, tq_err = ps.coeffs_tq(do_mc=True, n_mc_samples=10000)

    stellar_params.update({
        'u1': qd[0,0], 'u2': qd[0,1], 'u1_err': qd_err[0,0], 'u2_err': qd_err[0,1],
        'q1': tq[0,0], 'q2': tq[0,1], 'q1_err': tq_err[0,0], 'q2_err': tq_err[0,1],
    })
#    print(stellar_params)
    print('Downloading the Lightcurves\n')
    # ---- 2. Download light curves -------------------------------------------
    search_result = lk.search_lightcurve(target, author='SPOC', exptime=120)
    lc_col        = search_result.download_all()
    print(f"{target} observed in sectors: {lc_col.sector}")

    MODE          = params['MODE']
    GAP_TOLERANCE = params['GAP_TOLERANCE']
    MAX_SECTORS   = params['MAX_SECTORS']
    DATE_START    = params['DATE_START']
    DATE_END      = params['DATE_END']
    PERIOD_MAX    = params['PERIOD_MAX']

    if MODE == 'SINGLE':
        lc_tot = lc_col[0:1]
    elif MODE == 'FAST':
        consecutive, start_idx, end_idx = find_longest_consecutive_sequence(
            lc_col.sector, tolerance=GAP_TOLERANCE)
        if len(consecutive) > MAX_SECTORS:
            consecutive = consecutive[:MAX_SECTORS]
            end_idx = start_idx + MAX_SECTORS - 1
        lc_tot = lc_col[start_idx:end_idx + 1]
    else:
        if MODE == 'DATE':
            idx = sectors_in_date_range(lc_col, DATE_START, DATE_END)
            if idx:
                lc_tot = lc_col[idx]
            else:
                print(f"No sectors in [{DATE_START}, {DATE_END}]. Falling back to FAST.")
                consecutive, start_idx, end_idx = find_longest_consecutive_sequence(
                    lc_col.sector, tolerance=GAP_TOLERANCE)
                if len(consecutive) > MAX_SECTORS:
                    consecutive = consecutive[:MAX_SECTORS]
                    end_idx = start_idx + MAX_SECTORS - 1
                lc_tot = lc_col[start_idx:end_idx + 1]
        else:
            lc_tot = lc_col

    print(f"Mode {MODE}: sectors {list(lc_tot.sector)}")

    print('Cleaning the lightcurve\n')
    # ---- 3. Clean light curves ----------------------------------------------
    wl = 0.75
    if PERIOD_MAX is not None:
        wl = max(wl, 3 * wt.t14(R_s=stellar_params['Rstar'],
                                 M_s=stellar_params['Mstar'],
                                 P=PERIOD_MAX, small_planet=True))

    import matplotlib.pyplot as plt
    cleaned_lcs = []

    for lc in lc_tot:
        lc = lc[lc.quality == 0]

        pdcsap_norm = np.nanmedian(lc['pdcsap_flux'].value)
        sap_norm    = np.nanmedian(lc['sap_flux'].value)

        lc = lc.normalize()
        lc['pdcsap_flux_norm']     = lc['pdcsap_flux']     / pdcsap_norm
        lc['pdcsap_flux_err_norm'] = lc['pdcsap_flux_err'] / pdcsap_norm
        lc['sap_flux_norm']        = lc['sap_flux']        / sap_norm
        lc['sap_flux_err_norm']    = lc['sap_flux_err']    / sap_norm
        lc['sector']               = np.full(len(lc), lc.sector)

        flatten_lc, trend_lc = wt.flatten(
            lc.time.value, lc.flux.value, window_length=wl,
            method='biweight', return_trend=True, break_tolerance=0.1
        )
        lc_cln         = lc.copy()
        lc_cln['trend']= trend_lc
        lc_cln.flux    = flatten_lc * lc.flux.unit

        sector_num  = lc.sector[0]
        sig_before  = mad_std(lc.flux)
        sig_after   = mad_std(lc_cln.flux)
        print(f"Sector {sector_num}: σ {1e6*sig_before:.0f} -> {1e6*sig_after:.0f} ppm "
              f"({100*(sig_before/sig_after-1):.1f}% improvement)")
        if sig_after > sig_before:
            print("  Warning: detrending did not improve scatter.")

        lc_cln = lc_cln[sigma_clip_mask(lc_cln.flux, sigma_upper=5, sigma_lower=5)]

        # Per-sector plot
        t_phot, y_phot, yerr_phot = time_flux_err(lc)
        lc_bin = lc.bin(1. * u.d)
        title  = f"{target} PDCSAP Lightcurve sector {sector_num}"
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.errorbar(t_phot, y_phot, yerr_phot,
                    fmt='o', color='xkcd:grey', ecolor='xkcd:silver',
                    markersize=1.5, alpha=1, lw=0.4)
        ax.plot(lc_cln.time.value, lc_cln['trend'],
                'o', color='xkcd:charcoal', markersize=0.5, lw=1.4, zorder=1000)
        ax.plot(lc_bin.time.value, lc_bin.flux.value,
                'o', markerfacecolor='xkcd:vibrant green', markeredgecolor='xkcd:black',
                markersize=6.5, markeredgewidth=0.75, zorder=1000)
        ax.tick_params(direction='in', width=0.5, length=10, which='major')
        ax.tick_params(direction='in', width=0.5, length=4,  which='minor')
        ax.minorticks_on()
        ax.set_title(title, pad=16, fontsize=18)
        ax.set_xlabel('Time (BJD \u2212 2\u202f457\u202f000)', fontsize=14)
        ax.set_ylabel('Normalized Flux', fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir_by_sector, f"{title.replace(' ', '_')}.png"),
                    facecolor='white', dpi=240)
        plt.close(fig)

        cleaned_lcs.append(lc_cln)

    lc_full = vstack(cleaned_lcs, join_type="inner", metadata_conflicts="silent")

    # ---- 4. Transit search --------------------------------------------------
    print('Searching for planets\n')

    all_results_full = planet_search(
        lc_full, stellar_params,
        max_planets=10,
        sde_threshold=params['SDE_THRESHOLD'],
        snr_threshold=params['SNR_THRESHOLD'],
        period_max=PERIOD_MAX,
        min_transits=params['MIN_TRANSITS'],
        mask_mult=1.4,
        odd_even_sigma=params['ODD_EVEN_SIGMA'],
        depth_scatter_nsig=params['DEPTH_SCATTER_NSIG'],
        secondary_depth=params['SECONDARY_DEPTH'],
        target=target,
        outdir=outdir,
        outdir_plots=outdir_plots,
        outdir_tls=outdir_tls,
    )

    # ---- 5. Outputs ---------------------------------------------------------
    tgt = target.replace(' ', '_').replace('.', '-')
    if not all_results_full:
        print("No planets detected. Skipping planet output generation.")
    else:
        results_table = Table(all_results_full)
        human_readable_table(results_table).write(
            os.path.join(outdir, f'{tgt}_summary.txt'),
            format='ascii.fixed_width_two_line', overwrite=True
        )
        results_table.write(
            os.path.join(outdir, f'{tgt}_report.ecsv'), overwrite=True
        )
        make_pyorbit_yaml(all_results_full, stellar_params, target, outdir_pyorbit)
        print('Summary tables written.')

    print('Exporting the Lightcurves\n')    
    # Light curves
    lc_to_dat(lc_full, 'time', 'flux', 'flux_err',
              os.path.join(outdir_pyorbit, f'{tgt}_lc_clean.dat'))
    lc_to_dat(lc_full, 'time', 'pdcsap_flux_norm', 'pdcsap_flux_err_norm',
              os.path.join(outdir_pyorbit, f'{tgt}_lc_pdcsap_norm.dat'))
    lc_to_dat(lc_full, 'time', 'sap_flux_norm', 'sap_flux_err_norm',
              os.path.join(outdir_pyorbit, f'{tgt}_lc_sap_norm.dat'))

    # Out-of-transit mask
    from transitleastsquares import transit_mask
    t_arr, _, _ = time_flux_err(lc_full)
    intransit   = np.zeros(len(t_arr), dtype=bool)
    for res in all_results_full:
        intransit |= transit_mask(t_arr, res['p'], 1.4 * res['t14'], res['T0'])
    lc_oot = lc_full[~intransit]

    lc_pdcsap_bin = bin_lc(lc_oot, 4 * u.h,
                            flux_kw='pdcsap_flux_norm',
                            flux_err_kw='pdcsap_flux_err_norm')
    lc_to_dat(lc_pdcsap_bin, 'time', 'pdcsap_flux_norm', 'pdcsap_flux_err_norm',
              os.path.join(outdir_pyorbit, f'{tgt}_lc_pdcsap_norm_oot_bin4h.dat'))

    lc_sap_bin = bin_lc(lc_oot, 4 * u.h,
                         flux_kw='sap_flux_norm',
                         flux_err_kw='sap_flux_err_norm')
    lc_to_dat(lc_sap_bin, 'time', 'sap_flux_norm', 'sap_flux_err_norm',
              os.path.join(outdir_pyorbit, f'{tgt}_lc_sap_norm_oot_bin4h.dat'))

    print(f"\nDone: {target}\n")



# ---- Main loop --------------------------------------------------------------
n      = len(targets)
failed = []

for i, target in enumerate(targets):
    if n > 1:
        print(f"\n{'='*60}")
        print(f"  Processing target {i+1}/{n}: {target}")
        print(f"{'='*60}\n")
    else:
        print(f"\nProcessing: {target}\n")

    try:
        run_target(target, params)
    except Exception as e:
        print(f"Error processing {target}: {e}")
        traceback.print_exc()
        failed.append(target)

if n > 1:
    print(f"\n{'='*60}")
    print(f"Done. {n - len(failed)}/{n} targets completed.")
    if failed:
        print(f"Failed: {failed}")
    print(f"{'='*60}\n")

print("\nThank you for using fulmar :)\n")
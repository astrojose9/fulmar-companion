# FULMAR — Follow-Up Lightcurves Multitool Assisting Radial-velocities

**Author:** José Rodrigues (ORCID [0000-0001-5164-3602](https://orcid.org/0000-0001-5164-3602))

> **Note:** This repository accompanies the PhD thesis *Mitigating stellar activity to characterise Earth-sized exoplanets* (University of Porto / FCUP, 2026). It is shared for reproducibility purposes. A cleaned, standalone version is in preparation for publication.

---

## Overview

FULMAR analyses TESS 2-minute cadence light curves to identify transiting planet candidates and prepare follow-up radial-velocity observations. For each target it:

1. Retrieves stellar parameters from TIC v8, Gaia DR3, and SWEET-Cat.
2. Downloads SPOC light curves via `lightkurve`.
3. Detrends each sector with a biweight filter (`wotan`).
4. Searches for transit signals iteratively with the Transit Least Squares algorithm.
5. Applies basic vetting (SDE, SNR, odd/even, secondary eclipse, depth scatter).
6. Exports cleaned light curves and a starter `pyorbit` configuration for joint photometry + RV modelling.

The example configuration `TOI-512.yaml` should recover TOI-512b (Rodrigues et (2025))(https://ui.adsabs.harvard.edu/abs/2025A%26A...695A.237R/abstract). `TOI-174.yaml` recovers all five planets of HD 23472 published in (Barros et al. (2022))(https://ui.adsabs.harvard.edu/abs/2022A&A...665A.154B)

---

## Repository contents

| File | Description |
|---|---|
| `fulmar.py` | Main entry point |
| `fulmar_utils.py` | Helper functions (detrending, TLS wrapper, output writers) |
| `fulmar_config.py` | Default configuration (overridden by a run `.yaml`) |
| `TOI-174.yaml` | Example run file - HD 23472 / TOI-174 (5-planet system) |
| `TOI-512.yaml` | Example run file - TOI-512 (1-planet system) |
| `FULMAR_companion_clean.ipynb` | Companion notebook for interactive inspection |

---

## Installation

### 1. Create and activate a dedicated environment

```bash
conda create --name fulmar python=3.11
conda activate fulmar
```

### 2. Install `transitleastsquares` (Talens fork — required)

The standard PyPI release of `transitleastsquares` is **not** used. Install the
Geert Jan Talens fork directly from GitHub:

```bash
pip install git+https://github.com/talensgj/tls.git
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> `astroquery` queries the MAST archive and the Gaia TAP service; an internet
> connection is required at runtime.

---

## Usage

**Single target by name (TIC ID, TOI number, or common name):**

```bash
python fulmar.py -t "HD 23472"
python fulmar.py -t TOI-512
```

**Target list (one per line, `#` for comments):**

```bash
python fulmar.py targets.txt
```

**Run file with shared parameter overrides:**

```bash
python fulmar.py TOI-174.yaml
```

### Run file format

```yaml
targets:
  - HD 23472
  - TOI-512

parameters:            # optional — overrides fulmar_config.py
  MODE: DATE
  DATE_START: '2018-07-25'
  DATE_END:   '2021-11-15'
  PERIOD_MAX: 42
  SDE_THRESHOLD: 9.0
  SNR_THRESHOLD: 3.0
  MIN_TRANSITS: 2
```

---

## Configuration

All defaults live in `fulmar_config.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `MODE` | `FAST` | Sector selection strategy: `FULL`, `FAST`, `SINGLE`, or `DATE` |
| `MAX_SECTORS` | `5` | Maximum sectors used in `FAST` mode |
| `GAP_TOLERANCE` | `0` | Maximum missing sectors tolerated in `FAST` mode |
| `DATE_START` / `DATE_END` | `None` | Date range for `DATE` mode (BTJD, BJD, or ISO string) |
| `PERIOD_MAX` | `None` | Upper period limit for the transit search (days) |
| `SDE_THRESHOLD` | `9.0` | Minimum signal detection efficiency |
| `SNR_THRESHOLD` | `4.0` | Minimum signal-to-noise ratio |
| `MIN_TRANSITS` | `2` | Minimum number of transits required |

---

## Outputs

Results are written to `fulmar_output/<target>/`:

```
fulmar_output/HD_23472/
├── plots/
│   ├── lc_by_sector/          # per-sector detrended light curves
│   └── ...                    # TLS periodograms and folded transits
├── tls/                       # raw TLS result files
├── pyorbit/
│   ├── HD_23472_lc_clean.dat          # detrended, sigma-clipped light curve
│   ├── HD_23472_lc_pdcsap_norm.dat    # normalised PDCSAP light curve
│   ├── HD_23472_lc_sap_norm.dat       # normalised SAP light curve
│   ├── HD_23472_lc_pdcsap_norm_oot_bin4h.dat  # OOT, binned to 4 h
│   └── *.yaml                         # starter pyorbit configuration
├── HD_23472_summary.txt       # human-readable detection table
└── HD_23472_report.ecsv       # machine-readable detection table
```


---

## Licence

This code is shared for reproducibility. All rights reserved pending the publication of the standalone paper. Please contact the author before reuse.
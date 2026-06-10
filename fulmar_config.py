# ============================================================
# FULMAR default configuration
# ============================================================
# These values are used unless overridden by a run .yaml file.

# ---- Sector selection -------------------------------------------------------
MODE          = 'FAST'  # 'FULL', 'FAST', 'SINGLE', 'DATE'. Default is 'FULL'.
GAP_TOLERANCE = 0       # int, default=0. Max missing sectors in FAST mode.
MAX_SECTORS   = 5       # int, default=6. Max sectors to use in FAST mode.

# DATE mode: set the range of interest.
# Accepted formats: BTJD float string, BJD float string, ISO string, or None.
# None means no bound (from first or to last available sector).
DATE_START = None        # e.g. '1500.0', '2019-01-01', or None
DATE_END   = None        # e.g. '1800.0', '2021-06-01', or None

# ---- Transit search ---------------------------------------------------------
PERIOD_MAX = None   # float or None. Maximum period to search for (days).

# ---- Transit vetting --------------------------------------------------------
SDE_THRESHOLD      = 9.0  # Minimum SDE to consider a detection significant.
SNR_THRESHOLD      = 4.0  # Minimum SNR to consider a detection significant.
MIN_TRANSITS       = 2    # Minimum number of transits. Set to 1 for mono-transit search.
ODD_EVEN_SIGMA     = 5.0  # Flag if odd/even depth mismatch exceeds this (sigma).
DEPTH_SCATTER_NSIG = 5.0  # Flag if any individual transit depth deviates by more than this.
SECONDARY_DEPTH    = 0.5  # Flag if secondary eclipse depth exceeds this fraction of primary depth.
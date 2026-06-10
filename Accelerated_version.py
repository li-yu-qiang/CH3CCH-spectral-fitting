import numpy as np
from astropy import constants
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator


c = constants.c.cgs.value       # Speed of light (cm/s)
kB = constants.k_B.cgs.value    # Boltzmann coefficient (erg/K)
h = constants.h.cgs.value       # Planck constant (erg*s)

SQRT_2PI = np.sqrt(2.0 * np.pi)
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


# -----------------------------------------------------------------------------
# Catalog loading and one-time preprocessing
# -----------------------------------------------------------------------------

colspecs = [(0, 13), (24, 35), (37, 47), (48, 51), (61, 63), (63, 65)]

spec_df = pd.read_fwf("c040502.cat", header=None, colspecs=colspecs)
spec_df = spec_df.replace({"A": "10", "B": "11"}, regex=True)

# Force numeric ndarray, avoid object dtype
spec_df = spec_df.apply(pd.to_numeric)
spec = spec_df.to_numpy(dtype=np.float64)

freq_all = spec[:, 0]
A_log_all = spec[:, 1]
El_cm_all = spec[:, 2]
g_upper_all = spec[:, 3]
J_upper_all = spec[:, 4].astype(np.int64)
K_upper_all = spec[:, 5].astype(np.int64)

nu_Hz_all = freq_all * 1.0e6
El_all = El_cm_all * h * c / kB
Eu_all = El_all + h * nu_Hz_all / kB
A_all = 10.0 ** A_log_all

# Nuclear spin statistical weight for CH3CCH K ladders
gI_all = np.ones_like(K_upper_all, dtype=np.float64)
gI_all[(K_upper_all % 3 == 0) & (K_upper_all != 0)] = 2.0

# One-time part of partition-function summand
partition_weight_all = 2.0 * (2.0 * (J_upper_all - 1.0) + 1.0) * gI_all


# -----------------------------------------------------------------------------
# Partition function: precomputed lookup table + interpolation
# -----------------------------------------------------------------------------

def _partition_exact_numpy(T):
    """Exact partition function using the catalog, mainly for building the table."""
    return np.sum(partition_weight_all * np.exp(-El_all / T))


def build_partition_table(T_min=1.0, T_max=500.0, n_grid=50000):
    """Build Q(T) lookup table. Extend range if you fit outside 1--500 K."""
    T_grid = np.linspace(float(T_min), float(T_max), int(n_grid))
    Q_grid = np.empty_like(T_grid)

    for i, T in enumerate(T_grid):
        Q_grid[i] = _partition_exact_numpy(T)

    return T_grid, Q_grid


T_PARTITION_GRID, Q_PARTITION_GRID = build_partition_table()


@njit(cache=True, fastmath=True)
def partition_interp_numba(T, T_grid, Q_grid):
    """Linear interpolation for Q(T), compiled with numba when available."""
    if T <= T_grid[0]:
        return Q_grid[0]

    if T >= T_grid[-1]:
        return Q_grid[-1]

    lo = 0
    hi = T_grid.size - 1

    while hi - lo > 1:
        mid = (lo + hi) // 2

        if T_grid[mid] <= T:
            lo = mid
        else:
            hi = mid

    t0 = T_grid[lo]
    t1 = T_grid[hi]
    q0 = Q_grid[lo]
    q1 = Q_grid[hi]

    return q0 + (q1 - q0) * (T - t0) / (t1 - t0)


def partition(T):
    """Fast interpolated partition function, keeping the original function name."""
    return float(np.interp(T, T_PARTITION_GRID, Q_PARTITION_GRID))


def partition_exact(T):
    """Exact partition function, useful for validation/debugging."""
    return float(_partition_exact_numpy(T))


# -----------------------------------------------------------------------------
# Line selection and line-cache creation
# -----------------------------------------------------------------------------

def para(J, K0, K1):
    """Select catalog lines by upper J and K range, preserving original behavior."""
    mask = (J_upper_all == J) & (K_upper_all >= K0) & (K_upper_all <= K1)
    arr = spec[mask]

    return arr[::-1].copy()


def prepare_line_cache(p):
    """
    Precompute all catalog-dependent constants for selected K components.

    Returns:
    - E_us: upper-state energies in K
    - v_offsets: velocity offsets relative to first selected component, km/s
    - line_const: multiplicative line constants except T_rot, N_tot, sigma_v, Q(T)
    """
    p = np.asarray(p, dtype=np.float64)

    nu_0s = p[:, 0] * 1.0e6
    A_us = 10.0 ** p[:, 1]
    E_us = p[:, 2] * h * c / kB + h * nu_0s / kB
    g_us = p[:, 3]

    v_offsets = c / 1.0e5 * (1.0 - nu_0s / nu_0s[0])
    line_const = g_us * h * c**3 * A_us / (
        8.0 * np.pi * nu_0s**2 * 1.0e3 * kB
    )

    return (
        E_us.astype(np.float64),
        v_offsets.astype(np.float64),
        line_const.astype(np.float64),
    )


# -----------------------------------------------------------------------------
# Fast model: numba-compiled + numpy fallback
# -----------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def multi_gaussian_numba(
    vel,
    T_rot,
    logN_tot,
    sigma_v,
    v_0,
    E_us,
    v_offsets,
    line_const,
    T_grid,
    Q_grid,
):
    N_tot = 10.0 ** logN_tot
    Z = partition_interp_numba(T_rot, T_grid, Q_grid)
    phi = 1.0 / (SQRT_2PI * sigma_v)

    n_vel = vel.size
    n_line = E_us.size

    model = np.zeros(n_vel, dtype=np.float64)

    for j in range(n_line):
        center = v_offsets[j] + v_0

        amp = N_tot * np.exp(-E_us[j] / T_rot) / Z
        amp *= line_const[j] / T_rot * phi

        for i in range(n_vel):
            x = (vel[i] - center) / sigma_v
            model[i] += amp * np.exp(-0.5 * x * x)

    return model


def multi_gaussian_vectorized(vel, T_rot, logN_tot, sigma_v, v_0, line_cache):
    """Pure-numpy vectorized fallback / validation version."""
    E_us, v_offsets, line_const = line_cache

    N_tot = 10.0 ** logN_tot
    Z = partition(T_rot)
    phi = 1.0 / (SQRT_2PI * sigma_v)

    centers = v_offsets + v_0

    amp = N_tot * np.exp(-E_us / T_rot) / Z
    amp *= line_const / T_rot * phi

    x = (vel[:, None] - centers[None, :]) / sigma_v

    return np.exp(-0.5 * x**2) @ amp


def multi_gaussian(vel, T_rot, logN_tot, sigma_v, v_0, p=None, line_cache=None):
    """
    Fast replacement for original multi_gaussian.

    Pass either:
    - p=selected catalog rows
    - line_cache=prepare_line_cache(p)
    """
    vel = np.asarray(vel, dtype=np.float64)

    if line_cache is None:
        if p is None:
            raise ValueError("Pass either p or line_cache.")
        line_cache = prepare_line_cache(p)

    E_us, v_offsets, line_const = line_cache

    if NUMBA_AVAILABLE:
        return multi_gaussian_numba(
            vel,
            float(T_rot),
            float(logN_tot),
            float(sigma_v),
            float(v_0),
            E_us,
            v_offsets,
            line_const,
            T_PARTITION_GRID,
            Q_PARTITION_GRID,
        )

    return multi_gaussian_vectorized(
        vel,
        T_rot,
        logN_tot,
        sigma_v,
        v_0,
        line_cache,
    )


# -----------------------------------------------------------------------------
# Fitting function with plot switch
# -----------------------------------------------------------------------------

def fitting(
    J,
    K0,
    K1,
    velocity,
    intensity,
    v_0_init,
    FWHM_init,
    Trot=(15, 100),
    Ntot=(13, 16),
    sigmav=(0, 10),
    v0=(-10, 10),
    source_name=None,
    source=None,
    plot=True,
    warmup_numba=True,
):
    velocity = np.asarray(velocity, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    bounds = (
        [Trot[0], Ntot[0], sigmav[0], v0[0]],
        [Trot[1], Ntot[1], sigmav[1], v0[1]],
    )

    p = para(J, K0, K1)
    line_cache = prepare_line_cache(p)

    p0 = [
        30.0,
        14.0,
        FWHM_init * FWHM_TO_SIGMA,
        v_0_init,
    ]

    # Warm up numba before curve_fit so compilation time is not mixed into fitting.
    # In timing loop, this is already compiled, so overhead is tiny.
    if NUMBA_AVAILABLE and warmup_numba:
        _ = multi_gaussian(
            velocity[: min(3, velocity.size)],
            *p0,
            line_cache=line_cache,
        )

    popt, pcov = curve_fit(
        lambda vel, T_rot, logN_tot, sigma_v, v_0:
            multi_gaussian(
                vel,
                T_rot,
                logN_tot,
                sigma_v,
                v_0,
                line_cache=line_cache,
            ),
        velocity,
        intensity,
        p0=p0,
        maxfev=10000,
        bounds=bounds,
    )

    T_rot_fit, logN_tot_fit, sigma_v_fit, v_0_fit = popt
    errors = np.sqrt(np.diag(pcov))

    if plot:
        f = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        plt.plot(
            velocity,
            intensity,
            drawstyle="steps-mid",
            label="Spectrum",
            color="black",
            lw=1,
        )

        plt.plot(
            velocity,
            multi_gaussian(velocity, *popt, line_cache=line_cache),
            label="Best-fit model",
            color="red",
            lw=2,
        )

        if source_name is not None:
            plt.text(
                0.02,
                0.95,
                source_name,
                transform=ax.transAxes,
                color="r",
                fontsize=12,
            )

        plt.text(
            0.02,
            0.90,
            r"$T_\mathrm{{rot}}$ = {:.2f}$\pm${:.2f} K".format(
                T_rot_fit,
                errors[0],
            ),
            color="r",
            transform=ax.transAxes,
            fontsize=12,
        )

        plt.text(
            0.02,
            0.85,
            r"$\log N_\mathrm{{tot}}$ = {:.2f}$\pm${:.2f}".format(
                logN_tot_fit,
                errors[1],
            ),
            color="r",
            transform=ax.transAxes,
            fontsize=12,
        )

        plt.text(
            0.02,
            0.80,
            r"$V_\mathrm{{lsr}}$ = {:.2f}$\pm${:.2f} km/s".format(
                v_0_fit,
                errors[3],
            ),
            color="r",
            transform=ax.transAxes,
            fontsize=12,
        )

        plt.text(
            0.02,
            0.75,
            r"$\Delta V$ = {:.2f}$\pm${:.2f} km/s".format(
                sigma_v_fit * SIGMA_TO_FWHM,
                errors[2] * SIGMA_TO_FWHM,
            ),
            color="r",
            transform=ax.transAxes,
            fontsize=12,
        )

        plt.xlabel("Velocity (km/s)", fontsize=14)
        plt.ylabel(r"$T_\mathrm{{A}}$ (K)", fontsize=14)
        plt.legend(fontsize=12)

        output_pdf = f"{source or 'fit'}.pdf"
        plt.savefig(output_pdf, bbox_inches="tight")

        f.clear()
        plt.close()

    return popt, pcov, errors


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    data = np.loadtxt("G03519-0074_CH3CCH.dat")

    velocity = data[:, 0].astype(np.float64)

    eta_mb = 0.225
    intensity = data[:, 1].astype(np.float64) / eta_mb

    fit_kwargs = dict(
        J=5,
        K0=0,
        K1=2,
        velocity=velocity,
        intensity=intensity,
        v_0_init=34,
        FWHM_init=3,
        v0=(30, 40),
        sigmav=(
            3 * FWHM_TO_SIGMA - 1,
            3 * FWHM_TO_SIGMA + 1,
        ),
        source_name="G035.19-00.74",
        source="G035",
    )

    # -------------------------------------------------------------------------
    # Single run: choose whether to plot
    # -------------------------------------------------------------------------

    result = fitting(
        **fit_kwargs,
        plot=True,          
        warmup_numba=True,
    )

    popt, pcov, errors = result

    print("Single run fitting result:")
    print(f"T_rot       = {popt[0]:.6f} +/- {errors[0]:.6f} K")
    print(f"logN_tot    = {popt[1]:.6f} +/- {errors[1]:.6f}")
    print(f"sigma_v     = {popt[2]:.6f} +/- {errors[2]:.6f} km/s")
    print(
        f"FWHM        = {popt[2] * SIGMA_TO_FWHM:.6f} "
        f"+/- {errors[2] * SIGMA_TO_FWHM:.6f} km/s"
    )
    print(f"v_0         = {popt[3]:.6f} +/- {errors[3]:.6f} km/s")


# ==============================================================
# Streamlit App: Random Laser ASC Analyzer
# Complete Version with ND Correction, Energy Calibration & All Plots
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from io import StringIO, BytesIO
import zipfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import re
from datetime import datetime

# ==============================================================
# CHECK FOR KALEIDO (IMAGE EXPORT)
# ==============================================================

try:
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

# ==============================================================
# DATA CLASSES
# ==============================================================

@dataclass
class FitResult:
    """Data class for spectral analysis results"""
    peak_wavelength: float
    peak_intensity: float
    fwhm: float
    integrated_intensity: float
    fit_y: np.ndarray
    r_squared: float
    snr: float
    fit_params: Dict
    fit_success: bool = True


@dataclass
class ThresholdAnalysis:
    """Data class for threshold detection results"""
    threshold_qs: Optional[float]
    threshold_energy: Optional[float]
    slope_below: float
    slope_above: float
    threshold_found: bool


# ==============================================================
# SAMPLE METADATA EXTRACTION
# ==============================================================

def extract_thickness(filename: str) -> Optional[float]:
    """Extract thickness from filename"""
    patterns = [
        r'UL[_\s-]*(\d+\.?\d*)\s*mm',
        r'LL[_\s-]*(\d+\.?\d*)\s*mm',
        r'(\d+\.?\d*)\s*mm',
        r't[_\s-]*(\d+\.?\d*)',
        r'thickness[_\s-]*(\d+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def extract_concentration(filename: str) -> Optional[Dict[str, float]]:
    """Extract concentration from filename (handles upper and lower layers)"""
    conc_data = {'upper': None, 'lower': None}

    ul_pattern = r'UL[_\s-]*(\d+\.?\d*)\s*%\s*IL'
    ul_match = re.search(ul_pattern, filename, re.IGNORECASE)
    if ul_match:
        conc_data['upper'] = float(ul_match.group(1))

    ll_pattern = r'LL[_\s-]*(\d+\.?\d*)\s*%\s*IL'
    ll_match = re.search(ll_pattern, filename, re.IGNORECASE)
    if ll_match:
        conc_data['lower'] = float(ll_match.group(1))

    if conc_data['upper'] is None and conc_data['lower'] is None:
        simple_patterns = [
            r'(\d+\.?\d*)\s*%',
            r'(\d+)p(\d+)',
            r'c[_\s-]*(\d+\.?\d*)',
            r'conc[_\s-]*(\d+\.?\d*)',
        ]
        for pattern in simple_patterns:
            match = re.search(pattern, filename.lower())
            if match:
                if 'p' in pattern and len(match.groups()) > 1:
                    conc_data['upper'] = float(f"{match.group(1)}.{match.group(2)}")
                else:
                    conc_data['upper'] = float(match.group(1))
                break

    if conc_data['upper'] is None and conc_data['lower'] is None:
        return None
    return conc_data


def extract_nd(filename: str) -> float:
    """Extract ND/OD filter value from filename"""
    patterns = [
        r'OD\s*[=_-]*\s*(\d+\.?\d*)',
        r'ND\s*[=_-]*\s*(\d+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return 0.0


def extract_qs(filename: str) -> float:
    """Extract Q-switch value from filename"""
    patterns = [
        r'QS[_\s-]+(\d+\.?\d*)',
        r'QS(\d+\.?\d*)',
        r'qs[_\s-]*(\d+\.?\d*)',
        r'q[_\s-]*(\d+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            try:
                return float(match.group(1))
            except Exception:
                continue
    return np.nan


def extract_dye_amount(filename: str) -> Optional[float]:
    """Extract dye amount from filename"""
    patterns = [
        r'(\d+\.?\d*)\s*mg\s*R6G',
        r'(\d+\.?\d*)\s*mg',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def extract_repetitions(filename: str) -> Optional[int]:
    """Extract number of repetitions from filename"""
    patterns = [r'(\d+)\s*rep']
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def get_sample_label(thickness: Optional[float], concentration: Optional[Dict],
                     dye_amount: Optional[float] = None) -> str:
    """Generate a comprehensive label for the sample"""
    parts = []
    if thickness is not None:
        parts.append(f"UL {thickness}mm")
    if concentration is not None:
        if concentration.get('upper') is not None and concentration.get('lower') is not None:
            parts.append(f"UL {concentration['upper']}%IL | LL {concentration['lower']}%IL")
        elif concentration.get('upper') is not None:
            parts.append(f"{concentration['upper']}%IL")
        elif concentration.get('lower') is not None:
            parts.append(f"LL {concentration['lower']}%IL")
    if dye_amount is not None:
        parts.append(f"{dye_amount}mg R6G")
    if parts:
        return " | ".join(parts)
    return "No Label"


def get_short_label(thickness: Optional[float], concentration: Optional[Dict]) -> str:
    """Generate a short label for plotting"""
    parts = []
    if thickness is not None:
        parts.append(f"{thickness}mm")
    if concentration is not None:
        if concentration.get('upper') is not None and concentration.get('lower') is not None:
            parts.append(f"UL{concentration['upper']}%-LL{concentration['lower']}%")
        elif concentration.get('upper') is not None:
            parts.append(f"{concentration['upper']}%")
    if parts:
        return " ".join(parts)
    return "No Label"


def apply_nd_correction(counts: np.ndarray, nd_value: float) -> np.ndarray:
    """Apply ND filter correction: multiply by 10^ND"""
    if nd_value == 0:
        return counts
    correction_factor = 10 ** nd_value
    return counts * correction_factor


# ==============================================================
# ENERGY CALIBRATION FUNCTIONS (SMART PARSER)
# ==============================================================

@st.cache_data
def parse_energy_file_simple(text: str) -> dict:
    """
    Parse a simple energy table like:

        200 190 180 ... 110
        7.48E-06 ...
        ...
        (10 rows of readings)
        mean-row (optional)
        OD-row (optional)
    """

    from io import StringIO

    # Load as CSV/TSV by detecting whitespace or tab
    df = pd.read_csv(
        StringIO(text),
        sep=r"\s+",
        engine="python",
        header=None
    )

    # First row = QS levels
    qs_levels = df.iloc[0].dropna().astype(float).tolist()
    n_qs = len(qs_levels)

    # Energy readings = next 10 rows
    readings = df.iloc[1:11, :n_qs].astype(float)

    # Optional mean and OD
    od_row = None
    if df.shape[0] > 12:
        possible_od = df.iloc[-1]
        if any("2" in str(x) or "0" in str(x) for x in possible_od):
            od_row = pd.to_numeric(possible_od[:n_qs], errors="coerce")

    # Build map
    energy_map = {}
    for i, qs in enumerate(qs_levels):
        vals = readings.iloc[:, i].dropna().astype(float)

        # Convert J → mJ
        vals_mJ = vals * 1000

        energy_map[qs] = {
            "mean": vals_mJ.mean(),
            "std": vals_mJ.std(),
            "readings": vals_mJ.tolist(),
            "n_readings": len(vals_mJ),
            "od": float(od_row[i]) if od_row is not None else 0.0,
        }

    return energy_map



def interpolate_energy(qs_value: float, energy_map: Dict[float, Dict]) -> Tuple[float, float]:
    """
    Given a QS value and the energy_map from parse_energy_file, return
    (mean_energy_mJ, std_energy_mJ) using interpolation / nearest-neighbour.
    """
    if energy_map is None or not energy_map or np.isnan(qs_value):
        return np.nan, np.nan

    qs_keys = np.array(sorted(energy_map.keys()), dtype=float)
    means = np.array([energy_map[q]['mean'] for q in qs_keys], dtype=float)
    stds = np.array([energy_map[q]['std'] for q in qs_keys], dtype=float)

    # Exact match
    if qs_value in energy_map:
        d = energy_map[qs_value]
        return float(d['mean']), float(d['std'])

    # Outside range: use nearest neighbour
    if qs_value <= qs_keys.min():
        idx = 0
        return float(means[idx]), float(stds[idx])
    if qs_value >= qs_keys.max():
        idx = -1
        return float(means[idx]), float(stds[idx])

    # Linear interpolation inside range
    mean_interp = float(np.interp(qs_value, qs_keys, means))
    std_interp = float(np.interp(qs_value, qs_keys, stds))
    return mean_interp, std_interp


# ==============================================================
# CORE SPECTRAL ANALYSIS FUNCTIONS
# ==============================================================

def lorentzian(x: np.ndarray, A: float, x0: float, gamma: float, y0: float) -> np.ndarray:
    """Lorentzian lineshape function"""
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + y0


def calculate_r_squared(y_actual: np.ndarray, y_fit: np.ndarray) -> float:
    """Calculate coefficient of determination (R²)"""
    ss_res = np.sum((y_actual - y_fit) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def calculate_snr(signal: np.ndarray, noise_percentile: int = 10) -> float:
    """Calculate signal-to-noise ratio"""
    noise = np.percentile(signal, noise_percentile)
    peak = np.max(signal)
    return peak / noise if noise > 0 else np.inf


@st.cache_data
def analyze_spectrum(wl: np.ndarray, counts: np.ndarray) -> FitResult:
    """Perform Lorentzian fitting and extract spectral parameters"""
    try:
        peak_idx = np.argmax(counts)
        x0_init = wl[peak_idx]
        baseline_est = np.percentile(counts, 5)
        A_init = np.max(counts) - baseline_est
        y0_init = baseline_est

        half_max = baseline_est + A_init / 2
        above_half = counts > half_max

        if np.sum(above_half) > 2:
            indices = np.where(above_half)[0]
            width = wl[indices[-1]] - wl[indices[0]]
            gamma_init = max(width / 2, 0.1)
        else:
            gamma_init = (wl.max() - wl.min()) / 10

        bounds = ([0, wl.min(), 0, -np.inf], [np.inf, wl.max(), np.inf, np.inf])
        p0 = [A_init, x0_init, gamma_init, y0_init]

        try:
            popt, pcov = curve_fit(lorentzian, wl, counts, p0=p0, maxfev=50000, method='lm')
        except Exception:
            popt, pcov = curve_fit(
                lorentzian, wl, counts, p0=p0, bounds=bounds, maxfev=50000, method='trf'
            )

        A, x0, gamma, y0 = popt
        A = abs(A)
        gamma = abs(gamma)

        fwhm = 2 * gamma
        fit_y = lorentzian(wl, A, x0, gamma, y0)
        area = np.trapz(np.maximum(counts - y0, 0), wl)
        r_squared = calculate_r_squared(counts, fit_y)
        snr = calculate_snr(counts)

        if r_squared < 0.3:
            raise ValueError(f"Poor fit quality: R² = {r_squared:.3f}")

        fit_params = {
            'Amplitude': float(A),
            'Center': float(x0),
            'Gamma': float(gamma),
            'Baseline': float(y0),
            'Std_Errors': [float(x) for x in np.sqrt(np.diag(pcov))],
        }

        return FitResult(
            float(x0),
            float(A + y0),
            float(fwhm),
            float(area),
            fit_y,
            float(r_squared),
            float(snr),
            fit_params,
            fit_success=True,
        )

    except Exception as e:
        peak_idx = np.argmax(counts)
        peak_wl = wl[peak_idx]
        peak_int = counts[peak_idx]
        half_max = (peak_int + np.min(counts)) / 2
        above_half = counts > half_max

        if np.sum(above_half) > 2:
            indices = np.where(above_half)[0]
            fwhm_estimate = wl[indices[-1]] - wl[indices[0]]
        else:
            fwhm_estimate = np.nan

        return FitResult(
            float(peak_wl),
            float(peak_int),
            float(fwhm_estimate) if not np.isnan(fwhm_estimate) else np.nan,
            float(np.trapz(counts - np.min(counts), wl)),
            counts.copy(),
            0.0,
            float(calculate_snr(counts)),
            {'error': str(e)},
            fit_success=False,
        )


# ==============================================================
# THRESHOLD DETECTION
# ==============================================================

def detect_threshold(x_values: np.ndarray, intensities: np.ndarray, min_points: int = 3) -> ThresholdAnalysis:
    """Detect lasing threshold using broken-stick algorithm"""
    if len(x_values) < 2 * min_points:
        return ThresholdAnalysis(None, None, 0, 0, False)

    try:
        idx = np.argsort(x_values)
        x_sorted = x_values[idx]
        int_sorted = intensities[idx]

        best_threshold = None
        best_r2_sum = -np.inf
        best_slopes = (0.0, 0.0)

        for i in range(min_points, len(x_sorted) - min_points):
            below = np.polyfit(x_sorted[:i], int_sorted[:i], 1)
            above = np.polyfit(x_sorted[i:], int_sorted[i:], 1)

            r2_below = calculate_r_squared(int_sorted[:i], np.polyval(below, x_sorted[:i]))
            r2_above = calculate_r_squared(int_sorted[i:], np.polyval(above, x_sorted[i:]))

            r2_sum = r2_below + r2_above
            if r2_sum > best_r2_sum and above[0] > below[0]:
                best_r2_sum = r2_sum
                best_threshold = x_sorted[i]
                best_slopes = (float(below[0]), float(above[0]))

        found = best_threshold is not None and best_slopes[1] > 2 * best_slopes[0]

        # Store threshold in BOTH fields so it works for energy or QS usage
        if found:
            return ThresholdAnalysis(
                threshold_qs=float(best_threshold),
                threshold_energy=float(best_threshold),
                slope_below=best_slopes[0],
                slope_above=best_slopes[1],
                threshold_found=True,
            )
        else:
            return ThresholdAnalysis(None, None, best_slopes[0], best_slopes[1], False)

    except Exception:
        return ThresholdAnalysis(None, None, 0, 0, False)


# ==============================================================
# FILE PARSING (.asc)
# ==============================================================

@st.cache_data
def parse_asc_file(file_content: str, skip_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse .asc file"""
    df = pd.read_csv(
        StringIO(file_content), sep='\t', decimal=',', skiprows=skip_rows, engine='python'
    )
    df = df.dropna(axis=1, how='all')

    if df.shape[1] < 2:
        raise ValueError("File must have at least 2 columns")

    wl = df.iloc[:, 0].to_numpy()
    counts = df.iloc[:, 1:].mean(axis=1).to_numpy()

    return wl, counts


# ==============================================================
# VISUALIZATION FUNCTIONS
# ==============================================================

def create_spectrum_plot(
    wl: np.ndarray,
    counts_raw: np.ndarray,
    counts_corrected: np.ndarray,
    fit_result: FitResult,
    filename: str,
    nd_value: float,
    energy_mean: float = None,
    energy_std: float = None,
) -> go.Figure:
    """Create spectrum plot with fit overlay"""
    fig = go.Figure()

    if nd_value > 0:
        fig.add_trace(
            go.Scatter(
                x=wl,
                y=counts_raw,
                mode='lines',
                name='Raw Data',
                line=dict(color='lightgray', width=2),
                opacity=0.5,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=wl,
            y=counts_corrected,
            mode='lines',
            name='OD-Corrected Data' if nd_value > 0 else 'Data',
            line=dict(color='#2E86AB', width=3),
        )
    )

    if fit_result.fit_success and not np.isnan(fit_result.fwhm):
        fig.add_trace(
            go.Scatter(
                x=wl,
                y=fit_result.fit_y,
                mode='lines',
                name='Lorentzian Fit',
                line=dict(color='red', width=3, dash='dash'),
                opacity=0.8,
            )
        )

        fig.add_vline(
            x=fit_result.peak_wavelength,
            line_dash="dot",
            line_color="green",
            line_width=2,
            annotation_text=f"Peak: {fit_result.peak_wavelength:.2f} nm",
        )

        gamma = fit_result.fwhm / 2
        x0 = fit_result.peak_wavelength
        half_max = fit_result.peak_intensity / 2

        fig.add_trace(
            go.Scatter(
                x=[x0 - gamma, x0 + gamma],
                y=[half_max, half_max],
                mode='markers+text',
                marker=dict(size=12, color='orange', symbol='diamond'),
                name=f'FWHM = {fit_result.fwhm:.2f} nm',
                text=['', f'FWHM={fit_result.fwhm:.2f}nm'],
                textposition='top center',
            )
        )

        fig.add_shape(
            type="line",
            x0=x0 - gamma,
            y0=half_max,
            x1=x0 + gamma,
            y1=half_max,
            line=dict(color="orange", width=2, dash="dash"),
        )

    title_html = f"<b>{filename}</b><br>"
    if energy_mean is not None and not np.isnan(energy_mean):
        title_html += f"<sub>Pump Energy: {energy_mean:.3f}±{energy_std:.3f} mJ</sub><br>"
    if nd_value > 0:
        title_html += f"<sub>OD: {nd_value} (×{10**nd_value:.0f})</sub><br>"
    if fit_result.fit_success:
        title_html += (
            f"<sub>Peak: {fit_result.peak_wavelength:.2f} nm | "
            f"FWHM: {fit_result.fwhm:.2f} nm | R²: {fit_result.r_squared:.4f}</sub>"
        )

    fig.update_layout(
        title=title_html,
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity (counts)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        showlegend=True,
    )

    return fig


def create_threshold_plot(df: pd.DataFrame, threshold: ThresholdAnalysis, use_energy: bool = True) -> go.Figure:
    """Create threshold analysis with smooth curves"""
    from scipy.interpolate import make_interp_spline

    x_col = 'Pump Energy (mJ)' if use_energy and 'Pump Energy (mJ)' in df.columns else 'QS Level'
    x_label = "Pump Energy (mJ)" if use_energy else "Q-Switch Level"

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Integrated Intensity vs {x_label}",
            f"FWHM vs {x_label}",
            f"Peak Wavelength vs {x_label}",
            f"Peak Intensity vs {x_label}",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    valid = df.dropna(subset=[x_col])
    x_values = valid[x_col].values
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]

    int_min, int_max = valid['Integrated Intensity'].min(), valid['Integrated Intensity'].max()
    int_padding = (int_max - int_min) * 0.1 if int_max > int_min else 0
    int_range = [int_min - int_padding, int_max + int_padding]

    peak_min, peak_max = valid['Peak Intensity'].min(), valid['Peak Intensity'].max()
    peak_padding = (peak_max - peak_min) * 0.1 if peak_max > peak_min else 0
    peak_range = [peak_min - peak_padding, peak_max + peak_padding]

    fwhm_min, fwhm_max = valid['FWHM (nm)'].min(), valid['FWHM (nm)'].max()
    fwhm_padding = (fwhm_max - fwhm_min) * 0.1 if fwhm_max > fwhm_min else 0
    fwhm_range = [max(0, fwhm_min - fwhm_padding), fwhm_max + fwhm_padding]

    wl_min, wl_max = valid['Peak λ (nm)'].min(), valid['Peak λ (nm)'].max()
    wl_padding = max((wl_max - wl_min) * 0.1, 2) if wl_max > wl_min else 2
    wl_range = [wl_min - wl_padding, wl_max + wl_padding]

    def create_smooth_curve(x, y, num_points=300):
        if len(x) < 4:
            return x, y
        try:
            spline = make_interp_spline(x, y, k=min(3, len(x) - 1))
            x_smooth = np.linspace(x.min(), x.max(), num_points)
            y_smooth = spline(x_smooth)
            return x_smooth, y_smooth
        except Exception:
            return x, y

    # Plot 1: Integrated Intensity
    y_int = valid['Integrated Intensity'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_int)
    fig.add_trace(
        go.Scatter(
            x=x_smooth, y=y_smooth, mode='lines', line=dict(width=3, color='red', shape='spline'), showlegend=False
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_sorted,
            y=y_int,
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle', line=dict(width=2, color='white')),
            error_x=dict(
                type='data',
                array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                visible='Energy Std (mJ)' in valid.columns,
                thickness=1.5,
                width=4,
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if threshold.threshold_found and threshold.threshold_energy is not None:
        fig.add_vline(
            x=threshold.threshold_energy,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Threshold: {threshold.threshold_energy:.4f} mJ",
            annotation_position="top",
            row=1,
            col=1,
        )

    # Plot 2: FWHM
    y_fwhm = valid['FWHM (nm)'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_fwhm)
    fig.add_trace(
        go.Scatter(
            x=x_smooth, y=y_smooth, mode='lines', line=dict(width=3, color='blue', shape='spline'), showlegend=False
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x_sorted,
            y=y_fwhm,
            mode='markers',
            marker=dict(size=10, color='blue', symbol='circle', line=dict(width=2, color='white')),
            error_x=dict(
                type='data',
                array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                visible='Energy Std (mJ)' in valid.columns,
                thickness=1.5,
                width=4,
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Plot 3: Peak Wavelength
    y_wl = valid['Peak λ (nm)'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_wl)
    fig.add_trace(
        go.Scatter(
            x=x_smooth, y=y_smooth, mode='lines', line=dict(width=3, color='purple', shape='spline'), showlegend=False
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_sorted,
            y=y_wl,
            mode='markers',
            marker=dict(size=10, color='purple', symbol='circle', line=dict(width=2, color='white')),
            error_x=dict(
                type='data',
                array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                visible='Energy Std (mJ)' in valid.columns,
                thickness=1.5,
                width=4,
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Plot 4: Peak Intensity
    y_peak = valid['Peak Intensity'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_peak)
    fig.add_trace(
        go.Scatter(
            x=x_smooth, y=y_smooth, mode='lines', line=dict(width=3, color='orange', shape='spline'), showlegend=False
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x_sorted,
            y=y_peak,
            mode='markers',
            marker=dict(size=10, color='orange', symbol='circle', line=dict(width=2, color='white')),
            error_x=dict(
                type='data',
                array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                visible='Energy Std (mJ)' in valid.columns,
                thickness=1.5,
                width=4,
            ),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text=x_label, row=row, col=col)

    fig.update_yaxes(title_text="Integrated Intensity", range=int_range, row=1, col=1)
    fig.update_yaxes(title_text="FWHM (nm)", range=fwhm_range, row=1, col=2)
    fig.update_yaxes(title_text="Wavelength (nm)", range=wl_range, row=2, col=1)
    fig.update_yaxes(title_text="Counts", range=peak_range, row=2, col=2)

    fig.update_layout(
        height=700,
        showlegend=False,
        template="plotly_white",
        title_text="<b>Threshold Analysis Dashboard</b>",
    )

    return fig


def create_energy_wavelength_plot(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create Energy vs Peak Wavelength plot"""
    from scipy.interpolate import make_interp_spline

    fig = go.Figure()

    if 'Pump Energy (mJ)' not in df.columns or df['Pump Energy (mJ)'].isna().all():
        st.warning("⚠️ Energy calibration data not available")
        return None

    if 'Sample Label Short' in df.columns:
        groups = df.groupby('Sample Label Short')
        colors = [
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf',
        ]

        for idx, (label, group_df) in enumerate(groups):
            if label == "No Label":
                continue

            group_df = group_df.dropna(subset=['Pump Energy (mJ)', 'Peak λ (nm)'])
            if len(group_df) == 0:
                continue

            group_df = group_df.sort_values('Pump Energy (mJ)')
            x_data = group_df['Pump Energy (mJ)'].values
            y_data = group_df['Peak λ (nm)'].values
            color = colors[idx % len(colors)]

            if len(x_data) >= 4:
                try:
                    spline = make_interp_spline(x_data, y_data, k=min(3, len(x_data) - 1))
                    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                    y_smooth = spline(x_smooth)
                    fig.add_trace(
                        go.Scatter(
                            x=x_smooth,
                            y=y_smooth,
                            mode='lines',
                            line=dict(width=3, color=color),
                            name=label,
                            showlegend=True,
                            legendgroup=label,
                        )
                    )
                except Exception:
                    pass

            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    marker=dict(size=12, color=color, symbol='circle', line=dict(width=2, color='white')),
                    name=label,
                    showlegend=False,
                    legendgroup=label,
                    error_x=dict(
                        type='data',
                        array=group_df['Energy Std (mJ)'].values
                        if 'Energy Std (mJ)' in group_df.columns
                        else None,
                        visible='Energy Std (mJ)' in group_df.columns,
                        thickness=1.5,
                        width=4,
                    ),
                    hovertemplate=(
                        f'<b>{label}</b><br>'
                        'Energy: %{x:.4f} mJ<br>'
                        'Peak λ: %{y:.2f} nm<br><extra></extra>'
                    ),
                )
            )

    fig.update_layout(
        title="<b>Peak Wavelength vs Pump Energy</b><br><sub>Grouped by Sample Conditions</sub>",
        xaxis_title="Pump Energy (mJ)",
        yaxis_title="Peak Wavelength (nm)",
        template="plotly_white",
        hovermode="closest",
        height=600,
        showlegend=True,
        legend=dict(
            title="Sample Conditions",
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1,
        ),
    )

    return fig


def create_energy_intensity_plot(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create Energy vs Peak Intensity plot"""
    from scipy.interpolate import make_interp_spline

    fig = go.Figure()

    if 'Pump Energy (mJ)' not in df.columns or df['Pump Energy (mJ)'].isna().all():
        st.warning("⚠️ Energy calibration data not available")
        return None

    if 'Sample Label Short' in df.columns:
        groups = df.groupby('Sample Label Short')
        colors = [
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf',
        ]

        for idx, (label, group_df) in enumerate(groups):
            if label == "No Label":
                continue

            group_df = group_df.dropna(subset=['Pump Energy (mJ)', 'Peak Intensity'])
            if len(group_df) == 0:
                continue

            group_df = group_df.sort_values('Pump Energy (mJ)')
            x_data = group_df['Pump Energy (mJ)'].values
            y_data = group_df['Peak Intensity'].values
            color = colors[idx % len(colors)]

            if len(x_data) >= 4:
                try:
                    spline = make_interp_spline(x_data, y_data, k=min(3, len(x_data) - 1))
                    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                    y_smooth = spline(x_smooth)
                    fig.add_trace(
                        go.Scatter(
                            x=x_smooth,
                            y=y_smooth,
                            mode='lines',
                            line=dict(width=3, color=color),
                            name=label,
                            showlegend=True,
                            legendgroup=label,
                        )
                    )
                except Exception:
                    pass

            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    marker=dict(size=12, color=color, symbol='circle', line=dict(width=2, color='white')),
                    name=label,
                    showlegend=False,
                    legendgroup=label,
                    error_x=dict(
                        type='data',
                        array=group_df['Energy Std (mJ)'].values
                        if 'Energy Std (mJ)' in group_df.columns
                        else None,
                        visible='Energy Std (mJ)' in group_df.columns,
                        thickness=1.5,
                        width=4,
                    ),
                    hovertemplate=(
                        f'<b>{label}</b><br>'
                        'Energy: %{x:.4f} mJ<br>'
                        'Peak Intensity: %{y:.0f}<br><extra></extra>'
                    ),
                )
            )

    fig.update_layout(
        title="<b>Peak Intensity vs Pump Energy</b><br><sub>Grouped by Sample Conditions</sub>",
        xaxis_title="Pump Energy (mJ)",
        yaxis_title="Peak Intensity (counts)",
        template="plotly_white",
        hovermode="closest",
        height=600,
        showlegend=True,
        legend=dict(
            title="Sample Conditions",
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1,
        ),
    )

    return fig


def fig_to_image(fig: go.Figure, format: str, width: int, height: int, scale: int) -> bytes:
    """Convert figure to image"""
    try:
        return fig.to_image(format=format, width=width, height=height, scale=scale, engine="kaleido")
    except Exception as e:
        st.error(f"Image export failed: {str(e)}")
        raise


# ==============================================================
# STREAMLIT APP LAYOUT
# ==============================================================

st.set_page_config(page_title="Random Laser Analyzer", layout="wide", page_icon="🔬")

st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🔬 Random Laser Analyzer</p>', unsafe_allow_html=True)
st.markdown("**Lorentzian fitting • OD correction • Energy calibration • Complete analysis suite**")

if not KALEIDO_AVAILABLE:
    st.warning("⚠️ Image export disabled. Install: `pip install kaleido`")

with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("📁 File Settings", expanded=True):
        skip_rows = st.number_input("Header rows to skip", 0, 100, 38)
        show_individual = st.checkbox("Show individual plots", True)
        show_fit_params = st.checkbox("Show fit parameters", False)
        apply_nd = st.checkbox("Apply OD/ND correction", True)

    if KALEIDO_AVAILABLE:
        with st.expander("💾 Export Settings"):
            image_format = st.selectbox("Format", ["png", "jpeg", "svg", "pdf"])
            image_width = st.number_input("Width (px)", 800, 3000, 1200)
            image_height = st.number_input("Height (px)", 400, 2000, 800)
            image_scale = st.slider("Scale", 1, 5, 2)

    st.markdown("---")
    st.markdown("### 📊 Features")
    st.markdown(
        """
    - ✅ Lorentzian fitting
    - ✅ OD/ND correction
    - ✅ Energy calibration
    - ✅ Wavelength vs Energy
    - ✅ Intensity vs Energy
    - ✅ Threshold detection
    - ✅ Sample grouping

    ### 📝 Filename Format
    `UL_5mm_QS_110_10rep_17mgR6G_UL_5%IL_LL_1%IL_OD=2.asc`

    Extracts:
    - Thickness: `UL_5mm`
    - QS: `QS_110`
    - Conc: `UL_5%IL`, `LL_1%IL`
    - Dye: `17mgR6G`
    - OD: `OD=2`
    """
    )

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Spectrum Files (.asc)")
    uploaded_files = st.file_uploader(
        "Upload spectrum files", accept_multiple_files=True, type=['asc'], key="spectrum_files"
    )

with col2:
    st.subheader("⚡ Energy Calibration")

    with st.expander("📋 Format (Example Layout)"):
        st.markdown(
            """
        Typical layout your parser can handle:

        - A few header rows (sample name, "in J", etc.)
        - One row with QS levels, e.g.:
          `200   190   180   170   160   150 ...`
        - Next ~10 rows: energy readings in J (auto-converted to mJ)
        - Optional row with `OD` or `ND` somewhere below the data
        """
        )

    energy_file = st.file_uploader(
        "Upload energy file (optional)", type=['csv', 'txt', 'tsv', 'xlsx', 'xls'], key="energy_file"
    )

    if energy_file:
        with st.expander("📊 Preview Energy Data", expanded=True):
            try:
                file_bytes = energy_file.read()
                energy_file.seek(0)

                st.info(f"📄 **File:** {energy_file.name} | **Size:** {len(file_bytes)} bytes")

                if energy_file.name.endswith(('.xlsx', '.xls')):
                    st.info("📊 Reading Excel energy file...")
                    energy_map = parse_energy_file(None, energy_file.type, file_bytes)
                else:
                    st.info("📄 Reading text energy file...")
                    energy_content = file_bytes.decode(errors='ignore')
                    with st.expander("🔍 First 5 lines of file"):
                        lines = energy_content.split('\n')[:5]
                        for i, line in enumerate(lines, 1):
                            st.code(f"Row {i}: {line}")
                    energy_map = parse_energy_file(energy_content, energy_file.type, file_bytes)

                if energy_map:
                    st.success(f"✅ Loaded {len(energy_map)} QS levels from energy file")

                    energy_df = pd.DataFrame(
                        [
                            {
                                'QS': qs,
                                'Mean (mJ)': d['mean'],
                                'Std (mJ)': d['std'],
                                'N': d['n_readings'],
                                'OD': d.get('od', 0.0),
                            }
                            for qs, d in energy_map.items()
                        ]
                    ).sort_values('QS', ascending=False)

                    st.dataframe(
                        energy_df.style.format(
                            {
                                'QS': '{:.0f}',
                                'Mean (mJ)': '{:.4f}',
                                'Std (mJ)': '{:.4f}',
                                'N': '{:.0f}',
                                'OD': '{:.1f}',
                            }
                        ),
                        use_container_width=True,
                    )

                    if energy_df['OD'].sum() > 0:
                        unique_ods = energy_df[energy_df['OD'] > 0]['OD'].unique()
                        st.success(
                            "🔍 **OD Filters detected:** "
                            + ", ".join([f"OD={od:.1f}" for od in unique_ods])
                        )
                    else:
                        st.info("ℹ️ No OD filters specified (all OD=0)")

                    fig_cal = go.Figure()
                    fig_cal.add_trace(
                        go.Scatter(
                            x=energy_df['QS'],
                            y=energy_df['Mean (mJ)'],
                            error_y=dict(
                                type='data',
                                array=energy_df['Std (mJ)'],
                                visible=True,
                            ),
                            mode='markers+lines',
                            marker=dict(size=10, color='blue'),
                            line=dict(width=2),
                        )
                    )
                    fig_cal.update_layout(
                        title="Energy Calibration Curve",
                        xaxis_title="QS Level",
                        yaxis_title="Pump Energy (mJ)",
                        template="plotly_white",
                        height=400,
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)
                else:
                    st.error("❌ Failed to parse energy file")
            except Exception as e:
                st.error(f"❌ Error reading energy file: {str(e)}")
                import traceback

                with st.expander("🐛 Full Error Details"):
                    st.code(traceback.format_exc())
    else:
        energy_map = {}

# ==============================================================
# MAIN PROCESSING
# ==============================================================

if uploaded_files:
    st.markdown("---")

    progress_bar = st.progress(0)
    status = st.empty()

    summary_data = []
    plot_zip = BytesIO()
    image_zip = BytesIO() if KALEIDO_AVAILABLE else None
    combined_fig = go.Figure()

    if energy_map:
        st.info("⚡ **Energy calibration active**")

    with zipfile.ZipFile(plot_zip, "w", zipfile.ZIP_DEFLATED) as html_buffer:
        img_buffer = zipfile.ZipFile(image_zip, "w", zipfile.ZIP_DEFLATED) if KALEIDO_AVAILABLE else None

        try:
            for idx, file in enumerate(uploaded_files):
                filename = file.name
                status.info(f"Processing: {filename} ({idx + 1}/{len(uploaded_files)})")

                try:
                    # Parse ASC
                    content = file.read().decode(errors='ignore')
                    wl, counts_raw = parse_asc_file(content, skip_rows)

                    # Metadata
                    qs = extract_qs(filename)
                    nd_value = extract_nd(filename) if apply_nd else 0.0
                    thickness = extract_thickness(filename)
                    concentration = extract_concentration(filename)
                    dye_amount = extract_dye_amount(filename)
                    repetitions = extract_repetitions(filename)
                    sample_label = get_sample_label(thickness, concentration, dye_amount)
                    sample_label_short = get_short_label(thickness, concentration)

                    # Energy from calibration
                    if energy_map and not np.isnan(qs):
                        energy_mean, energy_std = interpolate_energy(qs, energy_map)
                    else:
                        energy_mean, energy_std = np.nan, np.nan

                    # OD correction
                    if nd_value > 0:
                        counts_corrected = apply_nd_correction(counts_raw, nd_value)
                        if show_individual:
                            st.info(f"🔧 OD={nd_value} correction (×{10**nd_value:.0f})")
                    else:
                        counts_corrected = counts_raw.copy()

                    # Analyze spectrum
                    result = analyze_spectrum(wl, counts_corrected)

                    # Individual plot
                    if show_individual:
                        fig = create_spectrum_plot(
                            wl,
                            counts_raw,
                            counts_corrected,
                            result,
                            filename,
                            nd_value,
                            energy_mean,
                            energy_std,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        if KALEIDO_AVAILABLE:
                            c1, c2 = st.columns([3, 1])
                            with c2:
                                try:
                                    img_bytes = fig_to_image(
                                        fig, image_format, image_width, image_height, image_scale
                                    )
                                    st.download_button(
                                        f"📥 {image_format.upper()}",
                                        img_bytes,
                                        f"{filename.replace('.asc', '')}.{image_format}",
                                        f"image/{image_format}",
                                        key=f"dl_{idx}",
                                    )
                                    if img_buffer:
                                        img_buffer.writestr(
                                            f"{filename.replace('.asc', '')}.{image_format}", img_bytes
                                        )
                                except Exception as e:
                                    st.error(f"Export failed: {e}")

                        if show_fit_params and result.fit_success:
                            with st.expander(f"🔍 Parameters - {filename}"):
                                c1, c2, c3, c4, c5, c6 = st.columns(6)
                                c1.metric("Amplitude", f"{result.fit_params.get('Amplitude', 0):.1f}")
                                c2.metric("Center", f"{result.fit_params.get('Center', 0):.2f}")
                                c3.metric("Gamma", f"{result.fit_params.get('Gamma', 0):.2f}")
                                c4.metric("OD", f"{nd_value:.1f}" if nd_value > 0 else "None")
                                if not np.isnan(energy_mean):
                                    c5.metric("Energy (mJ)", f"{energy_mean:.3f}")
                                c6.metric("Sample", sample_label_short)

                        html = fig.to_html(full_html=False, include_plotlyjs='cdn').encode()
                        html_buffer.writestr(f"{filename.replace('.asc', '')}.html", html)

                    # Combined plot label
                    label = f"QS={qs:.0f}" if not np.isnan(qs) else filename
                    if not np.isnan(energy_mean):
                        label += f" ({energy_mean:.2f}mJ)"
                    if nd_value > 0:
                        label += f" [OD{nd_value}]"

                    combined_fig.add_trace(
                        go.Scatter(x=wl, y=counts_corrected, mode='lines', name=label)
                    )

                    # Summary row
                    summary_data.append(
                        {
                            "File": filename,
                            "Thickness (mm)": thickness,
                            "UL Concentration (%)": concentration.get('upper')
                            if concentration
                            else None,
                            "LL Concentration (%)": concentration.get('lower')
                            if concentration
                            else None,
                            "Dye Amount (mg)": dye_amount,
                            "Repetitions": repetitions,
                            "Sample Label": sample_label,
                            "Sample Label Short": sample_label_short,
                            "QS Level": qs,
                            "Pump Energy (mJ)": energy_mean,
                            "Energy Std (mJ)": energy_std,
                            "OD Filter": nd_value,
                            "Correction Factor": 10**nd_value if nd_value > 0 else 1,
                            "Peak λ (nm)": result.peak_wavelength,
                            "Peak Intensity": result.peak_intensity,
                            "FWHM (nm)": result.fwhm,
                            "Integrated Intensity": result.integrated_intensity,
                            "R²": result.r_squared,
                            "SNR": result.snr,
                            "Fit Success": "✅" if result.fit_success else "❌",
                        }
                    )

                except Exception as e:
                    st.error(f"Error: {filename}: {e}")
                    continue

                progress_bar.progress((idx + 1) / len(uploaded_files))

        finally:
            if img_buffer:
                img_buffer.close()

    status.success("✅ Processing complete!")
    progress_bar.empty()

    # ==============================================================
    # RESULTS
    # ==============================================================

    st.markdown("---")
    st.subheader("📊 Summary Statistics")

    summary_df = pd.DataFrame(summary_data)
    if (
        'Pump Energy (mJ)' in summary_df.columns
        and summary_df['Pump Energy (mJ)'].notna().any()
    ):
        summary_df = summary_df.sort_values("Pump Energy (mJ)")
    else:
        summary_df = summary_df.sort_values("QS Level")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Files", len(summary_df))
    c2.metric("Avg R²", f"{summary_df['R²'].mean():.3f}")
    c3.metric("Avg FWHM", f"{summary_df['FWHM (nm)'].mean():.2f} nm")
    c4.metric("OD Corrected", summary_df[summary_df["OD Filter"] > 0].shape[0])
    c5.metric("Energy Cal.", summary_df[summary_df["Pump Energy (mJ)"].notna()].shape[0])
    c6.metric("Conditions", summary_df['Sample Label Short'].nunique())

    # Combined plot
    st.markdown("---")
    st.subheader("🌈 Combined Spectra")
    combined_fig.update_layout(
        title="Spectral Evolution",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity (OD-corrected)",
        template="plotly_white",
        height=600,
    )
    st.plotly_chart(combined_fig, use_container_width=True)

    if KALEIDO_AVAILABLE:
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            try:
                img = fig_to_image(combined_fig, image_format, image_width, image_height, image_scale)
                st.download_button(
                    f"📥 Combined ({image_format.upper()})",
                    img,
                    f"combined.{image_format}",
                    f"image/{image_format}",
                )
            except Exception:
                pass

    # Results table
    st.markdown("---")
    st.subheader("📋 Results Table")

    def highlight_r2(val):
        if pd.isna(val):
            return ''
        if val > 0.95:
            return 'background-color: #d4edda'
        if val > 0.85:
            return 'background-color: #fff3cd'
        return 'background-color: #f8d7da'

    styled = summary_df.style.applymap(highlight_r2, subset=['R²']).format(
        {
            'Peak λ (nm)': '{:.2f}',
            'Peak Intensity': '{:.0f}',
            'FWHM (nm)': '{:.2f}',
            'Integrated Intensity': '{:.2e}',
            'R²': '{:.4f}',
            'SNR': '{:.1f}',
            'QS Level': lambda x: f'{x:.0f}' if not pd.isna(x) else '',
            'Pump Energy (mJ)': lambda x: f'{x:.4f}' if not pd.isna(x) else '',
            'Energy Std (mJ)': lambda x: f'{x:.4f}' if not pd.isna(x) else '',
            'OD Filter': lambda x: f'{x:.1f}' if x > 0 else '',
            'Thickness (mm)': lambda x: f'{x:.1f}' if not pd.isna(x) else '',
            'UL Concentration (%)': lambda x: f'{x:.1f}' if not pd.isna(x) else '',
            'LL Concentration (%)': lambda x: f'{x:.1f}' if not pd.isna(x) else '',
            'Dye Amount (mg)': lambda x: f'{x:.1f}' if not pd.isna(x) else '',
            'Repetitions': lambda x: f'{x:.0f}' if not pd.isna(x) else '',
        }
    )

    st.dataframe(styled, use_container_width=True)

    # Threshold detection
    use_energy_axis = (
        'Pump Energy (mJ)' in summary_df.columns
        and summary_df['Pump Energy (mJ)'].notna().sum() > 3
    )

    if use_energy_axis or summary_df['QS Level'].notna().sum() > 3:
        st.markdown("---")
        st.subheader("🎯 Threshold Detection")

        if use_energy_axis:
            valid = summary_df.dropna(subset=['Pump Energy (mJ)', 'Integrated Intensity'])
            threshold = detect_threshold(
                valid['Pump Energy (mJ)'].values,
                valid['Integrated Intensity'].values,
            )
        else:
            valid = summary_df.dropna(subset=['QS Level', 'Integrated Intensity'])
            threshold = detect_threshold(
                valid['QS Level'].values,
                valid['Integrated Intensity'].values,
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            if threshold.threshold_found:
                if use_energy_axis:
                    st.success(f"✅ Threshold: **{threshold.threshold_energy:.4f} mJ**")
                else:
                    st.success(f"✅ Threshold: QS **{threshold.threshold_qs:.1f}**")
            else:
                st.warning("⚠️ No threshold detected")
        with c2:
            st.metric("Slope (below)", f"{threshold.slope_below:.2e}")
        with c3:
            st.metric("Slope (above)", f"{threshold.slope_above:.2e}")

        threshold_fig = create_threshold_plot(summary_df, threshold, use_energy_axis)
        st.plotly_chart(threshold_fig, use_container_width=True)

        if KALEIDO_AVAILABLE:
            c1, c2, c3 = st.columns([2, 1, 2])
            with c2:
                try:
                    img = fig_to_image(
                        threshold_fig,
                        image_format,
                        int(image_width * 1.5),
                        int(image_height * 1.2),
                        image_scale,
                    )
                    st.download_button(
                        f"📥 Threshold ({image_format.upper()})",
                        img,
                        f"threshold.{image_format}",
                        f"image/{image_format}",
                    )
                except Exception:
                    pass

    # Energy vs wavelength
    if (
        'Pump Energy (mJ)' in summary_df.columns
        and summary_df['Pump Energy (mJ)'].notna().sum() > 2
    ):
        st.markdown("---")
        st.subheader("📈 Peak Wavelength Evolution")

        if 'Sample Label Short' in summary_df.columns:
            unique_conditions = [c for c in summary_df['Sample Label Short'].unique() if c != "No Label"]

            if len(unique_conditions) > 0:
                st.info(f"📊 **Sample Conditions**: {len(unique_conditions)} detected")

                with st.expander("🔍 Sample Conditions Details"):
                    conditions_summary = (
                        summary_df[summary_df['Sample Label Short'] != "No Label"]
                        .groupby('Sample Label Short')
                        .agg(
                            {
                                'File': 'count',
                                'Thickness (mm)': 'first',
                                'UL Concentration (%)': 'first',
                                'LL Concentration (%)': 'first',
                                'Dye Amount (mg)': 'first',
                                'Pump Energy (mJ)': ['min', 'max'],
                                'Peak λ (nm)': ['min', 'max'],
                            }
                        )
                        .round(4)
                    )
                    conditions_summary.columns = [
                        'Files',
                        'Thickness',
                        'UL%',
                        'LL%',
                        'Dye(mg)',
                        'E min',
                        'E max',
                        'λ min',
                        'λ max',
                    ]
                    st.dataframe(conditions_summary)

        energy_wl_fig = create_energy_wavelength_plot(summary_df)

        if energy_wl_fig:
            st.plotly_chart(energy_wl_fig, use_container_width=True)

            if KALEIDO_AVAILABLE:
                c1, c2, c3 = st.columns([2, 1, 2])
                with c2:
                    try:
                        img = fig_to_image(energy_wl_fig, image_format, image_width, image_height, image_scale)
                        st.download_button(
                            f"📥 Energy-λ ({image_format.upper()})",
                            img,
                            f"energy_wavelength.{image_format}",
                            f"image/{image_format}",
                            key="dl_e_wl",
                        )
                    except Exception:
                        pass

            with st.expander("📊 Wavelength Shift Statistics"):
                if 'Sample Label Short' in summary_df.columns:
                    for label in summary_df['Sample Label Short'].unique():
                        if label == "No Label":
                            continue

                        group = summary_df[
                            summary_df['Sample Label Short'] == label
                        ].dropna(subset=['Pump Energy (mJ)', 'Peak λ (nm)'])
                        if len(group) > 1:
                            min_wl = group['Peak λ (nm)'].min()
                            max_wl = group['Peak λ (nm)'].max()
                            shift = max_wl - min_wl

                            st.markdown(f"**{label}**")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("λ Range", f"{min_wl:.2f}-{max_wl:.2f} nm")
                            c2.metric("Shift", f"{shift:.2f} nm")
                            c3.metric("Points", len(group))

    # Energy vs intensity
    if (
        'Pump Energy (mJ)' in summary_df.columns
        and summary_df['Pump Energy (mJ)'].notna().sum() > 2
    ):
        st.markdown("---")
        st.subheader("💡 Peak Intensity Evolution")

        energy_int_fig = create_energy_intensity_plot(summary_df)

        if energy_int_fig:
            st.plotly_chart(energy_int_fig, use_container_width=True)

            if KALEIDO_AVAILABLE:
                c1, c2, c3 = st.columns([2, 1, 2])
                with c2:
                    try:
                        img = fig_to_image(energy_int_fig, image_format, image_width, image_height, image_scale)
                        st.download_button(
                            f"📥 Energy-Intensity ({image_format.upper()})",
                            img,
                            f"energy_intensity.{image_format}",
                            f"image/{image_format}",
                            key="dl_e_int",
                        )
                    except Exception:
                        pass

            with st.expander("📊 Intensity Growth Statistics"):
                if 'Sample Label Short' in summary_df.columns:
                    for label in summary_df['Sample Label Short'].unique():
                        if label == "No Label":
                            continue

                        group = summary_df[
                            summary_df['Sample Label Short'] == label
                        ].dropna(subset=['Pump Energy (mJ)', 'Peak Intensity'])
                        if len(group) > 1:
                            min_int = group['Peak Intensity'].min()
                            max_int = group['Peak Intensity'].max()
                            growth = max_int / min_int if min_int > 0 else 0

                            st.markdown(f"**{label}**")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Range", f"{min_int:.0f}-{max_int:.0f}")
                            c2.metric("Growth", f"{growth:.1f}×")
                            c3.metric("Points", len(group))

    # Downloads
    st.markdown("---")
    st.subheader("💾 Export All Results")

    cols = st.columns(4 if KALEIDO_AVAILABLE else 3)

    with cols[0]:
        csv = summary_df.to_csv(index=False).encode()
        st.download_button(
            "📥 CSV",
            csv,
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True,
        )

    with cols[1]:
        st.download_button(
            "📦 HTML",
            plot_zip.getvalue(),
            f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            "application/zip",
            use_container_width=True,
        )

    with cols[2]:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Results', index=False)
            if energy_map:
                energy_cal_df = pd.DataFrame(
                    [
                        {
                            'QS': qs,
                            'Mean (mJ)': d['mean'],
                            'Std (mJ)': d['std'],
                            'N': d['n_readings'],
                            'OD': d.get('od', 0.0),
                        }
                        for qs, d in energy_map.items()
                    ]
                ).sort_values('QS')
                energy_cal_df.to_excel(writer, sheet_name='Energy Cal', index=False)

        st.download_button(
            "📊 Excel",
            excel_buffer.getvalue(),
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.ms-excel",
            use_container_width=True,
        )

    if KALEIDO_AVAILABLE and image_zip:
        with cols[3]:
            st.download_button(
                "🖼️ Images",
                image_zip.getvalue(),
                f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                "application/zip",
                use_container_width=True,
            )

else:
    st.info("👆 Upload .asc files to begin")

    with st.expander("📖 Instructions"):
        st.markdown(
            """
        ### Quick Start
        1. Upload .asc spectrum files
        2. Upload energy calibration file (optional)
        3. View automated analysis
        4. Download results

        ### Filename Format Example
        `UL_5mm_QS_110_10rep_17mgR6G_UL_5%IL_LL_1%IL_OD=2.asc`

        **Auto-extracts:**
        - Thickness: `UL_5mm` → 5mm
        - QS: `QS_110` → 110
        - UL Conc: `UL_5%IL` → 5%
        - LL Conc: `LL_1%IL` → 1%
        - Dye: `17mgR6G` → 17mg
        - OD: `OD=2` → 2 (×100)

        ### Energy File
        Your uploaded Excel like `UL_5mm_pulse energy.xlsx` is supported:
        - Finds the QS row automatically (e.g. 200, 190, 180, ...)
        - Reads ~10 rows of energies below, auto-converts J → mJ
        - Uses optional OD row if present
        """
        )

st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
📧 varun.solanki@fau.de | Friedrich-Alexander-Universität Erlangen-Nürnberg
</div>
""",
    unsafe_allow_html=True,
)

# ==============================================================
# Streamlit App: Random Laser ASC Analyzer
# Complete Version with ND Correction & Energy Calibration
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
except:
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
    """
    Perform Lorentzian fitting and extract spectral parameters
    """
    try:
        # Basic statistics
        peak_idx = np.argmax(counts)
        peak_val = counts[peak_idx]
        x0_init = wl[peak_idx]
        
        baseline_est = np.percentile(counts, 5)
        A_init = np.max(counts) - baseline_est
        y0_init = baseline_est
        
        # Estimate gamma
        half_max = baseline_est + A_init / 2
        above_half = counts > half_max
        
        if np.sum(above_half) > 2:
            indices = np.where(above_half)[0]
            width = wl[indices[-1]] - wl[indices[0]]
            gamma_init = max(width / 2, 0.1)
        else:
            gamma_init = (wl.max() - wl.min()) / 10
        
        bounds = (
            [0, wl.min(), 0, -np.inf],
            [np.inf, wl.max(), np.inf, np.inf]
        )
        
        p0 = [A_init, x0_init, gamma_init, y0_init]
        
        # Perform fit
        try:
            popt, pcov = curve_fit(lorentzian, wl, counts, p0=p0, maxfev=50000, method='lm')
        except:
            popt, pcov = curve_fit(lorentzian, wl, counts, p0=p0, bounds=bounds, maxfev=50000, method='trf')
        
        A, x0, gamma, y0 = popt
        A = abs(A)
        gamma = abs(gamma)
        
        fwhm = 2 * gamma
        fit_y = lorentzian(wl, A, x0, gamma, y0)
        
        # Calculate metrics
        baseline_corrected = counts - y0
        area = np.trapz(np.maximum(baseline_corrected, 0), wl)
        r_squared = calculate_r_squared(counts, fit_y)
        snr = calculate_snr(counts)
        
        if r_squared < 0.3:
            raise ValueError(f"Poor fit quality: R² = {r_squared:.3f}")
        
        fit_params = {
            'Amplitude': float(A),
            'Center': float(x0),
            'Gamma': float(gamma),
            'Baseline': float(y0),
            'Std_Errors': [float(x) for x in np.sqrt(np.diag(pcov))]
        }
        
        return FitResult(
            float(x0), float(A + y0), float(fwhm), float(area), 
            fit_y, float(r_squared), float(snr), fit_params, fit_success=True
        )
        
    except Exception as e:
        # Fallback
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
            float(peak_wl), float(peak_int),
            float(fwhm_estimate) if not np.isnan(fwhm_estimate) else np.nan,
            float(np.trapz(counts - np.min(counts), wl)),
            counts.copy(), 0.0, float(calculate_snr(counts)),
            {'error': str(e)}, fit_success=False
        )

# ==============================================================
# ND FILTER CORRECTION FUNCTIONS
# ==============================================================
def extract_nd(filename: str) -> float:
    """Extract ND filter value from filename"""
    match = re.search(r'ND[=_\s-]*(\d+\.?\d*)', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0

def apply_nd_correction(counts: np.ndarray, nd_value: float) -> np.ndarray:
    """Apply ND filter correction: multiply by 10^ND"""
    if nd_value == 0:
        return counts
    correction_factor = 10 ** nd_value
    return counts * correction_factor

# ==============================================================
# ENERGY CALIBRATION FUNCTIONS
# ==============================================================
@st.cache_data
def parse_energy_file(file_content: str, file_type: str, file_bytes: bytes = None) -> Dict[float, Dict]:
    """
    Parse energy calibration file with TRANSPOSED format
    Expected format:
    - Row 1: QS_Level | 110 | 120 | 130 | ...
    - Rows 2-11: Energy 1-10 with values for each QS
    """
    energy_map = {}
    
    try:
        # Read file based on type
        if file_bytes and (file_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or \
           file_type == 'application/vnd.ms-excel' or \
           'xlsx' in str(file_type).lower() or 'xls' in str(file_type).lower()):
            import io
            df = pd.read_excel(io.BytesIO(file_bytes), header=None)
        else:
            # Try CSV/TSV
            for sep in ['\t', ',', ';', '|']:
                try:
                    df = pd.read_csv(StringIO(file_content), sep=sep, header=None)
                    break
                except:
                    continue
        
        # Check for transposed format
        first_row = df.iloc[0, :]
        
        # Detect if first cell is QS_Level label
        if str(first_row.iloc[0]).lower().replace('_', '').replace(' ', '') in ['qslevel', 'qs', 'qswitch']:
            # TRANSPOSED FORMAT
            
            # Extract QS levels from first row
            qs_levels = []
            for val in first_row.iloc[1:]:
                try:
                    qs_levels.append(float(val))
                except:
                    continue
            
            # Initialize storage
            for qs in qs_levels:
                energy_map[qs] = {'readings': []}
            
            # Read energy values from rows 2-11
            for row_idx in range(1, len(df)):
                row = df.iloc[row_idx, :]
                row_label = str(row.iloc[0]).lower()
                
                # Skip empty or non-energy rows
                if not any(x in row_label for x in ['energy', 'e', 'reading', 'measurement']):
                    continue
                
                # Read energy values
                for col_idx, qs in enumerate(qs_levels):
                    try:
                        energy_val = float(row.iloc[col_idx + 1])
                        # Convert J to mJ if needed
                        if energy_val < 0.1:
                            energy_val = energy_val * 1000
                        energy_map[qs]['readings'].append(energy_val)
                    except:
                        continue
            
            # Calculate statistics
            final_map = {}
            for qs in energy_map:
                readings = energy_map[qs]['readings']
                if readings:
                    final_map[qs] = {
                        'mean': np.mean(readings),
                        'std': np.std(readings),
                        'readings': readings,
                        'n_readings': len(readings)
                    }
            
            return final_map
        
        else:
            # STANDARD FORMAT (QS in first column)
            for idx, row in df.iterrows():
                try:
                    qs_level = float(row.iloc[0])
                    energy_readings = []
                    for val in row.iloc[1:]:
                        try:
                            if pd.notna(val) and str(val).strip():
                                energy_val = float(val)
                                if energy_val < 0.1:
                                    energy_val = energy_val * 1000
                                energy_readings.append(energy_val)
                        except:
                            continue
                    
                    if energy_readings:
                        energy_map[qs_level] = {
                            'mean': np.mean(energy_readings),
                            'std': np.std(energy_readings),
                            'readings': energy_readings,
                            'n_readings': len(energy_readings)
                        }
                except:
                    continue
            
            return energy_map
        
    except Exception as e:
        st.error(f"Error parsing energy file: {str(e)}")
        return {}

def interpolate_energy(qs_value: float, energy_map: Dict[float, Dict]) -> Tuple[float, float]:
    """Get energy for QS value with interpolation"""
    if not energy_map or np.isnan(qs_value):
        return np.nan, np.nan
    
    # Exact match
    if qs_value in energy_map:
        return energy_map[qs_value]['mean'], energy_map[qs_value]['std']
    
    # Interpolation
    qs_values = sorted(energy_map.keys())
    
    if qs_value < qs_values[0]:
        return energy_map[qs_values[0]]['mean'], energy_map[qs_values[0]]['std']
    if qs_value > qs_values[-1]:
        return energy_map[qs_values[-1]]['mean'], energy_map[qs_values[-1]]['std']
    
    energy_means = [energy_map[qs]['mean'] for qs in qs_values]
    energy_stds = [energy_map[qs]['std'] for qs in qs_values]
    
    interp_mean = np.interp(qs_value, qs_values, energy_means)
    interp_std = np.interp(qs_value, qs_values, energy_stds)
    
    return interp_mean, interp_std

# ==============================================================
# THRESHOLD DETECTION
# ==============================================================
def detect_threshold(x_values: np.ndarray, intensities: np.ndarray, 
                     min_points: int = 3) -> ThresholdAnalysis:
    """Detect lasing threshold using broken-stick algorithm"""
    if len(x_values) < 2 * min_points:
        return ThresholdAnalysis(None, None, 0, 0, False)
    
    try:
        idx = np.argsort(x_values)
        x_sorted = x_values[idx]
        int_sorted = intensities[idx]
        
        best_threshold = None
        best_r2_sum = -np.inf
        best_slopes = (0, 0)
        
        for i in range(min_points, len(x_sorted) - min_points):
            below = np.polyfit(x_sorted[:i], int_sorted[:i], 1)
            above = np.polyfit(x_sorted[i:], int_sorted[i:], 1)
            
            r2_below = calculate_r_squared(int_sorted[:i], np.polyval(below, x_sorted[:i]))
            r2_above = calculate_r_squared(int_sorted[i:], np.polyval(above, x_sorted[i:]))
            
            r2_sum = r2_below + r2_above
            if r2_sum > best_r2_sum and above[0] > below[0]:
                best_r2_sum = r2_sum
                best_threshold = x_sorted[i]
                best_slopes = (below[0], above[0])
        
        found = best_threshold is not None and best_slopes[1] > 2 * best_slopes[0]
        
        return ThresholdAnalysis(None, best_threshold, best_slopes[0], best_slopes[1], found)
        
    except:
        return ThresholdAnalysis(None, None, 0, 0, False)

# ==============================================================
# FILE PARSING
# ==============================================================
def extract_qs(filename: str) -> float:
    """Extract Q-switch value from filename"""
    patterns = [
        r'qs[_\s-]*(\d+\.?\d*)',
        r'(\d+\.?\d*)[_\s-]*qs',
        r'q[_\s-]*(\d+\.?\d*)',
        r'(\d+\.?\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    return np.nan

@st.cache_data
def parse_asc_file(file_content: str, skip_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse .asc file"""
    df = pd.read_csv(
        StringIO(file_content),
        sep='\t',
        decimal=',',
        skiprows=skip_rows,
        engine='python'
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
def create_spectrum_plot(wl: np.ndarray, counts_raw: np.ndarray, counts_corrected: np.ndarray,
                        fit_result: FitResult, filename: str, nd_value: float, 
                        energy_mean: float = None, energy_std: float = None) -> go.Figure:
    """Create spectrum plot with fit overlay"""
    fig = go.Figure()
    
    # Raw data if ND correction applied
    if nd_value > 0:
        fig.add_trace(go.Scatter(
            x=wl, y=counts_raw,
            mode='lines',
            name='Raw Data',
            line=dict(color='lightgray', width=2),
            opacity=0.5
        ))
    
    # Corrected data
    fig.add_trace(go.Scatter(
        x=wl, y=counts_corrected,
        mode='lines',
        name='ND-Corrected Data' if nd_value > 0 else 'Data',
        line=dict(color='#2E86AB', width=3)
    ))
    
    # Fit
    if fit_result.fit_success and not np.isnan(fit_result.fwhm):
        fig.add_trace(go.Scatter(
            x=wl, y=fit_result.fit_y,
            mode='lines',
            name='Lorentzian Fit',
            line=dict(color='red', width=3, dash='dash'),
            opacity=0.8
        ))
        
        # Peak marker
        fig.add_vline(
            x=fit_result.peak_wavelength,
            line_dash="dot",
            line_color="green",
            line_width=2,
            annotation_text=f"Peak: {fit_result.peak_wavelength:.2f} nm"
        )
        
        # FWHM
        gamma = fit_result.fwhm / 2
        x0 = fit_result.peak_wavelength
        half_max = fit_result.peak_intensity / 2
        
        fig.add_trace(go.Scatter(
            x=[x0-gamma, x0+gamma],
            y=[half_max, half_max],
            mode='markers+text',
            marker=dict(size=12, color='orange', symbol='diamond'),
            name=f'FWHM = {fit_result.fwhm:.2f} nm',
            text=['', f'FWHM={fit_result.fwhm:.2f}nm'],
            textposition='top center'
        ))
        
        fig.add_shape(
            type="line",
            x0=x0-gamma, y0=half_max,
            x1=x0+gamma, y1=half_max,
            line=dict(color="orange", width=2, dash="dash")
        )
    
    # Title
    title_html = f"<b>{filename}</b><br>"
    if energy_mean is not None and not np.isnan(energy_mean):
        title_html += f"<sub>Pump Energy: {energy_mean:.3f}±{energy_std:.3f} mJ</sub><br>"
    if nd_value > 0:
        title_html += f"<sub>ND: {nd_value} (×{10**nd_value:.0f})</sub><br>"
    if fit_result.fit_success:
        title_html += f"<sub>Peak: {fit_result.peak_wavelength:.2f} nm | "
        title_html += f"FWHM: {fit_result.fwhm:.2f} nm | R²: {fit_result.r_squared:.4f}</sub>"
    
    fig.update_layout(
        title=title_html,
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity (counts)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        showlegend=True
    )
    
    return fig

def create_threshold_plot(df: pd.DataFrame, threshold: ThresholdAnalysis, use_energy: bool = True) -> go.Figure:
    """Create threshold analysis with UNIFORM Y-AXIS SCALING and SMOOTH CURVES"""
    from scipy.interpolate import make_interp_spline
    
    x_col = 'Pump Energy (mJ)' if use_energy and 'Pump Energy (mJ)' in df.columns else 'QS Level'
    x_label = "Pump Energy (mJ)" if use_energy else "Q-Switch Level"
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Integrated Intensity vs {x_label}",
            f"FWHM vs {x_label}",
            f"Peak Wavelength vs {x_label}",
            f"Peak Intensity vs {x_label}"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    valid = df.dropna(subset=[x_col])
    x_values = valid[x_col].values
    
    # Sort by x values
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    
    # Calculate uniform y-ranges
    int_min = valid['Integrated Intensity'].min()
    int_max = valid['Integrated Intensity'].max()
    int_padding = (int_max - int_min) * 0.1
    int_range = [int_min - int_padding, int_max + int_padding]
    
    peak_min = valid['Peak Intensity'].min()
    peak_max = valid['Peak Intensity'].max()
    peak_padding = (peak_max - peak_min) * 0.1
    peak_range = [peak_min - peak_padding, peak_max + peak_padding]
    
    fwhm_min = valid['FWHM (nm)'].min()
    fwhm_max = valid['FWHM (nm)'].max()
    fwhm_padding = (fwhm_max - fwhm_min) * 0.1
    fwhm_range = [max(0, fwhm_min - fwhm_padding), fwhm_max + fwhm_padding]
    
    wl_min = valid['Peak λ (nm)'].min()
    wl_max = valid['Peak λ (nm)'].max()
    wl_padding = max((wl_max - wl_min) * 0.1, 2)
    wl_range = [wl_min - wl_padding, wl_max + wl_padding]
    
    # Helper function to create smooth curve
    def create_smooth_curve(x, y, num_points=300):
        """Create smooth curve using spline interpolation"""
        if len(x) < 4:
            # Not enough points for spline, return original
            return x, y
        
        try:
            # Create spline interpolation
            spline = make_interp_spline(x, y, k=min(3, len(x)-1))
            x_smooth = np.linspace(x.min(), x.max(), num_points)
            y_smooth = spline(x_smooth)
            return x_smooth, y_smooth
        except:
            # Fallback to original if spline fails
            return x, y
    
    # ============================================================
    # Plot 1: Integrated Intensity
    # ============================================================
    y_int = valid['Integrated Intensity'].values[sort_idx]
    
    # Smooth curve
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_int)
    fig.add_trace(
        go.Scatter(
            x=x_smooth, 
            y=y_smooth,
            mode='lines',
            line=dict(width=3, color='red', shape='spline'),
            name='Integrated Intensity',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Original data points as markers
    fig.add_trace(
        go.Scatter(
            x=x_sorted, 
            y=y_int,
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle', line=dict(width=2, color='white')),
            error_x=dict(
                type='data',
                array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                visible=True if 'Energy Std (mJ)' in valid.columns else False,
                thickness=1.5,
                width=4
            ),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Threshold line
    if threshold.threshold_found and threshold.threshold_energy:
        fig.add_vline(
            x=threshold.threshold_energy,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Threshold: {threshold.threshold_energy:.4f} mJ",
            annotation_position="top",
            row=1, col=1
        )
    
    # ============================================================
    # Plot 2: FWHM
    # ============================================================
    y_fwhm = valid['FWHM (nm)'].values[sort_idx]
    
    # Smooth curve
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_fwhm)
    fig.add_trace(
        go.Scatter(
            x=x_smooth, 
            y=y_smooth,
            mode='lines',
            line=dict(width=3, color='blue', shape='spline'),
            name='FWHM',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Data points
    fig.add_trace(
        go.Scatter(
            x=x_sorted, 
            y=y_fwhm,
            mode='markers',
            marker=dict(size=10, color='blue', symbol='circle', line=dict(width=2, color='white')),
            error_x=dict(
                type='data',
                array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                visible=True if 'Energy Std (mJ)' in valid.columns else False,
                thickness=1.5,
                width=4
            ),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # ============================================================
    # Plot 3: Peak Wavelength
    # ============================================================
    y_wl = valid['Peak λ (nm)'].values[sort_idx]
    
    # Smooth curve
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_wl)
    fig.add_trace(
        go.Scatter(
            x=x_smooth, 
            y=y_smooth,
            mode='lines',
            line=dict(width=3, color='purple', shape='spline'),
            name='Peak Wavelength',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Data points
    fig.add_trace(
        go.Scatter(
            x=x_sorted, 
            y=y_wl,
            mode='markers',
            marker=dict(size=10, color='purple', symbol='circle', line=dict(width=2, color='white')),
            error_x=dict(
                type='data',
                array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                visible=True if 'Energy Std (mJ)' in valid.columns else False,
                thickness=1.5,
                width=4
            ),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # ============================================================
    # Plot 4: Peak Intensity
    # ============================================================
    y_peak = valid['Peak Intensity'].values[sort_idx]
    
    # Smooth curve
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_peak)
    fig.add_trace(
        go.Scatter(
            x=x_smooth, 
            y=y_smooth,
            mode='lines',
            line=dict(width=3, color='orange', shape='spline'),
            name='Peak Intensity',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Data points
    fig.add_trace(
        go.Scatter(
            x=x_sorted, 
            y=y_peak,
            mode='markers',
            marker=dict(size=10, color='orange', symbol='circle', line=dict(width=2, color='white')),
            error_x=dict(
                type='data',
                array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                visible=True if 'Energy Std (mJ)' in valid.columns else False,
                thickness=1.5,
                width=4
            ),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # ============================================================
    # Update axes with uniform ranges
    # ============================================================
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text=x_label, row=row, col=col)
    
    fig.update_yaxes(title_text="Integrated Intensity", range=int_range, row=1, col=1)
    fig.update_yaxes(title_text="FWHM (nm)", range=fwhm_range, row=1, col=2)
    fig.update_yaxes(title_text="Wavelength (nm)", range=wl_range, row=2, col=1)
    fig.update_yaxes(title_text="Counts", range=peak_range, row=2, col=2)
    
    # Optional: log scale for large ranges
    if int_max > 0 and int_min > 0 and (int_max / int_min) > 1000:
        fig.update_yaxes(type="log", row=1, col=1)
    if peak_max > 0 and peak_min > 0 and (peak_max / peak_min) > 1000:
        fig.update_yaxes(type="log", row=2, col=2)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        template="plotly_white",
        title_text="<b>Threshold Analysis Dashboard</b>"
    )
    
    return fig
# ==============================================================
# IMAGE EXPORT
# ==============================================================
def fig_to_image(fig: go.Figure, format: str, width: int, height: int, scale: int) -> bytes:
    """Convert figure to image"""
    try:
        return fig.to_image(format=format, width=width, height=height, scale=scale, engine="kaleido")
    except Exception as e:
        st.error(f"Image export failed: {str(e)}")
        raise

# ==============================================================
# STREAMLIT APP
# ==============================================================
st.set_page_config(page_title="Random Laser Analyzer", layout="wide", page_icon="🔬")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .energy-box {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #0066cc;
        margin: 10px 0;
    }
    .nd-box {
        background-color: #ffe4b5;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff8c00;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔬 Random Laser Analyzer</p>', unsafe_allow_html=True)
st.markdown("**Lorentzian fitting • ND correction • Energy calibration • Threshold detection**")

if not KALEIDO_AVAILABLE:
    st.warning("⚠️ Image export disabled. Install kaleido: `pip install kaleido`")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("📁 File Settings", expanded=True):
        skip_rows = st.number_input("Header rows to skip", 0, 100, 38)
        show_individual = st.checkbox("Show individual plots", True)
        show_fit_params = st.checkbox("Show fit parameters", False)
        apply_nd = st.checkbox("Apply ND correction", True)
    
    if KALEIDO_AVAILABLE:
        with st.expander("💾 Export Settings"):
            image_format = st.selectbox("Format", ["png", "jpeg", "svg", "pdf"])
            image_width = st.number_input("Width (px)", 800, 3000, 1200)
            image_height = st.number_input("Height (px)", 400, 2000, 800)
            image_scale = st.slider("Scale", 1, 5, 2)
    
    st.markdown("---")
    st.markdown("### 📊 Features")
    st.markdown("""
    - ✅ Lorentzian fitting
    - ✅ ND filter correction
    - ✅ Energy calibration
    - ✅ Uniform y-axis scaling
    - ✅ Threshold detection
    - ✅ Interactive plots
    """)

# File Upload Section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Spectrum Files (.asc)")
    uploaded_files = st.file_uploader(
        "Upload spectrum files",
        accept_multiple_files=True,
        type=['asc'],
        key="spectrum_files"
    )

with col2:
    st.subheader("⚡ Energy Calibration")
    
    with st.expander("📋 Format Guide"):
        st.markdown("""
        **Excel/CSV with QS in columns:**
        ```
        QS_Level  110      120      130
        Energy 1  8.2E-06  2.6E-05  6.7E-05
        Energy 2  9.0E-06  2.5E-05  6.3E-05
        ...
        ```
        Values in Joules auto-convert to mJ
        """)
    
    energy_file = st.file_uploader(
        "Upload energy file (optional)",
        type=['csv', 'txt', 'tsv', 'xlsx', 'xls'],
        key="energy_file"
    )
    
    if energy_file:
        with st.expander("📊 Preview", expanded=False):
            try:
                file_bytes = energy_file.read()
                energy_file.seek(0)
                
                if energy_file.name.endswith(('.xlsx', '.xls')):
                    energy_map = parse_energy_file(None, energy_file.type, file_bytes)
                else:
                    energy_content = file_bytes.decode(errors='ignore')
                    energy_map = parse_energy_file(energy_content, energy_file.type, file_bytes)
                
                if energy_map:
                    energy_df = pd.DataFrame([
                        {
                            'QS': qs,
                            'Mean (mJ)': data['mean'],
                            'Std (mJ)': data['std'],
                            'N': data['n_readings']
                        }
                        for qs, data in energy_map.items()
                    ]).sort_values('QS')
                    
                    st.dataframe(energy_df.style.format({
                        'QS': '{:.0f}',
                        'Mean (mJ)': '{:.4f}',
                        'Std (mJ)': '{:.4f}',
                        'N': '{:.0f}'
                    }))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=energy_df['QS'],
                        y=energy_df['Mean (mJ)'],
                        error_y=dict(type='data', array=energy_df['Std (mJ)'], visible=True),
                        mode='markers+lines',
                        marker=dict(size=10, color='blue')
                    ))
                    fig.update_layout(
                        title="Energy Calibration",
                        xaxis_title="QS Level",
                        yaxis_title="Energy (mJ)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not parse file")
            except Exception as e:
                st.error(f"Error: {str(e)}")
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
        st.markdown('<div class="energy-box"><b>⚡ Energy calibration active</b></div>', unsafe_allow_html=True)
    
    with zipfile.ZipFile(plot_zip, "w", zipfile.ZIP_DEFLATED) as html_buffer:
        img_buffer = zipfile.ZipFile(image_zip, "w", zipfile.ZIP_DEFLATED) if KALEIDO_AVAILABLE else None
        
        try:
            for idx, file in enumerate(uploaded_files):
                filename = file.name
                status.info(f"Processing: {filename} ({idx+1}/{len(uploaded_files)})")
                
                try:
                    # Parse
                    content = file.read().decode(errors='ignore')
                    wl, counts_raw = parse_asc_file(content, skip_rows)
                    
                    # Metadata
                    qs = extract_qs(filename)
                    nd_value = extract_nd(filename) if apply_nd else 0.0
                    
                    # Energy
                    if energy_map and not np.isnan(qs):
                        energy_mean, energy_std = interpolate_energy(qs, energy_map)
                    else:
                        energy_mean, energy_std = np.nan, np.nan
                    
                    # ND correction
                    if nd_value > 0:
                        counts_corrected = apply_nd_correction(counts_raw, nd_value)
                        if show_individual:
                            st.info(f"🔧 ND={nd_value} correction (×{10**nd_value:.0f})")
                    else:
                        counts_corrected = counts_raw.copy()
                    
                    # Analyze
                    result = analyze_spectrum(wl, counts_corrected)
                    
                    # Plot
                    if show_individual:
                        fig = create_spectrum_plot(wl, counts_raw, counts_corrected, result, 
                                                  filename, nd_value, energy_mean, energy_std)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if KALEIDO_AVAILABLE:
                            col1, col2 = st.columns([3, 1])
                            with col2:
                                try:
                                    img_bytes = fig_to_image(fig, image_format, image_width, image_height, image_scale)
                                    st.download_button(
                                        f"📥 {image_format.upper()}",
                                        img_bytes,
                                        f"{filename.replace('.asc', '')}.{image_format}",
                                        f"image/{image_format}",
                                        key=f"dl_{idx}"
                                    )
                                    if img_buffer:
                                        img_buffer.writestr(f"{filename.replace('.asc', '')}.{image_format}", img_bytes)
                                except Exception as e:
                                    st.error(f"Export failed: {e}")
                        
                        if show_fit_params and result.fit_success:
                            with st.expander(f"🔍 Parameters - {filename}"):
                                c1, c2, c3, c4, c5 = st.columns(5)
                                c1.metric("Amp", f"{result.fit_params.get('Amplitude', 0):.1f}")
                                c2.metric("Center", f"{result.fit_params.get('Center', 0):.2f}")
                                c3.metric("Gamma", f"{result.fit_params.get('Gamma', 0):.2f}")
                                c4.metric("ND", f"{nd_value:.1f}" if nd_value > 0 else "None")
                                if not np.isnan(energy_mean):
                                    c5.metric("E (mJ)", f"{energy_mean:.3f}")
                        
                        html = fig.to_html(full_html=False, include_plotlyjs='cdn').encode()
                        html_buffer.writestr(f"{filename.replace('.asc', '')}.html", html)
                    
                    # Combined
                    label = f"QS={qs:.0f}" if not np.isnan(qs) else filename
                    if not np.isnan(energy_mean):
                        label += f" ({energy_mean:.2f}mJ)"
                    if nd_value > 0:
                        label += f" [ND{nd_value}]"
                    
                    combined_fig.add_trace(go.Scatter(
                        x=wl, y=counts_corrected,
                        mode='lines',
                        name=label
                    ))
                    
                    # Summary
                    summary_data.append({
                        "File": filename,
                        "QS Level": qs,
                        "Pump Energy (mJ)": energy_mean,
                        "Energy Std (mJ)": energy_std,
                        "ND Filter": nd_value,
                        "Correction Factor": 10**nd_value if nd_value > 0 else 1,
                        "Peak λ (nm)": result.peak_wavelength,
                        "Peak Intensity": result.peak_intensity,
                        "FWHM (nm)": result.fwhm,
                        "Integrated Intensity": result.integrated_intensity,
                        "R²": result.r_squared,
                        "SNR": result.snr,
                        "Fit Success": "✅" if result.fit_success else "❌"
                    })
                    
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
    if 'Pump Energy (mJ)' in summary_df.columns and summary_df['Pump Energy (mJ)'].notna().any():
        summary_df = summary_df.sort_values("Pump Energy (mJ)")
    else:
        summary_df = summary_df.sort_values("QS Level")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Files", len(summary_df))
    col2.metric("Avg R²", f"{summary_df['R²'].mean():.3f}")
    col3.metric("Avg FWHM", f"{summary_df['FWHM (nm)'].mean():.2f} nm")
    col4.metric("ND Corrected", summary_df[summary_df["ND Filter"] > 0].shape[0])
    col5.metric("Energy Cal.", summary_df[summary_df["Pump Energy (mJ)"].notna()].shape[0])
    
    # Combined plot
    st.markdown("---")
    st.subheader("🌈 Combined Spectra")
    combined_fig.update_layout(
        title="Spectral Evolution",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity (ND-corrected)",
        template="plotly_white",
        height=600
    )
    st.plotly_chart(combined_fig, use_container_width=True)
    
    if KALEIDO_AVAILABLE:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            try:
                img = fig_to_image(combined_fig, image_format, image_width, image_height, image_scale)
                st.download_button(
                    f"📥 Combined ({image_format.upper()})",
                    img,
                    f"combined.{image_format}",
                    f"image/{image_format}"
                )
            except:
                pass
    
    # Data table
    st.markdown("---")
    st.subheader("📋 Results Table")
    
    def highlight_r2(val):
        if pd.isna(val): return ''
        if val > 0.95: return 'background-color: #d4edda'
        if val > 0.85: return 'background-color: #fff3cd'
        return 'background-color: #f8d7da'
    
    styled = summary_df.style.applymap(highlight_r2, subset=['R²']).format({
        'Peak λ (nm)': '{:.2f}',
        'Peak Intensity': '{:.0f}',
        'FWHM (nm)': '{:.2f}',
        'Integrated Intensity': '{:.2e}',
        'R²': '{:.4f}',
        'SNR': '{:.1f}',
        'QS Level': lambda x: f'{x:.0f}' if not pd.isna(x) else '',
        'Pump Energy (mJ)': lambda x: f'{x:.4f}' if not pd.isna(x) else '',
        'Energy Std (mJ)': lambda x: f'{x:.4f}' if not pd.isna(x) else '',
        'ND Filter': lambda x: f'{x:.1f}' if x > 0 else ''
    })
    
    st.dataframe(styled, use_container_width=True)
    
    # Threshold
    use_energy = 'Pump Energy (mJ)' in summary_df.columns and summary_df['Pump Energy (mJ)'].notna().sum() > 3
    
    if use_energy or summary_df['QS Level'].notna().sum() > 3:
        st.markdown("---")
        st.subheader("🎯 Threshold Detection")
        
        if use_energy:
            valid = summary_df.dropna(subset=['Pump Energy (mJ)', 'Integrated Intensity'])
            threshold = detect_threshold(
                valid['Pump Energy (mJ)'].values,
                valid['Integrated Intensity'].values
            )
        else:
            valid = summary_df.dropna(subset=['QS Level', 'Integrated Intensity'])
            threshold = detect_threshold(
                valid['QS Level'].values,
                valid['Integrated Intensity'].values
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if threshold.threshold_found:
                if use_energy:
                    st.success(f"✅ Threshold: **{threshold.threshold_energy:.4f} mJ**")
                else:
                    st.success(f"✅ Threshold: QS **{threshold.threshold_qs:.1f}**")
            else:
                st.warning("⚠️ No threshold detected")
        with col2:
            st.metric("Slope (below)", f"{threshold.slope_below:.2e}")
        with col3:
            st.metric("Slope (above)", f"{threshold.slope_above:.2e}")
        
        threshold_fig = create_threshold_plot(summary_df, threshold, use_energy)
        st.plotly_chart(threshold_fig, use_container_width=True)
        
        if KALEIDO_AVAILABLE:
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                try:
                    img = fig_to_image(threshold_fig, image_format, int(image_width*1.5), int(image_height*1.2), image_scale)
                    st.download_button(
                        f"📥 Threshold ({image_format.upper()})",
                        img,
                        f"threshold.{image_format}",
                        f"image/{image_format}"
                    )
                except:
                    pass
    
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
            use_container_width=True
        )
    
    with cols[1]:
        st.download_button(
            "📦 HTML Plots",
            plot_zip.getvalue(),
            f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            "application/zip",
            use_container_width=True
        )
    
    with cols[2]:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Results', index=False)
            if energy_map:
                energy_cal_df = pd.DataFrame([
                    {'QS': qs, 'Mean (mJ)': d['mean'], 'Std (mJ)': d['std'], 'N': d['n_readings']}
                    for qs, d in energy_map.items()
                ]).sort_values('QS')
                energy_cal_df.to_excel(writer, sheet_name='Energy Cal', index=False)
        
        st.download_button(
            "📊 Excel",
            excel_buffer.getvalue(),
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.ms-excel",
            use_container_width=True
        )
    
    if KALEIDO_AVAILABLE and image_zip:
        with cols[3]:
            st.download_button(
                f"🖼️ Images ({image_format.upper()})",
                image_zip.getvalue(),
                f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                "application/zip",
                use_container_width=True
            )

else:
    # Welcome
    st.info("👆 Upload .asc files to begin")
    
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### Quick Start
        1. Upload .asc spectrum files
        2. Optionally upload energy calibration file
        3. View automated analysis
        4. Download results
        
        ### Energy File Format
        **Transposed Excel/CSV:**
        - Row 1: `QS_Level | 110 | 120 | 130 | ...`
        - Rows 2-11: `Energy 1-10` with values
        - Scientific notation OK (8.21E-06)
        - Auto-converts J to mJ
        
        ### File Naming
        - QS: `QS150.asc` or `sample_qs_100.asc`
        - ND: `QS150_ND2.asc` (intensity ×100)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
📧 varun.solanki@fau.de
</div>
""", unsafe_allow_html=True)


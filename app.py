# ==============================================================
# Streamlit App: Random Laser ASC Analyzer
# Complete Version with ND Correction, Energy Calibration & Wavelength Analysis
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
# SAMPLE METADATA EXTRACTION (FOR YOUR FORMAT)
# ==============================================================
def extract_thickness(filename: str) -> Optional[float]:
    """
    Extract thickness from filename
    Examples: 
    - "UL_5mm_..." -> 5.0 (upper layer)
    - "LL_10mm_..." -> 10.0 (lower layer)
    """
    patterns = [
        r'UL[_\s-]*(\d+\.?\d*)\s*mm',  # UL_5mm
        r'LL[_\s-]*(\d+\.?\d*)\s*mm',  # LL_10mm
        r'(\d+\.?\d*)\s*mm',            # 5mm, 10mm
        r't[_\s-]*(\d+\.?\d*)',         # t5, t_10
        r'thickness[_\s-]*(\d+\.?\d*)', # thickness_5
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

def extract_concentration(filename: str) -> Optional[Dict[str, float]]:
    """
    Extract concentration from filename (handles upper and lower layers)
    Examples:
    - "UL_5%IL_LL_1%IL" -> {'upper': 5.0, 'lower': 1.0}
    """
    conc_data = {'upper': None, 'lower': None}
    
    # Pattern for UL_5%IL (upper layer)
    ul_pattern = r'UL[_\s-]*(\d+\.?\d*)\s*%\s*IL'
    ul_match = re.search(ul_pattern, filename, re.IGNORECASE)
    if ul_match:
        conc_data['upper'] = float(ul_match.group(1))
    
    # Pattern for LL_1%IL (lower layer)
    ll_pattern = r'LL[_\s-]*(\d+\.?\d*)\s*%\s*IL'
    ll_match = re.search(ll_pattern, filename, re.IGNORECASE)
    if ll_match:
        conc_data['lower'] = float(ll_match.group(1))
    
    # Fallback patterns for simpler formats
    if conc_data['upper'] is None and conc_data['lower'] is None:
        simple_patterns = [
            r'(\d+\.?\d*)\s*%',          # 5%, 10.5%
            r'(\d+)p(\d+)',              # 1p5 -> 1.5
            r'c[_\s-]*(\d+\.?\d*)',      # c5, c_10
            r'conc[_\s-]*(\d+\.?\d*)',   # conc_5
        ]
        
        for pattern in simple_patterns:
            match = re.search(pattern, filename.lower())
            if match:
                if 'p' in pattern and len(match.groups()) > 1:
                    conc_data['upper'] = float(f"{match.group(1)}.{match.group(2)}")
                else:
                    conc_data['upper'] = float(match.group(1))
                break
    
    # Return None if no concentration found
    if conc_data['upper'] is None and conc_data['lower'] is None:
        return None
    
    return conc_data

def extract_nd(filename: str) -> float:
    """
    Extract ND/OD filter value from filename
    Examples: "OD=2" -> 2.0, "ND2" -> 2.0
    """
    patterns = [
        r'OD\s*[=_-]*\s*(\d+\.?\d*)',   # OD=2, OD_2
        r'ND\s*[=_-]*\s*(\d+\.?\d*)',   # ND=2, ND_2
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return 0.0

def extract_qs(filename: str) -> float:
    """Extract Q-switch value from filename"""
    patterns = [
        r'QS[_\s-]+(\d+\.?\d*)',        # QS_110, QS 110
        r'QS(\d+\.?\d*)',                # QS110
        r'qs[_\s-]*(\d+\.?\d*)',         # qs_110
        r'q[_\s-]*(\d+\.?\d*)',          # q_110
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    return np.nan

def extract_dye_amount(filename: str) -> Optional[float]:
    """
    Extract dye amount from filename
    Examples: "17mgR6G" -> 17.0
    """
    patterns = [
        r'(\d+\.?\d*)\s*mg\s*R6G',      # 17mgR6G, 10mg R6G
        r'(\d+\.?\d*)\s*mg',             # 17mg
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

def extract_repetitions(filename: str) -> Optional[int]:
    """
    Extract number of repetitions from filename
    Examples: "10rep" -> 10
    """
    patterns = [
        r'(\d+)\s*rep',                  # 10rep, 10 rep
    ]
    
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
        except:
            popt, pcov = curve_fit(lorentzian, wl, counts, p0=p0, bounds=bounds, maxfev=50000, method='trf')
        
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
            'Amplitude': float(A), 'Center': float(x0), 'Gamma': float(gamma),
            'Baseline': float(y0), 'Std_Errors': [float(x) for x in np.sqrt(np.diag(pcov))]
        }
        
        return FitResult(float(x0), float(A + y0), float(fwhm), float(area), 
                        fit_y, float(r_squared), float(snr), fit_params, fit_success=True)
        
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
        
        return FitResult(float(peak_wl), float(peak_int),
                        float(fwhm_estimate) if not np.isnan(fwhm_estimate) else np.nan,
                        float(np.trapz(counts - np.min(counts), wl)),
                        counts.copy(), 0.0, float(calculate_snr(counts)),
                        {'error': str(e)}, fit_success=False)

# ==============================================================
# ENERGY CALIBRATION FUNCTIONS
# ==============================================================
@st.cache_data
def parse_energy_file(file_content: str, file_type: str, file_bytes: bytes = None) -> Dict[float, Dict]:
    """Parse energy calibration file with TRANSPOSED format"""
    energy_map = {}
    
    try:
        if file_bytes and ('xlsx' in str(file_type).lower() or 'xls' in str(file_type).lower()):
            import io
            df = pd.read_excel(io.BytesIO(file_bytes), header=None)
        else:
            for sep in ['\t', ',', ';', '|']:
                try:
                    df = pd.read_csv(StringIO(file_content), sep=sep, header=None)
                    break
                except:
                    continue
        
        first_row = df.iloc[0, :]
        
        if str(first_row.iloc[0]).lower().replace('_', '').replace(' ', '') in ['qslevel', 'qs', 'qswitch']:
            qs_levels = []
            for val in first_row.iloc[1:]:
                try:
                    qs_levels.append(float(val))
                except:
                    continue
            
            for qs in qs_levels:
                energy_map[qs] = {'readings': []}
            
            for row_idx in range(1, len(df)):
                row = df.iloc[row_idx, :]
                row_label = str(row.iloc[0]).lower()
                if not any(x in row_label for x in ['energy', 'e', 'reading', 'measurement']):
                    continue
                
                for col_idx, qs in enumerate(qs_levels):
                    try:
                        energy_val = float(row.iloc[col_idx + 1])
                        if energy_val < 0.1:
                            energy_val = energy_val * 1000
                        energy_map[qs]['readings'].append(energy_val)
                    except:
                        continue
            
            final_map = {}
            for qs in energy_map:
                readings = energy_map[qs]['readings']
                if readings:
                    final_map[qs] = {
                        'mean': np.mean(readings), 'std': np.std(readings),
                        'readings': readings, 'n_readings': len(readings)
                    }
            return final_map
        else:
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
                            'mean': np.mean(energy_readings), 'std': np.std(energy_readings),
                            'readings': energy_readings, 'n_readings': len(energy_readings)
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
    
    if qs_value in energy_map:
        return energy_map[qs_value]['mean'], energy_map[qs_value]['std']
    
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
@st.cache_data
def parse_asc_file(file_content: str, skip_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse .asc file"""
    df = pd.read_csv(StringIO(file_content), sep='\t', decimal=',', skiprows=skip_rows, engine='python')
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
    
    if nd_value > 0:
        fig.add_trace(go.Scatter(x=wl, y=counts_raw, mode='lines', name='Raw Data',
                                line=dict(color='lightgray', width=2), opacity=0.5))
    
    fig.add_trace(go.Scatter(x=wl, y=counts_corrected, mode='lines',
                            name='ND-Corrected Data' if nd_value > 0 else 'Data',
                            line=dict(color='#2E86AB', width=3)))
    
    if fit_result.fit_success and not np.isnan(fit_result.fwhm):
        fig.add_trace(go.Scatter(x=wl, y=fit_result.fit_y, mode='lines', name='Lorentzian Fit',
                                line=dict(color='red', width=3, dash='dash'), opacity=0.8))
        
        fig.add_vline(x=fit_result.peak_wavelength, line_dash="dot", line_color="green",
                     line_width=2, annotation_text=f"Peak: {fit_result.peak_wavelength:.2f} nm")
        
        gamma = fit_result.fwhm / 2
        x0 = fit_result.peak_wavelength
        half_max = fit_result.peak_intensity / 2
        
        fig.add_trace(go.Scatter(x=[x0-gamma, x0+gamma], y=[half_max, half_max],
                                mode='markers+text', marker=dict(size=12, color='orange', symbol='diamond'),
                                name=f'FWHM = {fit_result.fwhm:.2f} nm',
                                text=['', f'FWHM={fit_result.fwhm:.2f}nm'], textposition='top center'))
        
        fig.add_shape(type="line", x0=x0-gamma, y0=half_max, x1=x0+gamma, y1=half_max,
                     line=dict(color="orange", width=2, dash="dash"))
    
    title_html = f"<b>{filename}</b><br>"
    if energy_mean is not None and not np.isnan(energy_mean):
        title_html += f"<sub>Pump Energy: {energy_mean:.3f}±{energy_std:.3f} mJ</sub><br>"
    if nd_value > 0:
        title_html += f"<sub>ND: {nd_value} (×{10**nd_value:.0f})</sub><br>"
    if fit_result.fit_success:
        title_html += f"<sub>Peak: {fit_result.peak_wavelength:.2f} nm | "
        title_html += f"FWHM: {fit_result.fwhm:.2f} nm | R²: {fit_result.r_squared:.4f}</sub>"
    
    fig.update_layout(title=title_html, xaxis_title="Wavelength (nm)", yaxis_title="Intensity (counts)",
                     template="plotly_white", hovermode="x unified", height=500, showlegend=True)
    
    return fig

def create_threshold_plot(df: pd.DataFrame, threshold: ThresholdAnalysis, use_energy: bool = True) -> go.Figure:
    """Create threshold analysis with smooth curves"""
    from scipy.interpolate import make_interp_spline
    
    x_col = 'Pump Energy (mJ)' if use_energy and 'Pump Energy (mJ)' in df.columns else 'QS Level'
    x_label = "Pump Energy (mJ)" if use_energy else "Q-Switch Level"
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=(f"Integrated Intensity vs {x_label}", f"FWHM vs {x_label}",
                                      f"Peak Wavelength vs {x_label}", f"Peak Intensity vs {x_label}"),
                       vertical_spacing=0.12, horizontal_spacing=0.1)
    
    valid = df.dropna(subset=[x_col])
    x_values = valid[x_col].values
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    
    # Calculate ranges
    int_min, int_max = valid['Integrated Intensity'].min(), valid['Integrated Intensity'].max()
    int_padding = (int_max - int_min) * 0.1
    int_range = [int_min - int_padding, int_max + int_padding]
    
    peak_min, peak_max = valid['Peak Intensity'].min(), valid['Peak Intensity'].max()
    peak_padding = (peak_max - peak_min) * 0.1
    peak_range = [peak_min - peak_padding, peak_max + peak_padding]
    
    fwhm_min, fwhm_max = valid['FWHM (nm)'].min(), valid['FWHM (nm)'].max()
    fwhm_padding = (fwhm_max - fwhm_min) * 0.1
    fwhm_range = [max(0, fwhm_min - fwhm_padding), fwhm_max + fwhm_padding]
    
    wl_min, wl_max = valid['Peak λ (nm)'].min(), valid['Peak λ (nm)'].max()
    wl_padding = max((wl_max - wl_min) * 0.1, 2)
    wl_range = [wl_min - wl_padding, wl_max + wl_padding]
    
    def create_smooth_curve(x, y, num_points=300):
        if len(x) < 4:
            return x, y
        try:
            spline = make_interp_spline(x, y, k=min(3, len(x)-1))
            x_smooth = np.linspace(x.min(), x.max(), num_points)
            y_smooth = spline(x_smooth)
            return x_smooth, y_smooth
        except:
            return x, y
    
    # Plot 1: Integrated Intensity
    y_int = valid['Integrated Intensity'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_int)
    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines',
                            line=dict(width=3, color='red', shape='spline'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_sorted, y=y_int, mode='markers',
                            marker=dict(size=10, color='red', symbol='circle', line=dict(width=2, color='white')),
                            error_x=dict(type='data',
                                       array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                                       visible=True if 'Energy Std (mJ)' in valid.columns else False,
                                       thickness=1.5, width=4),
                            showlegend=False), row=1, col=1)
    
    if threshold.threshold_found and threshold.threshold_energy:
        fig.add_vline(x=threshold.threshold_energy, line_dash="dash", line_color="green", line_width=2,
                     annotation_text=f"Threshold: {threshold.threshold_energy:.4f} mJ",
                     annotation_position="top", row=1, col=1)
    
    # Plot 2: FWHM
    y_fwhm = valid['FWHM (nm)'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_fwhm)
    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines',
                            line=dict(width=3, color='blue', shape='spline'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_sorted, y=y_fwhm, mode='markers',
                            marker=dict(size=10, color='blue', symbol='circle', line=dict(width=2, color='white')),
                            error_x=dict(type='data',
                                       array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                                       visible=True if 'Energy Std (mJ)' in valid.columns else False,
                                       thickness=1.5, width=4),
                            showlegend=False), row=1, col=2)
    
    # Plot 3: Peak Wavelength
    y_wl = valid['Peak λ (nm)'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_wl)
    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines',
                            line=dict(width=3, color='purple', shape='spline'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_sorted, y=y_wl, mode='markers',
                            marker=dict(size=10, color='purple', symbol='circle', line=dict(width=2, color='white')),
                            error_x=dict(type='data',
                                       array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                                       visible=True if 'Energy Std (mJ)' in valid.columns else False,
                                       thickness=1.5, width=4),
                            showlegend=False), row=2, col=1)
    
    # Plot 4: Peak Intensity
    y_peak = valid['Peak Intensity'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_peak)
    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines',
                            line=dict(width=3, color='orange', shape='spline'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_sorted, y=y_peak, mode='markers',
                            marker=dict(size=10, color='orange', symbol='circle', line=dict(width=2, color='white')),
                            error_x=dict(type='data',
                                       array=valid['Energy Std (mJ)'].values[sort_idx] if 'Energy Std (mJ)' in valid.columns else None,
                                       visible=True if 'Energy Std (mJ)' in valid.columns else False,
                                       thickness=1.5, width=4),
                            showlegend=False), row=2, col=2)
    
    # Update axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text=x_label, row=row, col=col)
    
    fig.update_yaxes(title_text="Integrated Intensity", range=int_range, row=1, col=1)
    fig.update_yaxes(title_text="FWHM (nm)", range=fwhm_range, row=1, col=2)
    fig.update_yaxes(title_text="Wavelength (nm)", range=wl_range, row=2, col=1)
    fig.update_yaxes(title_text="Counts", range=peak_range, row=2, col=2)
    
    fig.update_layout(height=700, showlegend=False, template="plotly_white",
                     title_text="<b>Threshold Analysis Dashboard</b>")
    
    return fig

def create_energy_wavelength_plot(df: pd.DataFrame) -> go.Figure:
    """Create Energy vs Peak Wavelength plot grouped by sample conditions"""
    from scipy.interpolate import make_interp_spline
    
    fig = go.Figure()
    
    if 'Pump Energy (mJ)' not in df.columns or df['Pump Energy (mJ)'].isna().all():
        st.warning("⚠️ Energy calibration data not available")
        return None
    
    if 'Sample Label Short' in df.columns:
        groups = df.groupby('Sample Label Short')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
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
                    spline = make_interp_spline(x_data, y_data, k=min(3, len(x_data)-1))
                    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                    y_smooth = spline(x_smooth)
                    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines',
                                            line=dict(width=3, color=color),
                                            name=label, showlegend=True, legendgroup=label))
                except:
                    pass
            
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='markers',
                marker=dict(size=12, color=color, symbol='circle', line=dict(width=2, color='white')),
                name=label, showlegend=False, legendgroup=label,
                error_x=dict(type='data',
                           array=group_df['Energy Std (mJ)'].values if 'Energy Std (mJ)' in group_df.columns else None,
                           visible=True if 'Energy Std (mJ)' in group_df.columns else False,
                           thickness=1.5, width=4),
                hovertemplate=f'<b>{label}</b><br>Energy: %{{x:.4f}} mJ<br>Peak λ: %{{y:.2f}} nm<br><extra></extra>'
            ))
    
    fig.update_layout(
        title="<b>Peak Wavelength vs Pump Energy</b><br><sub>Grouped by Sample Conditions</sub>",
        xaxis_title="Pump Energy (mJ)", yaxis_title="Peak Wavelength (nm)",
        template="plotly_white", hovermode="closest", height=600, showlegend=True,
        legend=dict(title="Sample Conditions", x=1.02, y=1,
                   bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1)
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
# STREAMLIT APP
# ==============================================================
st.set_page_config(page_title="Random Laser Analyzer", layout="wide", page_icon="🔬")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔬 Random Laser Analyzer</p>', unsafe_allow_html=True)
st.markdown("**Lorentzian fitting • ND/OD correction • Energy calibration • Wavelength analysis**")

if not KALEIDO_AVAILABLE:
    st.warning("⚠️ Image export disabled. Install kaleido: `pip install kaleido`")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("📁 File Settings", expanded=True):
        skip_rows = st.number_input("Header rows to skip", 0, 100, 38)
        show_individual = st.checkbox("Show individual plots", True)
        show_fit_params = st.checkbox("Show fit parameters", False)
        apply_nd = st.checkbox("Apply ND/OD correction", True)
    
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
    - ✅ ND/OD filter correction
    - ✅ Energy calibration
    - ✅ Wavelength vs Energy
    - ✅ UL/LL layer support
    - ✅ Threshold detection
    
    ### 📝 Your Format
    `UL_5mm_QS_110_10rep_17mgR6G_UL_5%IL_LL_1%IL_OD=2`
    
    Extracts:
    - Thickness: `UL_5mm`
    - QS: `QS_110`
    - Concentrations: `UL_5%IL`, `LL_1%IL`
    - Dye: `17mgR6G`
    - OD: `OD=2`
    """)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Spectrum Files (.asc)")
    uploaded_files = st.file_uploader("Upload spectrum files", accept_multiple_files=True,
                                     type=['asc'], key="spectrum_files")

with col2:
    st.subheader("⚡ Energy Calibration")
    
    with st.expander("📋 Format"):
        st.markdown("""
        ```
        QS_Level  110      120      130
        Energy 1  8.2E-06  2.6E-05  6.7E-05
        Energy 2  9.0E-06  2.5E-05  6.3E-05
        ...
        ```
        """)
    
    energy_file = st.file_uploader("Upload energy file (optional)",
                                   type=['csv', 'txt', 'tsv', 'xlsx', 'xls'], key="energy_file")
    
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
                        {'QS': qs, 'Mean (mJ)': d['mean'], 'Std (mJ)': d['std'], 'N': d['n_readings']}
                        for qs, d in energy_map.items()
                    ]).sort_values('QS')
                    
                    st.dataframe(energy_df.style.format({
                        'QS': '{:.0f}', 'Mean (mJ)': '{:.4f}', 'Std (mJ)': '{:.4f}', 'N': '{:.0f}'
                    }))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=energy_df['QS'], y=energy_df['Mean (mJ)'],
                        error_y=dict(type='data', array=energy_df['Std (mJ)'], visible=True),
                        mode='markers+lines', marker=dict(size=10, color='blue')
                    ))
                    fig.update_layout(title="Energy Calibration", xaxis_title="QS Level",
                                     yaxis_title="Energy (mJ)", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        energy_map = {}

# [CONTINUE IN NEXT MESSAGE - CHARACTER LIMIT]

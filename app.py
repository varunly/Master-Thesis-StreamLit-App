# ==============================================================
# Streamlit App: Random Laser ASC Analyzer
# With Thickness-Dependent Energy Calibration
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
            except:
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
# ENERGY CALIBRATION PARSER (WITH THICKNESS SUPPORT)
# ==============================================================
def parse_pasted_energy_data(pasted_text: str) -> Dict:
    """
    Parse pasted energy calibration data with optional thickness support
    
    Format 1 (Simple - no thickness):
    200  190  180  170
    0.008  0.025  0.058  0.122
    ...
    
    Format 2 (With thickness in first column):
    Thickness  200  190  180  170
    3          0.008  0.025  0.058  0.122
    5          0.010  0.030  0.070  0.150
    7          0.012  0.035  0.082  0.178
    ...
    """
    energy_map = {}
    has_thickness = False
    
    try:
        lines = [line.strip() for line in pasted_text.strip().split('\n') if line.strip()]
        
        if len(lines) < 2:
            st.error("❌ Need at least 2 rows (header + measurements)")
            return {}
        
        # Parse first line: check if it starts with "Thickness" or numeric values
        first_line = lines[0]
        
        # Try different separators
        for sep in ['\t', ',', ';', ' ']:
            parts = [p.strip() for p in first_line.split(sep) if p.strip()]
            if len(parts) > 1:
                # Check if first element is "Thickness" or similar
                if parts[0].lower() in ['thickness', 't', 'thick']:
                    has_thickness = True
                    qs_values = [float(x) for x in parts[1:]]
                    st.success(f"✅ Detected THICKNESS-DEPENDENT format")
                    st.info(f"✅ Found {len(qs_values)} QS levels: {qs_values}")
                else:
                    # Try to parse as QS values
                    try:
                        qs_values = [float(x) for x in parts]
                        if all(100 <= x <= 500 for x in qs_values):
                            has_thickness = False
                            st.success(f"✅ Detected SIMPLE format (no thickness)")
                            st.info(f"✅ Found {len(qs_values)} QS levels: {qs_values}")
                        else:
                            continue
                    except:
                        continue
                break
        
        if not qs_values:
            st.error("❌ Could not parse QS levels from first row")
            return {}
        
        # Parse data rows
        if has_thickness:
            # Format: Thickness in first column
            thickness_map = {}
            data_rows = lines[1:]  # Skip header
            
            for line in data_rows:
                for sep in ['\t', ',', ';', ' ']:
                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                    if len(parts) == len(qs_values) + 1:  # thickness + QS values
                        try:
                            thickness = float(parts[0])
                            energies = [float(x) for x in parts[1:]]
                            
                            # Convert J to mJ if needed
                            energies = [e * 1000 if e < 0.01 else e for e in energies]
                            
                            if thickness not in thickness_map:
                                thickness_map[thickness] = {qs: [] for qs in qs_values}
                            
                            for qs, energy in zip(qs_values, energies):
                                thickness_map[thickness][qs].append(energy)
                            break
                        except:
                            continue
            
            # Calculate statistics for each thickness-QS combination
            for thickness, qs_dict in thickness_map.items():
                energy_map[thickness] = {}
                for qs, readings in qs_dict.items():
                    if len(readings) > 0:
                        energy_map[thickness][qs] = {
                            'mean': np.mean(readings),
                            'std': np.std(readings),
                            'readings': readings,
                            'n_readings': len(readings),
                            'od': 0.0
                        }
            
            st.success(f"✅ Parsed {len(energy_map)} thickness levels with {len(qs_values)} QS levels each")
            
        else:
            # Format: Simple (no thickness) - original behavior
            temp_map = {qs: [] for qs in qs_values}
            data_rows = lines[1:]
            
            for line in data_rows:
                for sep in ['\t', ',', ';', ' ']:
                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                    if len(parts) == len(qs_values):
                        try:
                            energies = [float(x) for x in parts]
                            # Convert J to mJ if needed
                            energies = [e * 1000 if e < 0.01 else e for e in energies]
                            
                            for qs, energy in zip(qs_values, energies):
                                temp_map[qs].append(energy)
                            break
                        except:
                            continue
            
            # Calculate statistics (store under thickness=None for compatibility)
            energy_map[None] = {}
            for qs, readings in temp_map.items():
                if len(readings) > 0:
                    energy_map[None][qs] = {
                        'mean': np.mean(readings),
                        'std': np.std(readings),
                        'readings': readings,
                        'n_readings': len(readings),
                        'od': 0.0
                    }
            
            st.success(f"✅ Parsed {len(energy_map[None])} QS levels")
        
        return energy_map
        
    except Exception as e:
        st.error(f"❌ Error parsing data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return {}


def interpolate_energy(qs_value: float, thickness_value: Optional[float], 
                       energy_map: Dict) -> Tuple[float, float]:
    """
    Interpolate energy for a given QS and thickness using the calibration map.
    Returns (mean_energy, std_energy)
    
    Supports both:
    - Simple format: energy_map[None][qs]
    - Thickness format: energy_map[thickness][qs]
    """
    if not energy_map:
        return np.nan, np.nan
    
    # Check if thickness-dependent or simple format
    thickness_keys = list(energy_map.keys())
    
    if None in thickness_keys:
        # Simple format (no thickness dependency)
        qs_map = energy_map[None]
        qs_levels = sorted(qs_map.keys())
        
        # Exact QS match
        if qs_value in qs_map:
            return qs_map[qs_value]['mean'], qs_map[qs_value]['std']
        
        # QS interpolation
        if qs_value < min(qs_levels) or qs_value > max(qs_levels):
            nearest_qs = min(qs_levels, key=lambda x: abs(x - qs_value))
            return qs_map[nearest_qs]['mean'], qs_map[nearest_qs]['std']
        
        for i in range(len(qs_levels) - 1):
            if qs_levels[i] <= qs_value <= qs_levels[i+1]:
                qs1, qs2 = qs_levels[i], qs_levels[i+1]
                e1 = qs_map[qs1]['mean']
                e2 = qs_map[qs2]['mean']
                std1 = qs_map[qs1]['std']
                std2 = qs_map[qs2]['std']
                
                t = (qs_value - qs1) / (qs2 - qs1)
                return e1 + t * (e2 - e1), std1 + t * (std2 - std1)
    
    else:
        # Thickness-dependent format
        available_thicknesses = sorted([t for t in thickness_keys if t is not None])
        
        if not available_thicknesses:
            return np.nan, np.nan
        
        # Find closest thickness
        if thickness_value is None:
            st.warning(f"⚠️ No thickness found in filename. Using {available_thicknesses[0]}mm")
            thickness_value = available_thicknesses[0]
        
        # Exact thickness match
        if thickness_value in energy_map:
            qs_map = energy_map[thickness_value]
            qs_levels = sorted(qs_map.keys())
            
            # Exact QS match
            if qs_value in qs_map:
                return qs_map[qs_value]['mean'], qs_map[qs_value]['std']
            
            # QS interpolation
            if qs_value < min(qs_levels) or qs_value > max(qs_levels):
                nearest_qs = min(qs_levels, key=lambda x: abs(x - qs_value))
                return qs_map[nearest_qs]['mean'], qs_map[nearest_qs]['std']
            
            for i in range(len(qs_levels) - 1):
                if qs_levels[i] <= qs_value <= qs_levels[i+1]:
                    qs1, qs2 = qs_levels[i], qs_levels[i+1]
                    e1 = qs_map[qs1]['mean']
                    e2 = qs_map[qs2]['mean']
                    std1 = qs_map[qs1]['std']
                    std2 = qs_map[qs2]['std']
                    
                    t = (qs_value - qs1) / (qs2 - qs1)
                    return e1 + t * (e2 - e1), std1 + t * (std2 - std1)
        
        else:
            # Thickness interpolation needed
            if thickness_value < min(available_thicknesses):
                thick = min(available_thicknesses)
                st.warning(f"⚠️ Thickness {thickness_value}mm below calibration range. Using {thick}mm")
                thickness_value = thick
            elif thickness_value > max(available_thicknesses):
                thick = max(available_thicknesses)
                st.warning(f"⚠️ Thickness {thickness_value}mm above calibration range. Using {thick}mm")
                thickness_value = thick
            else:
                # Interpolate between two thicknesses
                for i in range(len(available_thicknesses) - 1):
                    t1, t2 = available_thicknesses[i], available_thicknesses[i+1]
                    if t1 <= thickness_value <= t2:
                        # Get energy for both thicknesses at this QS
                        e1, std1 = interpolate_energy(qs_value, t1, {t1: energy_map[t1]})
                        e2, std2 = interpolate_energy(qs_value, t2, {t2: energy_map[t2]})
                        
                        # Linear interpolation between thicknesses
                        t_frac = (thickness_value - t1) / (t2 - t1)
                        return e1 + t_frac * (e2 - e1), std1 + t_frac * (std2 - std1)
            
            # Fallback: use exact thickness value found above
            qs_map = energy_map[thickness_value]
            qs_levels = sorted(qs_map.keys())
            
            if qs_value in qs_map:
                return qs_map[qs_value]['mean'], qs_map[qs_value]['std']
            
            nearest_qs = min(qs_levels, key=lambda x: abs(x - qs_value))
            return qs_map[nearest_qs]['mean'], qs_map[nearest_qs]['std']
    
    return np.nan, np.nan

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
# VISUALIZATION FUNCTIONS (keeping all original ones)
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
                            name='OD-Corrected Data' if nd_value > 0 else 'Data',
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
        title_html += f"<sub>OD: {nd_value} (×{10**nd_value:.0f})</sub><br>"
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
    """Create Energy vs Peak Wavelength plot"""
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

def create_energy_intensity_plot(df: pd.DataFrame) -> go.Figure:
    """Create Energy vs Peak Intensity plot"""
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
            
            group_df = group_df.dropna(subset=['Pump Energy (mJ)', 'Peak Intensity'])
            if len(group_df) == 0:
                continue
            
            group_df = group_df.sort_values('Pump Energy (mJ)')
            x_data = group_df['Pump Energy (mJ)'].values
            y_data = group_df['Peak Intensity'].values
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
                hovertemplate=f'<b>{label}</b><br>Energy: %{{x:.4f}} mJ<br>Peak Intensity: %{{y:.0f}}<br><extra></extra>'
            ))
    
    fig.update_layout(
        title="<b>Peak Intensity vs Pump Energy</b><br><sub>Grouped by Sample Conditions</sub>",
        xaxis_title="Pump Energy (mJ)", yaxis_title="Peak Intensity (counts)",
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
st.markdown("**Lorentzian fitting • OD correction • Thickness-dependent energy calibration • Complete analysis**")

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
    st.markdown("""
    - ✅ Lorentzian fitting
    - ✅ OD/ND correction
    - ✅ **Thickness-dependent energy**
    - ✅ Wavelength vs Energy
    - ✅ Intensity vs Energy
    - ✅ Threshold detection
    - ✅ Sample grouping
    
    ### 📝 Filename Format
    `UL_5mm_QS_110_10rep_17mgR6G_UL_5%IL_LL_1%IL_OD=2.asc`
    
    Extracts:
    - **Thickness: `UL_5mm`** ← Important!
    - QS: `QS_110`
    - Conc: `UL_5%IL`, `LL_1%IL`
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
    
    with st.expander("📋 Paste Formats", expanded=False):
        st.markdown("""
        ### Format 1: Simple (No Thickness)
        Use when QS level is the only variable:
        ```
        200  190  180  170  160
        0.008  0.025  0.058  0.122  0.245
        0.007  0.026  0.060  0.120  0.250
        ... (more measurements)
        ```
        
        ---
        
        ### Format 2: With Thickness ✨ NEW!
        Use when varying thickness at same QS:
        ```
        Thickness  200  190  180  170  160
        3          0.008  0.025  0.058  0.122  0.245
        3          0.007  0.026  0.060  0.120  0.250
        5          0.010  0.030  0.070  0.150  0.300
        5          0.011  0.029  0.068  0.148  0.295
        7          0.012  0.035  0.082  0.178  0.355
        7          0.013  0.034  0.080  0.175  0.350
        ```
        
        **The code auto-detects which format!**
        
        ---
        
        ### Notes:
        - Tab or comma separated
        - Scientific notation OK
        - Auto-converts J to mJ
        - Multiple measurements per thickness/QS combo
        """)
    
    energy_input = st.text_area(
        "Paste Energy Calibration Data",
        height=300,
        placeholder="Thickness\t200\t190\t180\t170\n3\t0.008\t0.025\t0.058\t0.122\n5\t0.010\t0.030\t0.070\t0.150\n...",
        help="Paste table with or without thickness column"
    )
    
    energy_map = {}
    
    if energy_input.strip():
        energy_map = parse_pasted_energy_data(energy_input)
        
        if energy_map:
            with st.expander("📊 Calibration Data", expanded=True):
                # Check if thickness-dependent
                thickness_keys = [k for k in energy_map.keys() if k is not None]
                
                if thickness_keys:
                    # Thickness-dependent display
                    st.success(f"✅ **Thickness-Dependent Calibration**")
                    st.info(f"📏 Thickness levels: {sorted(thickness_keys)}")
                    
                    for thickness in sorted(thickness_keys):
                        st.markdown(f"### Thickness = {thickness} mm")
                        
                        energy_df = pd.DataFrame([
                            {
                                'QS': qs,
                                'Mean (mJ)': d['mean'],
                                'Std (mJ)': d['std'],
                                'N': d['n_readings']
                            }
                            for qs, d in energy_map[thickness].items()
                        ]).sort_values('QS', ascending=False)
                        
                        st.dataframe(energy_df.style.format({
                            'QS': '{:.0f}',
                            'Mean (mJ)': '{:.4f}',
                            'Std (mJ)': '{:.4f}',
                            'N': '{:.0f}'
                        }), use_container_width=True)
                    
                    # Plot all thickness curves
                    fig = go.Figure()
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    
                    for idx, thickness in enumerate(sorted(thickness_keys)):
                        energy_df = pd.DataFrame([
                            {'QS': qs, 'Mean (mJ)': d['mean'], 'Std (mJ)': d['std']}
                            for qs, d in energy_map[thickness].items()
                        ]).sort_values('QS', ascending=False)
                        
                        fig.add_trace(go.Scatter(
                            x=energy_df['QS'],
                            y=energy_df['Mean (mJ)'],
                            error_y=dict(type='data', array=energy_df['Std (mJ)'], visible=True),
                            mode='markers+lines',
                            marker=dict(size=10),
                            line=dict(width=2),
                            name=f'{thickness} mm'
                        ))
                    
                    fig.update_layout(
                        title="<b>Energy Calibration Curves (All Thicknesses)</b>",
                        xaxis_title="QS Level",
                        yaxis_title="Pump Energy (mJ)",
                        template="plotly_white",
                        height=500,
                        xaxis=dict(autorange='reversed')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # Simple format display
                    st.success(f"✅ **Simple Calibration (No Thickness)**")
                    
                    energy_df = pd.DataFrame([
                        {
                            'QS': qs,
                            'Mean (mJ)': d['mean'],
                            'Std (mJ)': d['std'],
                            'N': d['n_readings']
                        }
                        for qs, d in energy_map[None].items()
                    ]).sort_values('QS', ascending=False)
                    
                    st.dataframe(energy_df.style.format({
                        'QS': '{:.0f}',
                        'Mean (mJ)': '{:.4f}',
                        'Std (mJ)': '{:.4f}',
                        'N': '{:.0f}'
                    }), use_container_width=True)
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=energy_df['QS'],
                        y=energy_df['Mean (mJ)'],
                        error_y=dict(type='data', array=energy_df['Std (mJ)'], visible=True),
                        mode='markers+lines',
                        marker=dict(size=10, color='#667eea'),
                        line=dict(width=2, color='#667eea')
                    ))
                    fig.update_layout(
                        title="<b>Energy Calibration Curve</b>",
                        xaxis_title="QS Level",
                        yaxis_title="Pump Energy (mJ)",
                        template="plotly_white",
                        height=400,
                        xaxis=dict(autorange='reversed')
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Paste your energy calibration data above")

# ==============================================================
# MAIN PROCESSING (continue with rest of the code...)
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
        thickness_keys = [k for k in energy_map.keys() if k is not None]
        if thickness_keys:
            st.info(f"⚡ **Thickness-dependent energy calibration active** ({len(thickness_keys)} thickness levels)")
        else:
            st.info(f"⚡ **Energy calibration active** (simple mode)")
    
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
                    thickness = extract_thickness(filename)
                    nd_value = extract_nd(filename) if apply_nd else 0.0
                    concentration = extract_concentration(filename)
                    dye_amount = extract_dye_amount(filename)
                    repetitions = extract_repetitions(filename)
                    sample_label = get_sample_label(thickness, concentration, dye_amount)
                    sample_label_short = get_short_label(thickness, concentration)
                    
                    # Energy (now with thickness)
                    if energy_map and not np.isnan(qs):
                        energy_mean, energy_std = interpolate_energy(qs, thickness, energy_map)
                    else:
                        energy_mean, energy_std = np.nan, np.nan
                    
                    # OD correction
                    if nd_value > 0:
                        counts_corrected = apply_nd_correction(counts_raw, nd_value)
                        if show_individual:
                            st.info(f"🔧 OD={nd_value} correction (×{10**nd_value:.0f})")
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
                                    img_bytes = fig_to_image(fig, image_format, image_width, 
                                                            image_height, image_scale)
                                    st.download_button(f"📥 {image_format.upper()}", img_bytes,
                                                      f"{filename.replace('.asc', '')}.{image_format}",
                                                      f"image/{image_format}", key=f"dl_{idx}")
                                    if img_buffer:
                                        img_buffer.writestr(f"{filename.replace('.asc', '')}.{image_format}", 
                                                          img_bytes)
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
                    
                    # Combined
                    label = f"QS={qs:.0f}" if not np.isnan(qs) else filename
                    if thickness:
                        label += f" | {thickness}mm"
                    if not np.isnan(energy_mean):
                        label += f" ({energy_mean:.2f}mJ)"
                    if nd_value > 0:
                        label += f" [OD{nd_value}]"
                    
                    combined_fig.add_trace(go.Scatter(x=wl, y=counts_corrected, mode='lines', name=label))
                    
                    # Summary
                    summary_data.append({
                        "File": filename,
                        "Thickness (mm)": thickness,
                        "UL Concentration (%)": concentration.get('upper') if concentration else None,
                        "LL Concentration (%)": concentration.get('lower') if concentration else None,
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
                        "Fit Success": "✅" if result.fit_success else "❌"
                    })
                    
                except Exception as e:
                    st.error(f"Error: {filename}: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    continue
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
        
        finally:
            if img_buffer:
                img_buffer.close()
    
    status.success("✅ Processing complete!")
    progress_bar.empty()
    
    # ==============================================================
    # RESULTS (rest is same as before...)
    # ==============================================================
    
    st.markdown("---")
    st.subheader("📊 Summary Statistics")
    
    summary_df = pd.DataFrame(summary_data)
    if 'Pump Energy (mJ)' in summary_df.columns and summary_df['Pump Energy (mJ)'].notna().any():
        summary_df = summary_df.sort_values("Pump Energy (mJ)")
    else:
        summary_df = summary_df.sort_values("QS Level")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Files", len(summary_df))
    col2.metric("Avg R²", f"{summary_df['R²'].mean():.3f}")
    col3.metric("Avg FWHM", f"{summary_df['FWHM (nm)'].mean():.2f} nm")
    col4.metric("OD Corrected", summary_df[summary_df["OD Filter"] > 0].shape[0])
    col5.metric("Energy Cal.", summary_df[summary_df["Pump Energy (mJ)"].notna()].shape[0])
    col6.metric("Conditions", summary_df['Sample Label Short'].nunique())
    
    # Combined plot
    st.markdown("---")
    st.subheader("🌈 Combined Spectra")
    combined_fig.update_layout(title="Spectral Evolution", xaxis_title="Wavelength (nm)",
                               yaxis_title="Intensity (OD-corrected)", template="plotly_white", height=600)
    st.plotly_chart(combined_fig, use_container_width=True)
    
    if KALEIDO_AVAILABLE:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            try:
                img = fig_to_image(combined_fig, image_format, image_width, image_height, image_scale)
                st.download_button(f"📥 Combined ({image_format.upper()})", img,
                                  f"combined.{image_format}", f"image/{image_format}")
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
        'Peak λ (nm)': '{:.2f}', 'Peak Intensity': '{:.0f}', 'FWHM (nm)': '{:.2f}',
        'Integrated Intensity': '{:.2e}', 'R²': '{:.4f}', 'SNR': '{:.1f}',
        'QS Level': lambda x: f'{x:.0f}' if not pd.isna(x) else '',
        'Pump Energy (mJ)': lambda x: f'{x:.4f}' if not pd.isna(x) else '',
        'Energy Std (mJ)': lambda x: f'{x:.4f}' if not pd.isna(x) else '',
        'OD Filter': lambda x: f'{x:.1f}' if x > 0 else '',
        'Thickness (mm)': lambda x: f'{x:.1f}' if not pd.isna(x) else '',
        'UL Concentration (%)': lambda x: f'{x:.1f}' if not pd.isna(x) else '',
        'LL Concentration (%)': lambda x: f'{x:.1f}' if not pd.isna(x) else '',
        'Dye Amount (mg)': lambda x: f'{x:.1f}' if not pd.isna(x) else '',
        'Repetitions': lambda x: f'{x:.0f}' if not pd.isna(x) else ''
    })
    
    st.dataframe(styled, use_container_width=True)
    
    # Threshold
    use_energy = 'Pump Energy (mJ)' in summary_df.columns and summary_df['Pump Energy (mJ)'].notna().sum() > 3
    
    if use_energy or summary_df['QS Level'].notna().sum() > 3:
        st.markdown("---")
        st.subheader("🎯 Threshold Detection")
        
        if use_energy:
            valid = summary_df.dropna(subset=['Pump Energy (mJ)', 'Integrated Intensity'])
            threshold = detect_threshold(valid['Pump Energy (mJ)'].values, valid['Integrated Intensity'].values)
        else:
            valid = summary_df.dropna(subset=['QS Level', 'Integrated Intensity'])
            threshold = detect_threshold(valid['QS Level'].values, valid['Integrated Intensity'].values)
        
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
                    img = fig_to_image(threshold_fig, image_format, int(image_width*1.5), 
                                      int(image_height*1.2), image_scale)
                    st.download_button(f"📥 Threshold ({image_format.upper()})", img,
                                      f"threshold.{image_format}", f"image/{image_format}")
                except:
                    pass
    
    # Energy vs Wavelength plot
    if 'Pump Energy (mJ)' in summary_df.columns and summary_df['Pump Energy (mJ)'].notna().sum() > 2:
        st.markdown("---")
        st.subheader("📈 Peak Wavelength Evolution")
        
        energy_wl_fig = create_energy_wavelength_plot(summary_df)
        
        if energy_wl_fig:
            st.plotly_chart(energy_wl_fig, use_container_width=True)
            
            if KALEIDO_AVAILABLE:
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    try:
                        img = fig_to_image(energy_wl_fig, image_format, image_width, image_height, image_scale)
                        st.download_button(f"📥 Energy-λ ({image_format.upper()})", img,
                                          f"energy_wavelength.{image_format}", f"image/{image_format}",
                                          key="dl_e_wl")
                    except:
                        pass
    
    # Energy vs Intensity plot
    if 'Pump Energy (mJ)' in summary_df.columns and summary_df['Pump Energy (mJ)'].notna().sum() > 2:
        st.markdown("---")
        st.subheader("💡 Peak Intensity Evolution")
        
        energy_int_fig = create_energy_intensity_plot(summary_df)
        
        if energy_int_fig:
            st.plotly_chart(energy_int_fig, use_container_width=True)
            
            if KALEIDO_AVAILABLE:
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    try:
                        img = fig_to_image(energy_int_fig, image_format, image_width, image_height, image_scale)
                        st.download_button(f"📥 Energy-Intensity ({image_format.upper()})", img,
                                          f"energy_intensity.{image_format}", f"image/{image_format}",
                                          key="dl_e_int")
                    except:
                        pass
    
    # Downloads
    st.markdown("---")
    st.subheader("💾 Export All Results")
    
    cols = st.columns(4 if KALEIDO_AVAILABLE else 3)
    
    with cols[0]:
        csv = summary_df.to_csv(index=False).encode()
        st.download_button("📥 CSV", csv, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                          "text/csv", use_container_width=True)
    
    with cols[1]:
        st.download_button("📦 HTML", plot_zip.getvalue(),
                          f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                          "application/zip", use_container_width=True)
    
    with cols[2]:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Results', index=False)
            if energy_map:
                # Export calibration data
                cal_data = []
                for thickness_key in energy_map.keys():
                    for qs, d in energy_map[thickness_key].items():
                        cal_data.append({
                            'Thickness': thickness_key if thickness_key is not None else 'N/A',
                            'QS': qs,
                            'Mean (mJ)': d['mean'],
                            'Std (mJ)': d['std'],
                            'N': d['n_readings']
                        })
                energy_cal_df = pd.DataFrame(cal_data).sort_values(['Thickness', 'QS'])
                energy_cal_df.to_excel(writer, sheet_name='Energy Cal', index=False)
        
        st.download_button("📊 Excel", excel_buffer.getvalue(),
                          f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                          "application/vnd.ms-excel", use_container_width=True)
    
    if KALEIDO_AVAILABLE and image_zip:
        with cols[3]:
            st.download_button(f"🖼️ Images", image_zip.getvalue(),
                              f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                              "application/zip", use_container_width=True)

else:
    # Welcome
    st.info("👆 Upload .asc files to begin")
    
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### Quick Start
        1. **Upload .asc spectrum files** (left column)
        2. **Paste energy calibration** (right column, optional)
        3. View automated analysis
        4. Download results
        
        ### Filename Format
        `UL_5mm_QS_110_10rep_17mgR6G_UL_5%IL_LL_1%IL_OD=2.asc`
        
        **Auto-extracts:**
        - **Thickness: `UL_5mm` → 5mm** ← Important for thickness-dependent calibration!
        - QS: `QS_110` → 110
        - UL Conc: `UL_5%IL` → 5%
        - LL Conc: `LL_1%IL` → 1%
        - Dye: `17mgR6G` → 17mg
        - OD: `OD=2` → 2 (×100)
        
        ### Energy Calibration Formats
        
        **Format 1: Simple (same thickness, varying QS)**
        ```
        200  190  180  170
        0.008  0.025  0.058  0.122
        0.007  0.026  0.060  0.120
        ```
        
        **Format 2: Thickness-Dependent ✨**
        ```
        Thickness  200  190  180  170
        3          0.008  0.025  0.058  0.122
        3          0.007  0.026  0.060  0.120
        5          0.010  0.030  0.070  0.150
        5          0.011  0.029  0.068  0.148
        7          0.012  0.035  0.082  0.178
        7          0.013  0.034  0.080  0.175
        ```
        
        **The code automatically detects which format you're using!**
        
        ### Example Use Case
        You measured pump energy for:
        - QS=110 at thickness=3mm → E₁
        - QS=110 at thickness=5mm → E₂
        - QS=110 at thickness=7mm → E₃
        
        The code will interpolate energy based on BOTH QS and thickness!
        
        ### Plots Generated
        - **Individual Spectra**: Lorentzian fits with OD correction
        - **Combined Spectra**: All overlaid
        - **Threshold Dashboard**: 4-panel analysis
        - **Energy vs Wavelength**: Peak shifts
        - **Energy vs Intensity**: Growth curves
        
        All grouped by sample conditions!
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
📧 varun.solanki@fau.de | Friedrich-Alexander-Universität Erlangen-Nürnberg
</div>
""", unsafe_allow_html=True)

# ==============================================================
# Streamlit App: Random Laser ASC Analyzer
# Supports BOTH QS-based AND Thickness-based Energy Calibration
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
# ENERGY CALIBRATION PARSER - QS-BASED (Original Format)
# ==============================================================
def parse_qs_energy_data(pasted_text: str) -> Dict[float, Dict]:
    """
    Parse QS-energy calibration data (varying QS, same thickness)
    
    Format:
    Row 1: QS levels (200, 190, 180, ...)
    Rows 2-11: Energy measurements
    Row 13: Average (optional)
    Row 14: OD values (optional)
    """
    energy_map = {}
    
    try:
        lines = [line.strip() for line in pasted_text.strip().split('\n') if line.strip()]
        
        if len(lines) < 2:
            st.error("❌ Need at least 2 rows (QS levels + measurements)")
            return {}
        
        # Parse row 1: QS levels
        qs_line = lines[0]
        qs_values = None
        for sep in ['\t', ',', ';', ' ']:
            parts = [p.strip() for p in qs_line.split(sep) if p.strip()]
            if len(parts) > 1:
                try:
                    qs_values = [float(x) for x in parts]
                    if all(100 <= x <= 500 for x in qs_values):
                        st.success(f"✅ Found {len(qs_values)} QS levels: {qs_values}")
                        break
                except:
                    continue
        
        if qs_values is None:
            st.error("❌ Could not parse QS levels from first row")
            return {}
        
        # Initialize energy map
        for qs in qs_values:
            energy_map[qs] = {'readings': [], 'od': 0.0}
        
        # Parse measurement rows (2-11)
        measurement_rows = lines[1:11] if len(lines) >= 11 else lines[1:]
        
        for line in measurement_rows:
            # Skip if this is the average row or OD row
            if 'OD' in line.upper() or 'ND' in line.upper():
                continue
            
            values = None
            for sep in ['\t', ',', ';', ' ']:
                parts = [p.strip() for p in line.split(sep) if p.strip()]
                if len(parts) == len(qs_values):
                    try:
                        values = [float(x) for x in parts]
                        break
                    except:
                        continue
            
            if values is None:
                continue
            
            for qs, energy in zip(qs_values, values):
                # Convert J to mJ if needed
                if energy < 0.01:
                    energy *= 1000
                energy_map[qs]['readings'].append(energy)
        
        # Parse OD values (last row with OD=)
        for line in lines:
            if 'OD' in line.upper() or 'ND' in line.upper():
                for sep in ['\t', ',', ';', ' ']:
                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                    # Remove 'OD=' or similar labels
                    od_values = []
                    for part in parts:
                        if 'OD' not in part.upper() and 'ND' not in part.upper():
                            try:
                                od_values.append(float(part))
                            except:
                                continue
                    
                    if len(od_values) == len(qs_values):
                        for qs, od in zip(qs_values, od_values):
                            energy_map[qs]['od'] = od
                        st.info(f"✅ Parsed OD values: {od_values}")
                        break
        
        # Calculate statistics
        final_map = {}
        for qs, data in energy_map.items():
            if len(data['readings']) > 0:
                final_map[qs] = {
                    'mean': np.mean(data['readings']),
                    'std': np.std(data['readings']),
                    'readings': data['readings'],
                    'n_readings': len(data['readings']),
                    'od': data['od']
                }
        
        st.success(f"✅ Parsed {len(final_map)} QS levels with {sum(d['n_readings'] for d in final_map.values())} measurements")
        
        return final_map
        
    except Exception as e:
        st.error(f"❌ Error parsing QS-energy data: {str(e)}")
        return {}


# ==============================================================
# ENERGY CALIBRATION PARSER - THICKNESS-BASED
# ==============================================================
def parse_thickness_energy_data(pasted_text: str) -> Dict[float, Dict]:
    """
    Parse thickness-energy calibration data (same QS, varying thickness)
    
    Format:
    Thickness  Energy  StdDev
    3          0.150   0.005
    5          0.200   0.008
    """
    thickness_energy_map = {}
    
    try:
        lines = [line.strip() for line in pasted_text.strip().split('\n') if line.strip()]
        
        if len(lines) < 1:
            st.error("❌ Need at least 1 row of data")
            return {}
        
        # Check if first line is header
        first_line = lines[0]
        start_idx = 0
        
        if any(keyword in first_line.lower() for keyword in ['thickness', 'energy', 'std']):
            start_idx = 1
            st.info("✅ Detected header row, skipping it")
        
        data_lines = lines[start_idx:]
        
        for line in data_lines:
            for sep in ['\t', ',', ';', ' ']:
                parts = [p.strip() for p in line.split(sep) if p.strip()]
                
                if len(parts) >= 2:
                    try:
                        thickness = float(parts[0])
                        energy = float(parts[1])
                        std = float(parts[2]) if len(parts) > 2 else 0.0
                        
                        # Convert J to mJ if needed
                        if energy < 0.01:
                            energy *= 1000
                        if std < 0.01 and std > 0:
                            std *= 1000
                        
                        thickness_energy_map[thickness] = {
                            'energy': energy,
                            'std': std
                        }
                        break
                    except ValueError:
                        continue
        
        if len(thickness_energy_map) == 0:
            st.error("❌ No valid thickness-energy pairs found")
            return {}
        
        st.success(f"✅ Parsed {len(thickness_energy_map)} thickness-energy pairs")
        
        return thickness_energy_map
        
    except Exception as e:
        st.error(f"❌ Error parsing thickness-energy data: {str(e)}")
        return {}


# ==============================================================
# ENERGY INTERPOLATION FUNCTIONS
# ==============================================================
def interpolate_energy_by_qs(qs_value: float, energy_map: Dict[float, Dict]) -> Tuple[float, float]:
    """Get energy for a given QS value"""
    if not energy_map:
        return np.nan, np.nan
    
    qs_levels = sorted(energy_map.keys())
    
    # Exact match
    if qs_value in energy_map:
        return energy_map[qs_value]['mean'], energy_map[qs_value]['std']
    
    # Out of range
    if qs_value < min(qs_levels) or qs_value > max(qs_levels):
        nearest_qs = min(qs_levels, key=lambda x: abs(x - qs_value))
        return energy_map[nearest_qs]['mean'], energy_map[nearest_qs]['std']
    
    # Linear interpolation
    for i in range(len(qs_levels) - 1):
        if qs_levels[i] <= qs_value <= qs_levels[i+1]:
            qs1, qs2 = qs_levels[i], qs_levels[i+1]
            e1, e2 = energy_map[qs1]['mean'], energy_map[qs2]['mean']
            std1, std2 = energy_map[qs1]['std'], energy_map[qs2]['std']
            
            t = (qs_value - qs1) / (qs2 - qs1)
            return e1 + t * (e2 - e1), std1 + t * (std2 - std1)
    
    return np.nan, np.nan


def interpolate_energy_by_thickness(thickness_value: Optional[float], 
                                    thickness_energy_map: Dict[float, Dict]) -> Tuple[float, float]:
    """Get energy for a given thickness (same QS)"""
    if not thickness_energy_map or thickness_value is None:
        return np.nan, np.nan
    
    thickness_levels = sorted(thickness_energy_map.keys())
    
    # Exact match
    if thickness_value in thickness_energy_map:
        return thickness_energy_map[thickness_value]['energy'], thickness_energy_map[thickness_value]['std']
    
    # Out of range
    if thickness_value < min(thickness_levels):
        nearest = min(thickness_levels)
        return thickness_energy_map[nearest]['energy'], thickness_energy_map[nearest]['std']
    
    if thickness_value > max(thickness_levels):
        nearest = max(thickness_levels)
        return thickness_energy_map[nearest]['energy'], thickness_energy_map[nearest]['std']
    
    # Linear interpolation
    for i in range(len(thickness_levels) - 1):
        t1, t2 = thickness_levels[i], thickness_levels[i+1]
        if t1 <= thickness_value <= t2:
            e1 = thickness_energy_map[t1]['energy']
            e2 = thickness_energy_map[t2]['energy']
            std1 = thickness_energy_map[t1]['std']
            std2 = thickness_energy_map[t2]['std']
            
            t_frac = (thickness_value - t1) / (t2 - t1)
            return e1 + t_frac * (e2 - e1), std1 + t_frac * (std2 - std1)
    
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
# VISUALIZATION FUNCTIONS (abbreviated - same as before)
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
st.markdown("**Lorentzian fitting • OD correction • Flexible energy calibration • Complete analysis**")

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
    - ✅ **QS-based energy** (varying QS)
    - ✅ **Thickness-based energy** (varying thickness)
    - ✅ Threshold detection
    - ✅ Complete plots
    """)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Spectrum Files (.asc)")
    uploaded_files = st.file_uploader("Upload spectrum files", accept_multiple_files=True,
                                     type=['asc'], key="spectrum_files")

with col2:
    st.subheader("⚡ Energy Calibration")
    
    # Calibration mode selector
    cal_mode = st.radio(
        "Calibration Mode:",
        ["QS-based (Varying QS)", "Thickness-based (Same QS)"],
        help="Choose based on your experiment"
    )
    
    if cal_mode == "QS-based (Varying QS)":
        # QS-BASED MODE
        with st.expander("📋 QS-Based Format", expanded=False):
            st.markdown("""
            ### Your Original Format
            
            **Row 1:** QS levels
            **Rows 2-11:** Energy measurements
            **Row 13:** Average (optional)
            **Row 14:** OD values (optional)
            
            ```
            200	190	180	170	160
            7.48E-06	2.28E-05	5.24E-05	1.19E-04	2.15E-04
            6.01E-06	2.46E-05	5.80E-05	1.12E-04	2.03E-04
            ... (8 more measurement rows)
            
            0.000008	0.000025	0.000058	0.000122	0.000215
            0	0	2	2	2
            ```
            
            - Auto-converts J to mJ
            - Reads OD values per QS
            """)
        
        energy_input = st.text_area(
            "Paste QS-Energy Calibration",
            height=300,
            placeholder="200\t190\t180\t170\n7.48E-06\t2.28E-05\t5.24E-05\t1.19E-04\n...",
            key="qs_energy"
        )
        
        qs_energy_map = {}
        thickness_energy_map = {}
        
        if energy_input.strip():
            qs_energy_map = parse_qs_energy_data(energy_input)
            
            if qs_energy_map:
                with st.expander("📊 QS Calibration Data", expanded=True):
                    energy_df = pd.DataFrame([
                        {
                            'QS': qs,
                            'Energy (mJ)': d['mean'],
                            'Std (mJ)': d['std'],
                            'N': d['n_readings'],
                            'OD': d['od']
                        }
                        for qs, d in qs_energy_map.items()
                    ]).sort_values('QS', ascending=False)
                    
                    st.dataframe(energy_df.style.format({
                        'QS': '{:.0f}',
                        'Energy (mJ)': '{:.4f}',
                        'Std (mJ)': '{:.4f}',
                        'N': '{:.0f}',
                        'OD': '{:.1f}'
                    }), use_container_width=True)
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=energy_df['QS'],
                        y=energy_df['Energy (mJ)'],
                        error_y=dict(type='data', array=energy_df['Std (mJ)'], visible=True),
                        mode='markers+lines',
                        marker=dict(size=10, color='#667eea'),
                        line=dict(width=2, color='#667eea')
                    ))
                    fig.update_layout(
                        title="<b>Energy vs QS</b>",
                        xaxis_title="QS Level",
                        yaxis_title="Pump Energy (mJ)",
                        template="plotly_white",
                        height=350,
                        xaxis=dict(autorange='reversed')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if energy_df['OD'].sum() > 0:
                        st.info(f"🔍 **OD filters:** {energy_df[energy_df['OD']>0]['OD'].unique().tolist()}")
        else:
            st.info("💡 Paste your QS-energy calibration data")
    
    else:
        # THICKNESS-BASED MODE
        with st.expander("📋 Thickness-Based Format", expanded=False):
            st.markdown("""
            ### Same QS, Different Thickness
            
            **Simple:**
            ```
            3  0.150
            5  0.200
            7  0.250
            ```
            
            **With StdDev:**
            ```
            Thickness  Energy  StdDev
            3          0.150   0.005
            5          0.200   0.008
            7          0.250   0.010
            ```
            """)
        
        thickness_input = st.text_area(
            "Paste Thickness-Energy Calibration",
            height=250,
            placeholder="3\t0.150\t0.005\n5\t0.200\t0.008\n7\t0.250\t0.010",
            key="thickness_energy"
        )
        
        qs_energy_map = {}
        thickness_energy_map = {}
        
        if thickness_input.strip():
            thickness_energy_map = parse_thickness_energy_data(thickness_input)
            
            if thickness_energy_map:
                with st.expander("📊 Thickness Calibration Data", expanded=True):
                    cal_df = pd.DataFrame([
                        {
                            'Thickness (mm)': t,
                            'Energy (mJ)': d['energy'],
                            'Std (mJ)': d['std']
                        }
                        for t, d in thickness_energy_map.items()
                    ]).sort_values('Thickness (mm)')
                    
                    st.dataframe(cal_df.style.format({
                        'Thickness (mm)': '{:.1f}',
                        'Energy (mJ)': '{:.4f}',
                        'Std (mJ)': '{:.4f}'
                    }), use_container_width=True)
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cal_df['Thickness (mm)'],
                        y=cal_df['Energy (mJ)'],
                        error_y=dict(type='data', array=cal_df['Std (mJ)'], visible=True),
                        mode='markers+lines',
                        marker=dict(size=12, color='#ff7f0e'),
                        line=dict(width=2, color='#ff7f0e')
                    ))
                    fig.update_layout(
                        title="<b>Energy vs Thickness</b>",
                        xaxis_title="Thickness (mm)",
                        yaxis_title="Pump Energy (mJ)",
                        template="plotly_white",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 Paste your thickness-energy calibration data")

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
    
    if qs_energy_map:
        st.info(f"⚡ **QS-based energy calibration active** ({len(qs_energy_map)} QS levels)")
    elif thickness_energy_map:
        st.info(f"⚡ **Thickness-based energy calibration active** ({len(thickness_energy_map)} thickness levels)")
    
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
                    
                    # Energy (based on calibration mode)
                    if qs_energy_map and not np.isnan(qs):
                        energy_mean, energy_std = interpolate_energy_by_qs(qs, qs_energy_map)
                    elif thickness_energy_map and thickness is not None:
                        energy_mean, energy_std = interpolate_energy_by_thickness(thickness, thickness_energy_map)
                    else:
                        energy_mean, energy_std = np.nan, np.nan
                    
                    # OD correction
                    if nd_value > 0:
                        counts_corrected = apply_nd_correction(counts_raw, nd_value)
                    else:
                        counts_corrected = counts_raw.copy()
                    
                    # Analyze
                    result = analyze_spectrum(wl, counts_corrected)
                    
                    # Plot individual
                    if show_individual:
                        fig = create_spectrum_plot(wl, counts_raw, counts_corrected, result, 
                                                  filename, nd_value, energy_mean, energy_std)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        html = fig.to_html(full_html=False, include_plotlyjs='cdn').encode()
                        html_buffer.writestr(f"{filename.replace('.asc', '')}.html", html)
                    
                    # Combined label
                    label = f"QS={qs:.0f}" if not np.isnan(qs) else filename
                    if thickness:
                        label += f" | {thickness}mm"
                    if not np.isnan(energy_mean):
                        label += f" ({energy_mean:.3f}mJ)"
                    
                    combined_fig.add_trace(go.Scatter(x=wl, y=counts_corrected, mode='lines', name=label))
                    
                    # Summary
                    summary_data.append({
                        "File": filename,
                        "Thickness (mm)": thickness,
                        "QS Level": qs,
                        "Pump Energy (mJ)": energy_mean,
                        "Energy Std (mJ)": energy_std,
                        "OD Filter": nd_value,
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
    
    # Results
    st.markdown("---")
    st.subheader("📊 Summary Statistics")
    
    summary_df = pd.DataFrame(summary_data)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Files", len(summary_df))
    col2.metric("Avg R²", f"{summary_df['R²'].mean():.3f}")
    col3.metric("Avg FWHM", f"{summary_df['FWHM (nm)'].mean():.2f} nm")
    col4.metric("Energy Cal.", summary_df[summary_df["Pump Energy (mJ)"].notna()].shape[0])
    
    # Combined plot
    st.markdown("---")
    st.subheader("🌈 Combined Spectra")
    combined_fig.update_layout(title="Spectral Evolution", xaxis_title="Wavelength (nm)",
                               yaxis_title="Intensity (counts)", template="plotly_white", height=600)
    st.plotly_chart(combined_fig, use_container_width=True)
    
    # Data table
    st.markdown("---")
    st.subheader("📋 Results Table")
    st.dataframe(summary_df, use_container_width=True)
    
    # Downloads
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    cols = st.columns(3)
    
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
        
        st.download_button("📊 Excel", excel_buffer.getvalue(),
                          f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                          "application/vnd.ms-excel", use_container_width=True)

else:
    st.info("👆 Upload .asc files to begin")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
📧 varun.solanki@fau.de | Friedrich-Alexander-Universität Erlangen-Nürnberg
</div>
""", unsafe_allow_html=True)

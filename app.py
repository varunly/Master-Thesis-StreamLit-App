# Complete Code with Vertical-Only Error Bars

#Here's the complete updated code with all horizontal error bars removed:

#```python
# ==============================================================
# Streamlit App: Random Laser ASC Analyzer
# WITH GAUSSIAN ERROR CORRECTION & THICKNESS-DEPENDENT ENERGY
# VERTICAL ERROR BARS ONLY (Standard Convention)
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
# DATA CLASSES (WITH UNCERTAINTIES)
# ==============================================================
@dataclass
class FitResult:
    """Data class for spectral analysis results with Gaussian uncertainties"""
    peak_wavelength: float
    peak_intensity: float
    fwhm: float
    integrated_intensity: float
    fit_y: np.ndarray
    r_squared: float
    snr: float
    fit_params: Dict
    fit_success: bool = True
    
    # Uncertainty estimates
    peak_intensity_uncertainty: float = 0.0
    integrated_intensity_uncertainty: float = 0.0
    wavelength_uncertainty: float = 0.0
    fwhm_uncertainty: float = 0.0

@dataclass
class ThresholdAnalysis:
    """Data class for threshold detection results"""
    threshold_qs: Optional[float]
    threshold_energy: Optional[float]
    slope_below: float
    slope_above: float
    threshold_found: bool
    threshold_uncertainty: float = 0.0

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
    """
    energy_map = {}
    has_thickness = False
    
    try:
        lines = [line.strip() for line in pasted_text.strip().split('\n') if line.strip()]
        
        if len(lines) < 2:
            st.error("Need at least 2 rows (header + measurements)")
            return {}
        
        first_line = lines[0]
        qs_values = None
        
        for sep in ['\t', ',', ';', ' ']:
            parts = [p.strip() for p in first_line.split(sep) if p.strip()]
            if len(parts) > 1:
                if parts[0].lower() in ['thickness', 't', 'thick']:
                    has_thickness = True
                    qs_values = [float(x) for x in parts[1:]]
                    st.success("Detected THICKNESS-DEPENDENT format")
                    st.info(f"QS levels: {qs_values}")
                else:
                    try:
                        qs_values = [float(x) for x in parts]
                        if all(100 <= x <= 500 for x in qs_values):
                            has_thickness = False
                            st.success("Detected SIMPLE format (no thickness)")
                            st.info(f"QS levels: {qs_values}")
                        else:
                            continue
                    except:
                        continue
                break
        
        if not qs_values:
            st.error("Could not parse QS levels from first row")
            return {}
        
        if has_thickness:
            thickness_map = {}
            data_rows = lines[1:]
            
            for line in data_rows:
                for sep in ['\t', ',', ';', ' ']:
                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                    if len(parts) == len(qs_values) + 1:
                        try:
                            thickness = float(parts[0])
                            energies = [float(x) for x in parts[1:]]
                            energies = [e * 1000 if e < 0.01 else e for e in energies]
                            
                            if thickness not in thickness_map:
                                thickness_map[thickness] = {qs: [] for qs in qs_values}
                            
                            for qs, energy in zip(qs_values, energies):
                                thickness_map[thickness][qs].append(energy)
                            break
                        except:
                            continue
            
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
            
            st.success(f"Parsed {len(energy_map)} thickness levels with {len(qs_values)} QS levels each")
        else:
            temp_map = {qs: [] for qs in qs_values}
            data_rows = lines[1:]
            
            for line in data_rows:
                for sep in ['\t', ',', ';', ' ']:
                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                    if len(parts) == len(qs_values):
                        try:
                            energies = [float(x) for x in parts]
                            energies = [e * 1000 if e < 0.01 else e for e in energies]
                            
                            for qs, energy in zip(qs_values, energies):
                                temp_map[qs].append(energy)
                            break
                        except:
                            continue
            
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
            
            st.success(f"Parsed {len(energy_map[None])} QS levels")
        
        return energy_map
        
    except Exception as e:
        st.error(f"Error parsing data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return {}

def interpolate_energy(qs_value: float, thickness_value: Optional[float], 
                       energy_map: Dict) -> Tuple[float, float]:
    """Interpolate energy for a given QS and thickness using the calibration map"""
    if not energy_map:
        return np.nan, np.nan
    
    thickness_keys = list(energy_map.keys())
    
    if None in thickness_keys:
        qs_map = energy_map[None]
        qs_levels = sorted(qs_map.keys())
        
        if qs_value in qs_map:
            return qs_map[qs_value]['mean'], qs_map[qs_value]['std']
        
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
        available_thicknesses = sorted([t for t in thickness_keys if t is not None])
        if not available_thicknesses:
            return np.nan, np.nan
        
        if thickness_value is None:
            thickness_value = available_thicknesses[0]
        
        if thickness_value in energy_map:
            qs_map = energy_map[thickness_value]
            qs_levels = sorted(qs_map.keys())
            
            if qs_value in qs_map:
                return qs_map[qs_value]['mean'], qs_map[qs_value]['std']
            
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
            if thickness_value < min(available_thicknesses):
                thickness_value = min(available_thicknesses)
            elif thickness_value > max(available_thicknesses):
                thickness_value = max(available_thicknesses)
            else:
                for i in range(len(available_thicknesses) - 1):
                    t1, t2 = available_thicknesses[i], available_thicknesses[i+1]
                    if t1 <= thickness_value <= t2:
                        e1, std1 = interpolate_energy(qs_value, t1, {t1: energy_map[t1]})
                        e2, std2 = interpolate_energy(qs_value, t2, {t2: energy_map[t2]})
                        t_frac = (thickness_value - t1) / (t2 - t1)
                        return e1 + t_frac * (e2 - e1), std1 + t_frac * (std2 - std1)
            
            qs_map = energy_map[thickness_value]
            qs_levels = sorted(qs_map.keys())
            if qs_value in qs_map:
                return qs_map[qs_value]['mean'], qs_map[qs_value]['std']
            nearest_qs = min(qs_levels, key=lambda x: abs(x - qs_value))
            return qs_map[nearest_qs]['mean'], qs_map[nearest_qs]['std']
    
    return np.nan, np.nan

# ==============================================================
# GAUSSIAN ERROR PROPAGATION & NORMALIZATION
# ==============================================================
def propagate_gaussian_error(intensity: float, intensity_std: float, 
                            energy: float, energy_std: float,
                            normalization_mode: str = 'none') -> Tuple[float, float]:
    """Propagate Gaussian uncertainties through normalization"""
    if normalization_mode == 'none':
        return intensity, intensity_std
    
    if energy == 0 or np.isnan(energy):
        return np.nan, np.nan
    
    if normalization_mode == 'linear':
        I_norm = intensity / energy
        rel_I = intensity_std / intensity if intensity != 0 else 0
        rel_E = energy_std / energy if energy != 0 else 0
        sigma_norm = I_norm * np.sqrt(rel_I**2 + rel_E**2)
        return I_norm, sigma_norm
    
    if normalization_mode == 'quadratic':
        I_norm = intensity / (energy ** 2)
        rel_I = intensity_std / intensity if intensity != 0 else 0
        rel_E = energy_std / energy if energy != 0 else 0
        sigma_norm = I_norm * np.sqrt(rel_I**2 + (2 * rel_E)**2)
        return I_norm, sigma_norm
    
    return intensity, intensity_std

def calculate_total_uncertainty(spectrum_uncertainty: float, 
                                energy_uncertainty: float,
                                intensity: float,
                                pump_energy: float) -> float:
    """Combine spectral and energy uncertainties (Gaussian quadrature)"""
    if intensity == 0 or pump_energy == 0:
        return spectrum_uncertainty
    rel_spectrum = spectrum_uncertainty / intensity if intensity != 0 else 0
    rel_energy = energy_uncertainty / pump_energy if pump_energy != 0 else 0
    rel_total = np.sqrt(rel_spectrum**2 + rel_energy**2)
    return intensity * rel_total

def weighted_least_squares_threshold(energies: np.ndarray, 
                                     intensities: np.ndarray,
                                     energy_uncertainties: np.ndarray,
                                     intensity_uncertainties: np.ndarray) -> Dict:
    """Gaussian-weighted threshold fit"""
    total_uncertainties = np.sqrt(
        intensity_uncertainties**2 + 
        (intensities * energy_uncertainties / (energies + 1e-10))**2
    )
    total_uncertainties = np.where(total_uncertainties > 0, total_uncertainties, 1.0)
    weights = 1 / (total_uncertainties**2)
    weights = weights / np.sum(weights)
    
    best_threshold = None
    best_residual = np.inf
    best_slopes = (0, 0)
    
    for i in range(2, len(energies)-2):
        E_below = energies[:i]
        I_below = intensities[:i]
        w_below = weights[:i]
        
        E_above = energies[i:]
        I_above = intensities[i:]
        w_above = weights[i:]
        
        try:
            p_below = np.polyfit(E_below, I_below, 1, w=w_below)
            p_above = np.polyfit(E_above, I_above, 1, w=w_above)
            
            if p_above[0] > 2 * p_below[0]:
                residual_below = np.sum(w_below * (I_below - np.polyval(p_below, E_below))**2)
                residual_above = np.sum(w_above * (I_above - np.polyval(p_above, E_above))**2)
                total_residual = residual_below + residual_above
                
                if total_residual < best_residual:
                    best_residual = total_residual
                    best_threshold = energies[i]
                    best_slopes = (p_below[0], p_above[0])
        except:
            continue
    
    if best_threshold is not None:
        near = np.abs(energies - best_threshold) < 0.2 * best_threshold
        if np.sum(near) > 0:
            threshold_uncertainty = np.mean(energy_uncertainties[near])
        else:
            threshold_uncertainty = np.mean(energy_uncertainties)
    else:
        threshold_uncertainty = np.nan
    
    return {
        'threshold': best_threshold,
        'uncertainty': threshold_uncertainty,
        'slope_below': best_slopes[0],
        'slope_above': best_slopes[1],
        'method': 'gaussian_weighted'
    }

# ==============================================================
# CORE SPECTRAL ANALYSIS FUNCTIONS
# ==============================================================
def lorentzian(x: np.ndarray, A: float, x0: float, gamma: float, y0: float) -> np.ndarray:
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + y0

def calculate_r_squared(y_actual: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = np.sum((y_actual - y_fit) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

def calculate_snr(signal: np.ndarray, noise_percentile: int = 10) -> float:
    noise = np.percentile(signal, noise_percentile)
    peak = np.max(signal)
    return peak / noise if noise > 0 else np.inf

@st.cache_data
def analyze_spectrum(wl: np.ndarray, counts: np.ndarray) -> FitResult:
    """Perform Lorentzian fitting with uncertainty estimation"""
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
        
        param_uncertainties = np.sqrt(np.diag(pcov))
        amplitude_unc = param_uncertainties[0]
        center_unc = param_uncertainties[1]
        gamma_unc = param_uncertainties[2]
        baseline_unc = param_uncertainties[3]
        
        peak_intensity_unc = np.sqrt(amplitude_unc**2 + baseline_unc**2)
        fwhm_unc = 2 * gamma_unc
        integrated_unc = amplitude_unc * (area / A) if A > 0 else 0
        
        fit_params = {
            'Amplitude': float(A), 'Center': float(x0), 'Gamma': float(gamma),
            'Baseline': float(y0), 'Std_Errors': [float(x) for x in param_uncertainties]
        }
        
        return FitResult(
            float(x0), float(A + y0), float(fwhm), float(area), 
            fit_y, float(r_squared), float(snr), fit_params, 
            fit_success=True,
            peak_intensity_uncertainty=float(peak_intensity_unc),
            integrated_intensity_uncertainty=float(integrated_unc),
            wavelength_uncertainty=float(center_unc),
            fwhm_uncertainty=float(fwhm_unc)
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
            float(peak_wl), float(peak_int),
            float(fwhm_estimate) if not np.isnan(fwhm_estimate) else np.nan,
            float(np.trapz(counts - np.min(counts), wl)),
            counts.copy(), 0.0, float(calculate_snr(counts)),
            {'error': str(e)}, fit_success=False,
            peak_intensity_uncertainty=float(peak_int * 0.1),
            integrated_intensity_uncertainty=float(peak_int * 0.15),
            wavelength_uncertainty=0.5,
            fwhm_uncertainty=1.0
        )

# ==============================================================
# THRESHOLD DETECTION
# ==============================================================
def detect_threshold(x_values: np.ndarray, intensities: np.ndarray, min_points: int = 3) -> ThresholdAnalysis:
    if len(x_values) < 2 * min_points:
        return ThresholdAnalysis(None, None, 0, 0, False, 0.0)
    
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
        return ThresholdAnalysis(None, best_threshold, best_slopes[0], best_slopes[1], found, 0.0)
    except:
        return ThresholdAnalysis(None, None, 0, 0, False, 0.0)

# ==============================================================
# FILE PARSING
# ==============================================================
@st.cache_data
def parse_asc_file(file_content: str, skip_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(StringIO(file_content), sep='\t', decimal=',', skiprows=skip_rows, engine='python')
    df = df.dropna(axis=1, how='all')
    if df.shape[1] < 2:
        raise ValueError("File must have at least 2 columns")
    wl = df.iloc[:, 0].to_numpy()
    counts = df.iloc[:, 1:].mean(axis=1).to_numpy()
    return wl, counts

# ==============================================================
# VISUALIZATION FUNCTIONS (VERTICAL ERROR BARS ONLY)
# ==============================================================
def create_spectrum_plot(wl: np.ndarray, counts_raw: np.ndarray, counts_corrected: np.ndarray,
                        fit_result: FitResult, filename: str, nd_value: float, 
                        energy_mean: float = None, energy_std: float = None) -> go.Figure:
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
                                mode='markers+text',
                                marker=dict(size=12, color='orange', symbol='diamond'),
                                name=f'FWHM = {fit_result.fwhm:.2f} nm',
                                text=['', f'FWHM={fit_result.fwhm:.2f}nm'],
                                textposition='top center'))
        
        fig.add_shape(type="line", x0=x0-gamma, y0=half_max, x1=x0+gamma, y1=half_max,
                     line=dict(color="orange", width=2, dash="dash"))
    
    title_html = f"<b>{filename}</b><br>"
    if energy_mean is not None and not np.isnan(energy_mean):
        title_html += f"<sub>Pump Energy: {energy_mean:.3f}±{energy_std:.3f} mJ</sub><br>"
    if nd_value > 0:
        title_html += f"<sub>OD: {nd_value} (×{10**nd_value:.0f})</sub><br>"
    if fit_result.fit_success:
        title_html += (
            f"<sub>Peak: {fit_result.peak_wavelength:.2f}±{fit_result.wavelength_uncertainty:.2f} nm | "
            f"FWHM: {fit_result.fwhm:.2f}±{fit_result.fwhm_uncertainty:.2f} nm | "
            f"R²: {fit_result.r_squared:.4f}</sub>"
        )
    
    fig.update_layout(title=title_html, xaxis_title="Wavelength (nm)", yaxis_title="Intensity (counts)",
                     template="plotly_white", hovermode="x unified", height=500, showlegend=True)
    
    return fig

def create_energy_wavelength_plot(df: pd.DataFrame, show_error_bars: bool = True) -> go.Figure:
    """Energy vs Peak Wavelength with VERTICAL error bars only."""
    from scipy.interpolate import make_interp_spline
    
    fig = go.Figure()
    
    if 'Pump Energy (mJ)' not in df.columns or df['Pump Energy (mJ)'].isna().all():
        st.warning("Energy calibration data not available")
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
            
            # Smooth curve
            if len(x_data) >= 4:
                try:
                    spline = make_interp_spline(x_data, y_data, k=min(3, len(x_data)-1))
                    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                    y_smooth = spline(x_smooth)
                    fig.add_trace(go.Scatter(
                        x=x_smooth, y=y_smooth, mode='lines',
                        line=dict(width=3, color=color),
                        name=label, showlegend=True, legendgroup=label
                    ))
                except:
                    pass
            
            # ========== FIX: Handle small/missing error values ==========
            error_y_values = None
            if 'Wavelength Uncertainty' in group_df.columns:
                raw_errors = group_df['Wavelength Uncertainty'].values
                
                # Replace NaN with 0, then ensure minimum visible error
                raw_errors = np.nan_to_num(raw_errors, nan=0.0)
                
                # If errors are very small, use a minimum value for visibility
                # Minimum 0.1 nm or 0.5% of wavelength range
                y_range = y_data.max() - y_data.min() if len(y_data) > 1 else 1.0
                min_error = max(0.1, y_range * 0.01)  # At least 0.1 nm or 1% of range
                
                # Apply minimum only if original errors are too small to see
                error_y_values = np.where(raw_errors < min_error * 0.1, min_error, raw_errors)
                
                # If all zeros, set a default visible error
                if np.all(error_y_values == 0) or np.all(np.isnan(error_y_values)):
                    error_y_values = np.full_like(y_data, min_error)
            else:
                # No uncertainty column - create default errors
                y_range = y_data.max() - y_data.min() if len(y_data) > 1 else 1.0
                error_y_values = np.full_like(y_data, max(0.2, y_range * 0.02))
            # ============================================================
            
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='markers',
                marker=dict(size=12, color=color, symbol='circle', line=dict(width=2, color='white')),
                name=label, showlegend=False, legendgroup=label,
                error_y=dict(
                    type='data', 
                    array=error_y_values,
                    visible=show_error_bars,
                    thickness=2, 
                    width=6,
                    color=color
                ),
                hovertemplate=(
                    f'<b>{label}</b><br>'
                    'Energy: %{x:.4f} mJ<br>'
                    'Peak λ: %{y:.2f} nm<br>'
                    '<extra></extra>'
                )
            ))
    
    fig.update_layout(
        title="<b>Peak Wavelength vs Pump Energy</b><br><sub>Vertical error bars: wavelength uncertainty (±σ<sub>λ</sub>)</sub>",
        xaxis_title="Pump Energy (mJ)", yaxis_title="Peak Wavelength (nm)",
        template="plotly_white", hovermode="closest", height=600, showlegend=True,
        legend=dict(title="Sample Conditions", x=1.02, y=1,
                   bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1)
    )
    
    return fig

def create_energy_intensity_plot(df: pd.DataFrame, show_error_bars: bool = True) -> go.Figure:
    """Energy vs Peak Intensity with VERTICAL error bars only."""
    from scipy.interpolate import make_interp_spline
    
    fig = go.Figure()
    
    if 'Pump Energy (mJ)' not in df.columns or df['Pump Energy (mJ)'].isna().all():
        st.warning("Energy calibration data not available")
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
            
            # Smooth curve
            if len(x_data) >= 4:
                try:
                    spline = make_interp_spline(x_data, y_data, k=min(3, len(x_data)-1))
                    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                    y_smooth = spline(x_smooth)
                    fig.add_trace(go.Scatter(
                        x=x_smooth, y=y_smooth, mode='lines',
                        line=dict(width=3, color=color),
                        name=label, showlegend=True, legendgroup=label
                    ))
                except:
                    pass
            
            # ONLY VERTICAL error bars (intensity uncertainty)
            error_y = group_df['Peak Intensity Uncertainty'].values if 'Peak Intensity Uncertainty' in group_df.columns else None
            
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='markers',
                marker=dict(size=12, color=color, symbol='circle', line=dict(width=2, color='white')),
                name=label, showlegend=False, legendgroup=label,
                # VERTICAL ERROR BARS ONLY
                error_y=dict(
                    type='data', 
                    array=error_y,
                    visible=show_error_bars and error_y is not None,
                    thickness=2, 
                    width=6,
                    color=color
                ),
                hovertemplate=(
                    f'<b>{label}</b><br>'
                    'Energy: %{x:.4f} mJ<br>'
                    'Peak Intensity: %{y:.0f}<br>'
                    '<extra></extra>'
                )
            ))
    
    fig.update_layout(
        title="<b>Peak Intensity vs Pump Energy</b><br><sub>Vertical error bars: intensity uncertainty (±σ<sub>I</sub>)</sub>",
        xaxis_title="Pump Energy (mJ)", yaxis_title="Peak Intensity (counts)",
        template="plotly_white", hovermode="closest", height=600, showlegend=True,
        legend=dict(title="Sample Conditions", x=1.02, y=1,
                   bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1)
    )
    
    return fig

def create_threshold_plot(df: pd.DataFrame, threshold: ThresholdAnalysis, use_energy: bool = True, 
                          show_error_bars: bool = True) -> go.Figure:
    """Threshold dashboard with VERTICAL error bars only."""
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
        vertical_spacing=0.12, horizontal_spacing=0.1
    )
    
    valid = df.dropna(subset=[x_col])
    x_values = valid[x_col].values
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    
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
    
    # ONLY VERTICAL error arrays
    error_int_array = valid['Integrated Uncertainty'].values[sort_idx] if 'Integrated Uncertainty' in valid.columns else None
    error_fwhm_array = valid['FWHM Uncertainty'].values[sort_idx] if 'FWHM Uncertainty' in valid.columns else None
    error_wl_array = valid['Wavelength Uncertainty'].values[sort_idx] if 'Wavelength Uncertainty' in valid.columns else None
    error_peak_array = valid['Peak Intensity Uncertainty'].values[sort_idx] if 'Peak Intensity Uncertainty' in valid.columns else None
    
    # Plot 1: Integrated Intensity
    y_int = valid['Integrated Intensity'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_int)
    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines',
                             line=dict(width=3, color='red'), showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_sorted, y=y_int, mode='markers',
        marker=dict(size=10, color='red', line=dict(width=2, color='white')),
        error_y=dict(
            type='data', array=error_int_array,
            visible=show_error_bars and error_int_array is not None,
            thickness=2, width=5, color='rgba(255,0,0,0.6)'
        ),
        showlegend=False
    ), row=1, col=1)
    if threshold.threshold_found and threshold.threshold_energy:
        fig.add_vline(
            x=threshold.threshold_energy, line_dash="dash", line_color="green", line_width=2,
            annotation_text=f"Threshold: {threshold.threshold_energy:.4f} mJ",
            annotation_position="top", row=1, col=1
        )
    
    # Plot 2: FWHM
    y_fwhm = valid['FWHM (nm)'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_fwhm)
    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines',
                             line=dict(width=3, color='blue'), showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(
        x=x_sorted, y=y_fwhm, mode='markers',
        marker=dict(size=10, color='blue', line=dict(width=2, color='white')),
        error_y=dict(
            type='data', array=error_fwhm_array,
            visible=show_error_bars and error_fwhm_array is not None,
            thickness=2, width=5, color='rgba(0,0,255,0.6)'
        ),
        showlegend=False
    ), row=1, col=2)
    
    # Plot 3: Peak Wavelength
    y_wl = valid['Peak λ (nm)'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_wl)
    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines',
                             line=dict(width=3, color='purple'), showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x_sorted, y=y_wl, mode='markers',
        marker=dict(size=10, color='purple', line=dict(width=2, color='white')),
        error_y=dict(
            type='data', array=error_wl_array,
            visible=show_error_bars and error_wl_array is not None,
            thickness=2, width=5, color='rgba(128,0,128,0.6)'
        ),
        showlegend=False
    ), row=2, col=1)
    
    # Plot 4: Peak Intensity
    y_peak = valid['Peak Intensity'].values[sort_idx]
    x_smooth, y_smooth = create_smooth_curve(x_sorted, y_peak)
    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines',
                             line=dict(width=3, color='orange'), showlegend=False),
                  row=2, col=2)
    fig.add_trace(go.Scatter(
        x=x_sorted, y=y_peak, mode='markers',
        marker=dict(size=10, color='orange', line=dict(width=2, color='white')),
        error_y=dict(
            type='data', array=error_peak_array,
            visible=show_error_bars and error_peak_array is not None,
            thickness=2, width=5, color='rgba(255,165,0,0.6)'
        ),
        showlegend=False
    ), row=2, col=2)
    
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text=x_label, row=row, col=col)
    
    fig.update_yaxes(title_text="Integrated Intensity", range=int_range, row=1, col=1)
    fig.update_yaxes(title_text="FWHM (nm)", range=fwhm_range, row=1, col=2)
    fig.update_yaxes(title_text="Wavelength (nm)", range=wl_range, row=2, col=1)
    fig.update_yaxes(title_text="Counts", range=peak_range, row=2, col=2)
    
    fig.update_layout(
        height=700, showlegend=False, template="plotly_white",
        title_text="<b>Threshold Analysis Dashboard</b><br><sub>Vertical error bars show measurement uncertainty</sub>"
    )
    
    return fig

def fig_to_image(fig: go.Figure, format: str, width: int, height: int, scale: int) -> bytes:
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
st.markdown("**Lorentzian fitting • Gaussian error analysis • Vertical error bars**")

if not KALEIDO_AVAILABLE:
    st.warning("Image export disabled. Install: `pip install kaleido`")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("📁 File Settings", expanded=True):
        skip_rows = st.number_input("Header rows to skip", 0, 100, 38)
        show_individual = st.checkbox("Show individual plots", True)
        show_fit_params = st.checkbox("Show fit parameters", False)
        apply_nd = st.checkbox("Apply OD/ND correction", True)
    
    with st.expander("📊 Gaussian Error Correction", expanded=True):
        enable_normalization = st.checkbox(
            "Enable Intensity Normalization", value=False,
            help="Normalize intensities by pump energy (I/E or I/E²)"
        )
        if enable_normalization:
            normalization_mode = st.selectbox(
                "Normalization Mode",
                ['linear', 'quadratic'],
                format_func=lambda x: {
                    'linear': 'Linear (I/E) - linear processes',
                    'quadratic': 'Quadratic (I/E²) - nonlinear processes'
                }[x]
            )
        else:
            normalization_mode = 'none'
        
        enable_weighted_threshold = st.checkbox(
            "Use Gaussian-Weighted Threshold", value=True,
            help="Account for energy uncertainties in threshold fitting"
        )
        
        show_error_bars = st.checkbox(
            "Show Vertical Error Bars", value=True,
            help="Show vertical error bars on all summary plots"
        )
    
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
    - ✅ Gaussian error propagation
    - ✅ OD/ND correction
    - ✅ Thickness-dependent energy calibration
    - ✅ Threshold detection
    - ✅ **Vertical error bars only**
    """)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Spectrum Files (.asc)")
    
    # Clear button functionality
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    
    uploaded_files = st.file_uploader(
        "Upload spectrum files",
        accept_multiple_files=True,
        type=['asc'],
        key=f"spectrum_files_{st.session_state.uploader_key}"
    )
    
    if uploaded_files:
        col_info, col_clear = st.columns([3, 1])
        with col_info:
            st.info(f"📁 {len(uploaded_files)} file(s) loaded")
        with col_clear:
            if st.button("🗑️ Clear all", type="secondary", use_container_width=True):
                st.session_state.uploader_key += 1
                st.rerun()

with col2:
    st.subheader("⚡ Energy Calibration")
    
    with st.expander("📋 Paste Formats", expanded=False):
        st.markdown("""
        ### Format 1: Simple (no thickness)
        ```
        200  190  180  170
        0.008  0.025  0.058  0.122
        0.007  0.026  0.060  0.120
        ```
        
        ### Format 2: With Thickness
        ```
        Thickness  200  190  180
        3          0.008  0.025  0.058
        5          0.010  0.030  0.070
        ```
        """)
    
    energy_input = st.text_area(
        "Paste Energy Calibration Data",
        height=250,
        placeholder="200\t190\t180\t170\n7.48E-06\t2.28E-05\t5.24E-05\t1.19E-04\n...",
        help="Paste from Excel; auto-detects format"
    )
    
    energy_map = {}
    
    if energy_input.strip():
        energy_map = parse_pasted_energy_data(energy_input)
        
        if energy_map:
            with st.expander("📊 Calibration Data", expanded=True):
                thickness_keys = [k for k in energy_map.keys() if k is not None]
                
                if thickness_keys:
                    st.success("Thickness-Dependent Calibration")
                    for thickness in sorted(thickness_keys):
                        st.markdown(f"**Thickness = {thickness} mm**")
                        energy_df = pd.DataFrame([
                            {'QS': qs, 'Mean (mJ)': d['mean'], 'Std (mJ)': d['std'], 'N': d['n_readings']}
                            for qs, d in energy_map[thickness].items()
                        ]).sort_values('QS', ascending=False)
                        st.dataframe(energy_df, use_container_width=True)
                    
                    fig = go.Figure()
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    for idx, thickness in enumerate(sorted(thickness_keys)):
                        energy_df = pd.DataFrame([
                            {'QS': qs, 'Mean (mJ)': d['mean'], 'Std (mJ)': d['std']}
                            for qs, d in energy_map[thickness].items()
                        ]).sort_values('QS', ascending=False)
                        fig.add_trace(go.Scatter(
                            x=energy_df['QS'], y=energy_df['Mean (mJ)'],
                            error_y=dict(type='data', array=energy_df['Std (mJ)'], visible=True),
                            mode='markers+lines', marker=dict(size=10),
                            line=dict(width=2), name=f'{thickness} mm'
                        ))
                    fig.update_layout(
                        title="<b>Energy Calibration Curves (by Thickness)</b>",
                        xaxis_title="QS Level", yaxis_title="Pump Energy (mJ)",
                        template="plotly_white", height=400,
                        xaxis=dict(autorange='reversed')
                    )
                    st.plotly_chart(fig, use_container_width=True, key="cal_thick")
                else:
                    st.success("Simple Calibration")
                    energy_df = pd.DataFrame([
                        {'QS': qs, 'Mean (mJ)': d['mean'], 'Std (mJ)': d['std'], 'N': d['n_readings']}
                        for qs, d in energy_map[None].items()
                    ]).sort_values('QS', ascending=False)
                    st.dataframe(energy_df, use_container_width=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=energy_df['QS'], y=energy_df['Mean (mJ)'],
                        error_y=dict(type='data', array=energy_df['Std (mJ)'], visible=True),
                        mode='markers+lines', marker=dict(size=10, color='#667eea'),
                        line=dict(width=2, color='#667eea')
                    ))
                    fig.update_layout(
                        title="<b>Energy Calibration Curve</b>",
                        xaxis_title="QS Level", yaxis_title="Pump Energy (mJ)",
                        template="plotly_white", height=400,
                        xaxis=dict(autorange='reversed')
                    )
                    st.plotly_chart(fig, use_container_width=True, key="cal_simple")
    else:
        st.info("💡 Paste energy calibration data to enable energy-based analysis")

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
        thickness_keys = [k for k in energy_map.keys() if k is not None]
        if thickness_keys:
            st.info(f"⚡ Thickness-dependent energy calibration active ({len(thickness_keys)} thickness levels)")
        else:
            st.info("⚡ Energy calibration active (simple mode)")
    
    with zipfile.ZipFile(plot_zip, "w", zipfile.ZIP_DEFLATED) as html_buffer:
        img_buffer = zipfile.ZipFile(image_zip, "w", zipfile.ZIP_DEFLATED) if KALEIDO_AVAILABLE else None
        
        try:
            for idx, file in enumerate(uploaded_files):
                filename = file.name
                status.info(f"Processing: {filename} ({idx+1}/{len(uploaded_files)})")
                
                try:
                    content = file.read().decode(errors='ignore')
                    wl, counts_raw = parse_asc_file(content, skip_rows)
                    
                    qs = extract_qs(filename)
                    thickness = extract_thickness(filename)
                    nd_value = extract_nd(filename) if apply_nd else 0.0
                    concentration = extract_concentration(filename)
                    dye_amount = extract_dye_amount(filename)
                    repetitions = extract_repetitions(filename)
                    sample_label = get_sample_label(thickness, concentration, dye_amount)
                    sample_label_short = get_short_label(thickness, concentration)
                    
                    if energy_map and not np.isnan(qs):
                        energy_mean, energy_std = interpolate_energy(qs, thickness, energy_map)
                    else:
                        energy_mean, energy_std = np.nan, np.nan
                    
                    if nd_value > 0:
                        counts_corrected = apply_nd_correction(counts_raw, nd_value)
                    else:
                        counts_corrected = counts_raw.copy()
                    
                    result = analyze_spectrum(wl, counts_corrected)
                    
                    # Gaussian-corrected uncertainties
                    total_peak_unc = calculate_total_uncertainty(
                        result.peak_intensity_uncertainty,
                        energy_std if not np.isnan(energy_std) else 0.0,
                        result.peak_intensity,
                        energy_mean if not np.isnan(energy_mean) else 1.0
                    )
                    total_integrated_unc = calculate_total_uncertainty(
                        result.integrated_intensity_uncertainty,
                        energy_std if not np.isnan(energy_std) else 0.0,
                        result.integrated_intensity,
                        energy_mean if not np.isnan(energy_mean) else 1.0
                    )
                    
                    peak_norm, peak_norm_unc = propagate_gaussian_error(
                        result.peak_intensity, result.peak_intensity_uncertainty,
                        energy_mean if not np.isnan(energy_mean) else 1.0,
                        energy_std if not np.isnan(energy_std) else 0.0,
                        normalization_mode
                    )
                    integrated_norm, integrated_norm_unc = propagate_gaussian_error(
                        result.integrated_intensity, result.integrated_intensity_uncertainty,
                        energy_mean if not np.isnan(energy_mean) else 1.0,
                        energy_std if not np.isnan(energy_std) else 0.0,
                        normalization_mode
                    )
                    
                    if show_individual:
                        fig = create_spectrum_plot(wl, counts_raw, counts_corrected, result, 
                                                  filename, nd_value, energy_mean, energy_std)
                        st.plotly_chart(fig, use_container_width=True, key=f"spec_{idx}")
                        
                        if show_fit_params and result.fit_success:
                            with st.expander(f"🔍 Parameters - {filename}"):
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Peak λ", f"{result.peak_wavelength:.2f}±{result.wavelength_uncertainty:.2f} nm")
                                c2.metric("FWHM", f"{result.fwhm:.2f}±{result.fwhm_uncertainty:.2f} nm")
                                c3.metric("R²", f"{result.r_squared:.4f}")
                                if not np.isnan(energy_mean):
                                    c4.metric("Energy", f"{energy_mean:.3f}±{energy_std:.3f} mJ")
                        
                        html = fig.to_html(full_html=False, include_plotlyjs='cdn').encode()
                        html_buffer.writestr(f"{filename.replace('.asc', '')}.html", html)
                    
                    label = f"QS={qs:.0f}" if not np.isnan(qs) else filename
                    if thickness:
                        label += f" | {thickness}mm"
                    if not np.isnan(energy_mean):
                        label += f" ({energy_mean:.3f}mJ)"
                    if nd_value > 0:
                        label += f" [OD{nd_value}]"
                    
                    combined_fig.add_trace(go.Scatter(x=wl, y=counts_corrected, mode='lines', name=label))
                    
                    energy_contrib_pct = (energy_std / energy_mean * 100) if (not np.isnan(energy_mean) and energy_mean > 0) else 0
                    rel_total_unc_pct = (total_peak_unc / result.peak_intensity * 100) if result.peak_intensity > 0 else 0
                    
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
                        "Wavelength Uncertainty": result.wavelength_uncertainty,
                        "Peak Intensity": result.peak_intensity,
                        "Peak Intensity Uncertainty": total_peak_unc,
                        "FWHM (nm)": result.fwhm,
                        "FWHM Uncertainty": result.fwhm_uncertainty,
                        "Integrated Intensity": result.integrated_intensity,
                        "Integrated Uncertainty": total_integrated_unc,
                        
                        "Peak Intensity (Normalized)": peak_norm,
                        "Peak Norm Uncertainty": peak_norm_unc,
                        "Integrated Intensity (Normalized)": integrated_norm,
                        "Integrated Norm Uncertainty": integrated_norm_unc,
                        
                        "Energy Contribution to Error (%)": energy_contrib_pct,
                        "Relative Total Uncertainty (%)": rel_total_unc_pct,
                        
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
    # RESULTS
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
    
    # Combined spectra
    st.markdown("---")
    st.subheader("🌈 Combined Spectra")
    combined_fig.update_layout(title="Spectral Evolution", xaxis_title="Wavelength (nm)",
                               yaxis_title="Intensity (OD-corrected)", template="plotly_white", height=600)
    st.plotly_chart(combined_fig, use_container_width=True, key="combined_spec")
    
    if KALEIDO_AVAILABLE:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            try:
                img = fig_to_image(combined_fig, image_format, image_width, image_height, image_scale)
                st.download_button(f"📥 Combined ({image_format.upper()})", img,
                                  f"combined.{image_format}", f"image/{image_format}")
            except:
                pass
    
    # Results table
    st.markdown("---")
    st.subheader("📋 Results Table")
    
    def highlight_r2(val):
        if pd.isna(val): return ''
        if val > 0.95: return 'background-color: #d4edda'
        if val > 0.85: return 'background-color: #fff3cd'
        return 'background-color: #f8d7da'
    
    styled = summary_df.style.applymap(highlight_r2, subset=['R²']).format({
        'Peak λ (nm)': '{:.2f}', 'Wavelength Uncertainty': '{:.2f}',
        'Peak Intensity': '{:.0f}', 'Peak Intensity Uncertainty': '{:.2f}',
        'FWHM (nm)': '{:.2f}', 'FWHM Uncertainty': '{:.2f}',
        'Integrated Intensity': '{:.2e}', 'Integrated Uncertainty': '{:.2e}',
        'R²': '{:.4f}', 'SNR': '{:.1f}',
        'QS Level': lambda x: f'{x:.0f}' if not pd.isna(x) else '',
        'Pump Energy (mJ)': lambda x: f'{x:.4f}' if not pd.isna(x) else '',
        'Energy Std (mJ)': lambda x: f'{x:.4f}' if not pd.isna(x) else '',
        'Relative Total Uncertainty (%)': lambda x: f'{x:.2f}' if not pd.isna(x) else '',
        'OD Filter': lambda x: f'{x:.1f}' if x > 0 else '',
        'Thickness (mm)': lambda x: f'{x:.1f}' if not pd.isna(x) else ''
    })
    
    st.dataframe(styled, use_container_width=True)
    
    # Threshold detection
    use_energy = 'Pump Energy (mJ)' in summary_df.columns and summary_df['Pump Energy (mJ)'].notna().sum() > 3
    
    if use_energy or summary_df['QS Level'].notna().sum() > 3:
        st.markdown("---")
        st.subheader("🎯 Threshold Detection")
        
        if use_energy:
            valid = summary_df.dropna(subset=['Pump Energy (mJ)', 'Integrated Intensity'])
            
            intensity_col = 'Integrated Intensity (Normalized)' if enable_normalization else 'Integrated Intensity'
            uncertainty_col = 'Integrated Norm Uncertainty' if enable_normalization else 'Integrated Uncertainty'
            
            if enable_weighted_threshold and not valid['Energy Std (mJ)'].isna().all():
                thr_res = weighted_least_squares_threshold(
                    valid['Pump Energy (mJ)'].values,
                    valid[intensity_col].values,
                    valid['Energy Std (mJ)'].values,
                    valid[uncertainty_col].values
                )
                threshold_obj = ThresholdAnalysis(
                    None, thr_res['threshold'],
                    thr_res['slope_below'], thr_res['slope_above'],
                    thr_res['threshold'] is not None,
                    thr_res['uncertainty']
                )
                st.info("✅ Gaussian-weighted threshold fitting enabled")
            else:
                threshold_obj = detect_threshold(
                    valid['Pump Energy (mJ)'].values,
                    valid[intensity_col].values
                )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if threshold_obj.threshold_found and threshold_obj.threshold_energy is not None:
                    if threshold_obj.threshold_uncertainty > 0:
                        st.success(f"✅ Threshold: {threshold_obj.threshold_energy:.4f} ± {threshold_obj.threshold_uncertainty:.4f} mJ")
                    else:
                        st.success(f"✅ Threshold: {threshold_obj.threshold_energy:.4f} mJ")
                else:
                    st.warning("⚠️ No threshold detected")
            with col2:
                st.metric("Slope (below)", f"{threshold_obj.slope_below:.2e}")
            with col3:
                st.metric("Slope (above)", f"{threshold_obj.slope_above:.2e}")
        else:
            valid = summary_df.dropna(subset=['QS Level', 'Integrated Intensity'])
            threshold_obj = detect_threshold(valid['QS Level'].values, valid['Integrated Intensity'].values)
            col1, col2, col3 = st.columns(3)
            with col1:
                if threshold_obj.threshold_found:
                    st.success(f"✅ Threshold: QS {threshold_obj.threshold_qs:.1f}")
                else:
                    st.warning("⚠️ No threshold detected")
            with col2:
                st.metric("Slope (below)", f"{threshold_obj.slope_below:.2e}")
            with col3:
                st.metric("Slope (above)", f"{threshold_obj.slope_above:.2e}")
        
        threshold_fig = create_threshold_plot(summary_df, threshold_obj, use_energy, show_error_bars)
        st.plotly_chart(threshold_fig, use_container_width=True, key="threshold_plot")
    
    # Energy vs Wavelength
    if 'Pump Energy (mJ)' in summary_df.columns and summary_df['Pump Energy (mJ)'].notna().sum() > 2:
        st.markdown("---")
        st.subheader("📈 Peak Wavelength vs Pump Energy")
        energy_wl_fig = create_energy_wavelength_plot(summary_df, show_error_bars)
        if energy_wl_fig:
            st.plotly_chart(energy_wl_fig, use_container_width=True, key="energy_wl")
            
            if KALEIDO_AVAILABLE:
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    try:
                        img = fig_to_image(energy_wl_fig, image_format, image_width, image_height, image_scale)
                        st.download_button(f"📥 Wavelength Plot ({image_format.upper()})", img,
                                          f"wavelength_vs_energy.{image_format}", f"image/{image_format}")
                    except:
                        pass
    
    # Energy vs Intensity
    if 'Pump Energy (mJ)' in summary_df.columns and summary_df['Pump Energy (mJ)'].notna().sum() > 2:
        st.markdown("---")
        st.subheader("💡 Peak Intensity vs Pump Energy")
        energy_int_fig = create_energy_intensity_plot(summary_df, show_error_bars)
        if energy_int_fig:
            st.plotly_chart(energy_int_fig, use_container_width=True, key="energy_int")
            
            if KALEIDO_AVAILABLE:
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    try:
                        img = fig_to_image(energy_int_fig, image_format, image_width, image_height, image_scale)
                        st.download_button(f"📥 Intensity Plot ({image_format.upper()})", img,
                                          f"intensity_vs_energy.{image_format}", f"image/{image_format}")
                    except:
                        pass
    
    # Downloads
    st.markdown("---")
    st.subheader("💾 Export All Results")
    
    cols = st.columns(4 if KALEIDO_AVAILABLE else 3)
    
    with cols[0]:
        csv = summary_df.to_csv(index=False).encode()
        st.download_button("📥 CSV", csv,
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv", use_container_width=True)
    
    with cols[1]:
        st.download_button("📦 HTML Plots", plot_zip.getvalue(),
            f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            "application/zip", use_container_width=True)
    
    with cols[2]:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Results', index=False)
            if energy_map:
                cal_data = []
                for thickness_key in energy_map.keys():
                    for qs, d in energy_map[thickness_key].items():
                        cal_data.append({
                            'Thickness': thickness_key if thickness_key is not None else 'N/A',
                            'QS': qs, 'Mean (mJ)': d['mean'],
                            'Std (mJ)': d['std'], 'N': d['n_readings']
                        })
                energy_cal_df = pd.DataFrame(cal_data).sort_values(['Thickness', 'QS'])
                energy_cal_df.to_excel(writer, sheet_name='Energy Cal', index=False)
        st.download_button("📊 Excel", excel_buffer.getvalue(),
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.ms-excel", use_container_width=True)
    
    if KALEIDO_AVAILABLE and image_zip:
        with cols[3]:
            st.download_button("🖼️ Images ZIP", image_zip.getvalue(),
                f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                "application/zip", use_container_width=True)

else:
    st.info("👆 Upload .asc files to begin")
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### How to Use This App
        
        1. **Upload Spectrum Files**
           - Upload your `.asc` spectrum files on the left
           - Files should contain wavelength and intensity data
           - Use the "Clear all" button to reset uploaded files
        
        2. **Energy Calibration (Optional)**
           - Paste your energy calibration data on the right
           - Supports simple format (QS levels only) or thickness-dependent format
        
        3. **Configure Settings**
           - Adjust header rows to skip in sidebar
           - Enable/disable OD correction
           - Toggle error bars and normalization
        
        4. **View Results**
           - Individual spectrum plots with Lorentzian fits
           - Combined spectra overlay
           - Threshold analysis dashboard
           - Peak wavelength vs energy (with vertical error bars)
           - Peak intensity vs energy (with vertical error bars)
        
        5. **Export**
           - Download CSV, Excel, or HTML plots
           - Export individual plot images (requires kaleido)
        
        ### File Naming Convention
        
        The app extracts metadata from filenames:
        - **QS Level**: `QS180`, `QS_200`, `qs-170`
        - **Thickness**: `UL3mm`, `5mm`, `thickness_7`
        - **Concentration**: `5%IL`, `UL10%IL`, `LL15%IL`
        - **OD Filter**: `OD1`, `OD=2`, `ND0.5`
        
        ### Error Bars
        
        All plots show **vertical error bars only** (standard convention):
        - **Wavelength plots**: ±σ_λ (from Lorentzian fit)
        - **Intensity plots**: ±σ_I (from Lorentzian fit + energy calibration)
        - **FWHM plots**: ±σ_FWHM (from Lorentzian fit)
        """)
    
    with st.expander("📊 Example Data Format"):
        st.markdown("""
        ### ASC File Format
        ```
        [Header lines - will be skipped]
        ...
        Wavelength    Intensity
        550.00        1234
        550.50        1456
        551.00        1678
        ...
        ```
        
        ### Energy Calibration Format (Simple)
        ```
        200    190    180    170    160
        0.008  0.025  0.058  0.122  0.245
        0.007  0.026  0.060  0.120  0.250
        0.009  0.024  0.055  0.125  0.240
        ```
        
        ### Energy Calibration Format (With Thickness)
        ```
        Thickness  200    190    180    170
        3          0.008  0.025  0.058  0.122
        5          0.010  0.030  0.070  0.150
        7          0.012  0.035  0.082  0.178
        ```
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 Random Laser Spectrum Analyzer</p>
    <p>📧 varun.solanki@fau.de | FAU Erlangen-Nürnberg</p>
    <p><small>Vertical error bars show measurement uncertainty (±σ)</small></p>
</div>
""", unsafe_allow_html=True)





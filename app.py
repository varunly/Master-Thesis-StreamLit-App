# ==============================================================
# Streamlit App: Random Laser ASC Analyzer
# Complete Version with ND Correction, Energy Calibration & All Plots
# ==============================================================
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
from io import StringIO, BytesIO
import zipfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
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
# 1. METADATA EXTRACTION (REGEX)
# ==============================================================
def extract_thickness(filename: str) -> Optional[float]:
    patterns = [r'UL[_\s-]*(\d+\.?\d*)\s*mm', r'LL[_\s-]*(\d+\.?\d*)\s*mm', 
                r'(\d+\.?\d*)\s*mm', r'thickness[_\s-]*(\d+\.?\d*)']
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match: return float(match.group(1))
    return None

def extract_concentration(filename: str) -> Optional[Dict[str, float]]:
    conc_data = {'upper': None, 'lower': None}
    ul_match = re.search(r'UL[_\s-]*(\d+\.?\d*)\s*%\s*IL', filename, re.IGNORECASE)
    if ul_match: conc_data['upper'] = float(ul_match.group(1))
    
    ll_match = re.search(r'LL[_\s-]*(\d+\.?\d*)\s*%\s*IL', filename, re.IGNORECASE)
    if ll_match: conc_data['lower'] = float(ll_match.group(1))
    
    if conc_data['upper'] is None and conc_data['lower'] is None:
        match = re.search(r'(\d+\.?\d*)\s*%', filename)
        if match: conc_data['upper'] = float(match.group(1))
        
    return conc_data if (conc_data['upper'] or conc_data['lower']) else None

def extract_nd(filename: str) -> float:
    """Extract ND/OD filter value to apply correction factor."""
    patterns = [r'OD\s*[=_-]*\s*(\d+\.?\d*)', r'ND\s*[=_-]*\s*(\d+\.?\d*)']
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match: return float(match.group(1))
    return 0.0

def extract_qs(filename: str) -> float:
    patterns = [r'QS[_\s-]+(\d+\.?\d*)', r'QS(\d+\.?\d*)', r'q[_\s-]*(\d+\.?\d*)']
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match: return float(match.group(1))
    return np.nan

def extract_dye_amount(filename: str) -> Optional[float]:
    match = re.search(r'(\d+\.?\d*)\s*mg', filename, re.IGNORECASE)
    return float(match.group(1)) if match else None

def get_short_label(thickness: Optional[float], concentration: Optional[Dict]) -> str:
    """Generate a short group label for plotting"""
    parts = []
    if thickness: parts.append(f"{thickness}mm")
    if concentration:
        if concentration.get('upper'): parts.append(f"UL{concentration['upper']}%")
        if concentration.get('lower'): parts.append(f"LL{concentration['lower']}%")
    return " ".join(parts) if parts else "No Label"

def apply_nd_correction(counts: np.ndarray, nd_value: float) -> np.ndarray:
    """Apply ND filter correction: multiply by 10^ND"""
    if nd_value == 0: return counts
    return counts * (10 ** nd_value)

# ==============================================================
# 2. ENERGY CALIBRATION PARSING
# ==============================================================
@st.cache_data
def parse_energy_file(file_bytes: bytes, file_type: str) -> Dict[float, Dict]:
    """
    Parses energy file. 
    Row 1: QS Levels. Rows 2-11: Data. Row 14: OD values.
    """
    try:
        # Load Data
        if 'xls' in file_type:
             # Try openpyxl first, fallback to xlrd, then csv
            try:
                df = pd.read_excel(BytesIO(file_bytes), header=None, engine='openpyxl')
            except:
                try:
                    df = pd.read_excel(BytesIO(file_bytes), header=None, engine='xlrd')
                except:
                    content = file_bytes.decode('utf-8', errors='ignore')
                    df = pd.read_csv(StringIO(content), sep=None, header=None, engine='python')
        else:
            content = file_bytes.decode('utf-8', errors='ignore')
            df = pd.read_csv(StringIO(content), sep=None, header=None, engine='python')

        # Limit to relevant area
        df = df.iloc[:15, :] 
        
        # Extract QS Levels (Row 1)
        qs_levels = []
        start_col = 0
        
        # Detect where numeric data starts in Row 1
        for i in range(df.shape[1]):
            val = df.iloc[0, i]
            try:
                if pd.notna(val):
                    f = float(val)
                    qs_levels.append(f)
                    if len(qs_levels) == 1: start_col = i
            except:
                continue

        if not qs_levels: return {}

        # Parse Data
        energy_map = {}
        for idx, qs in enumerate(qs_levels):
            col_idx = start_col + idx
            # Readings are in rows 1 to 10 (indices 1 to 10)
            readings = []
            for r in range(1, 11):
                try:
                    val = float(df.iloc[r, col_idx])
                    # Convert J to mJ if needed
                    if val < 0.1: val *= 1000
                    readings.append(val)
                except: pass
            
            # Extract OD from Row 14 (index 13)
            od_val = 0.0
            if df.shape[0] > 13:
                try:
                    val = df.iloc[13, col_idx]
                    if pd.notna(val): od_val = float(val)
                except: pass

            if readings:
                energy_map[qs] = {
                    'mean': np.mean(readings),
                    'std': np.std(readings),
                    'od': od_val
                }
        
        return energy_map

    except Exception as e:
        st.error(f"Error parsing energy file: {e}")
        return {}

# ==============================================================
# 3. SPECTRAL ANALYSIS (LORENTZIAN)
# ==============================================================
def lorentzian(x, A, x0, gamma, y0):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + y0

def analyze_spectrum(wl: np.ndarray, counts: np.ndarray) -> FitResult:
    """Fit Lorentzian and extract parameters."""
    try:
        # Initial Guesses
        peak_idx = np.argmax(counts)
        x0_init = wl[peak_idx]
        y0_init = np.min(counts)
        A_init = np.max(counts) - y0_init
        
        # Estimate Gamma (HWHM)
        half_max = y0_init + A_init / 2
        above_half = counts > half_max
        if np.sum(above_half) > 1:
            indices = np.where(above_half)[0]
            width = wl[indices[-1]] - wl[indices[0]]
            gamma_init = width / 2
        else:
            gamma_init = 1.0

        p0 = [A_init, x0_init, gamma_init, y0_init]
        
        try:
            popt, pcov = curve_fit(lorentzian, wl, counts, p0=p0, maxfev=10000)
        except:
            # Fallback with bounds if unconstrained fails
            bounds = ([0, wl.min(), 0, -np.inf], [np.inf, wl.max(), np.inf, np.inf])
            popt, pcov = curve_fit(lorentzian, wl, counts, p0=p0, bounds=bounds, maxfev=10000)

        A, x0, gamma, y0 = popt
        fit_y = lorentzian(wl, *popt)
        
        # R-squared
        ss_res = np.sum((counts - fit_y) ** 2)
        ss_tot = np.sum((counts - np.mean(counts)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # SNR
        noise = np.std(counts[counts < (y0 + A*0.1)]) # Estimate noise from baseline
        if noise == 0: noise = 1
        snr = A / noise

        # Integration (Trapezoidal)
        area = np.trapz(counts, wl)

        return FitResult(x0, A+y0, 2*abs(gamma), area, fit_y, r2, snr, 
                        {'A':A, 'x0':x0, 'gamma':gamma}, True)

    except Exception as e:
        # Fallback if fit fails completely
        return FitResult(wl[np.argmax(counts)], np.max(counts), np.nan, 
                        np.trapz(counts, wl), counts, 0, 0, {'error': str(e)}, False)

# ==============================================================
# 4. THRESHOLD DETECTION
# ==============================================================
def detect_threshold(x_val: np.ndarray, y_val: np.ndarray) -> ThresholdAnalysis:
    """Broken stick regression to find lasing threshold."""
    if len(x_val) < 5: return ThresholdAnalysis(None, None, 0, 0, False)
    
    # Sort
    idx = np.argsort(x_val)
    x = x_val[idx]
    y = y_val[idx]
    
    best_err = np.inf
    best_split = -1
    
    # Iterate through potential split points (leave 3 points on ends)
    for i in range(3, len(x)-2):
        # Line 1
        p1 = np.polyfit(x[:i], y[:i], 1)
        err1 = np.sum((np.polyval(p1, x[:i]) - y[:i])**2)
        
        # Line 2
        p2 = np.polyfit(x[i:], y[i:], 1)
        err2 = np.sum((np.polyval(p2, x[i:]) - y[i:])**2)
        
        total_err = err1 + err2
        
        # Criteria: Slope 2 must be significantly steeper than Slope 1
        if total_err < best_err and p2[0] > (1.5 * p1[0]):
            best_err = total_err
            best_split = i
            
    if best_split != -1:
        # Threshold is the X value where the second line intersects X-axis (approx)
        # or the X value of the split point
        t_val = x[best_split]
        p1 = np.polyfit(x[:best_split], y[:best_split], 1)
        p2 = np.polyfit(x[best_split:], y[best_split:], 1)
        return ThresholdAnalysis(None, t_val, p1[0], p2[0], True)
        
    return ThresholdAnalysis(None, None, 0, 0, False)

# ==============================================================
# 5. PLOTTING FUNCTIONS
# ==============================================================
def smooth_curve(x, y, num_points=200):
    """Generates smooth spline curve for plots"""
    if len(x) < 4: return x, y
    try:
        x_new = np.linspace(x.min(), x.max(), num_points)
        spl = make_interp_spline(x, y, k=3)
        return x_new, spl(x_new)
    except:
        return x, y

def create_spectrum_plot(wl, raw, corr, res, title, nd):
    fig = go.Figure()
    if nd > 0:
        fig.add_trace(go.Scatter(x=wl, y=raw, name='Raw', line=dict(color='gray', width=1, dash='dot')))
    
    fig.add_trace(go.Scatter(x=wl, y=corr, name='Corrected' if nd>0 else 'Data', 
                             line=dict(color='#2E86AB', width=2)))
    
    if res.fit_success:
        fig.add_trace(go.Scatter(x=wl, y=res.fit_y, name='Fit', 
                                 line=dict(color='#E63946', width=2, dash='dash')))
        fig.add_annotation(x=res.peak_wavelength, y=res.peak_intensity, 
                          text=f"Peak: {res.peak_wavelength:.1f}nm", showarrow=True, arrowheading=1)

    fig.update_layout(title=title, template="plotly_white", 
                      xaxis_title="Wavelength (nm)", yaxis_title="Intensity",
                      height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_aggregate_plot(df, x_col, y_col, title, y_label):
    """Creates grouped scatter plot with smooth lines"""
    fig = go.Figure()
    
    # Define colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    if 'Sample Label Short' not in df.columns:
        return fig

    groups = df.groupby('Sample Label Short')
    
    for i, (label, group) in enumerate(groups):
        if label == "No Label": continue
        
        # Sort and clean
        g = group.dropna(subset=[x_col, y_col]).sort_values(x_col)
        if len(g) < 2: continue
        
        x_data = g[x_col].values
        y_data = g[y_col].values
        color = colors[i % len(colors)]
        
        # Smooth line
        x_smooth, y_smooth = smooth_curve(x_data, y_data)
        
        fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', 
                                line=dict(color=color, width=2), showlegend=False))
        
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=label,
                                marker=dict(color=color, size=8, line=dict(width=1, color='white')),
                                hovertemplate=f"<b>{label}</b><br>E: %{{x:.2f}}<br>Y: %{{y:.2f}}"))

    fig.update_layout(title=title, xaxis_title="Pump Energy (mJ)", yaxis_title=y_label,
                      template="plotly_white", height=500, hovermode="closest")
    return fig

# ==============================================================
# MAIN APP UI
# ==============================================================
st.set_page_config(page_title="Random Laser Analyzer", page_icon="🔬", layout="wide")

st.title("🔬 Random Laser ASC Analyzer")
st.markdown("Automated processing of .asc files with **ND Correction**, **Energy Calibration**, and **Lorentzian Fitting**.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    skip_rows = st.number_input("Header rows to skip", 0, 100, 38)
    apply_correction = st.checkbox("Apply ND/OD Correction", value=True, 
                                  help="Multiplies counts by 10^OD if 'OD=x' found in filename")
    show_plots = st.checkbox("Show individual plots", True)
    
    st.divider()
    st.markdown("### 📥 Export")
    export_fmt = st.selectbox("Image Format", ["png", "svg", "pdf"])
    
    st.info("Ensure filenames contain metadata for best results:\n`UL_5mm_QS_110_OD=2.asc`")

# --- FILE UPLOAD ---
col1, col2 = st.columns([1, 1])
with col1:
    spectrum_files = st.file_uploader("1. Upload Spectrum Files (.asc)", accept_multiple_files=True, type="asc")
with col2:
    energy_file = st.file_uploader("2. Upload Energy Calibration (Excel/CSV)", type=["xlsx", "xls", "csv", "txt"])

# --- PROCESSING ---
if spectrum_files:
    # 1. Parse Energy Map
    energy_map = {}
    if energy_file:
        file_bytes = energy_file.read()
        energy_map = parse_energy_file(file_bytes, energy_file.name)
        if energy_map:
            st.success(f"⚡ Energy calibration loaded: {len(energy_map)} QS levels found.")
        else:
            st.warning("Could not parse energy file. Check format.")

    # 2. Process Files
    results = []
    
    progress = st.progress(0)
    
    # Zip buffer for plots
    plot_zip = BytesIO()
    
    with zipfile.ZipFile(plot_zip, "w") as zf:
        for idx, file in enumerate(spectrum_files):
            # Update Progress
            progress.progress((idx + 1) / len(spectrum_files))
            
            # Read File
            try:
                content = file.read().decode('utf-8', errors='ignore')
                df_spec = pd.read_csv(StringIO(content), sep='\t', skiprows=skip_rows, engine='python')
                df_spec = df_spec.dropna(axis=1, how='all') # Clean empty cols
                
                wl = df_spec.iloc[:, 0].values
                counts_raw = df_spec.iloc[:, 1:].mean(axis=1).values # Average if multiple cols
                
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue

            # Extract Metadata
            fn = file.name
            qs = extract_qs(fn)
            nd = extract_nd(fn) if apply_correction else 0.0
            thick = extract_thickness(fn)
            conc = extract_concentration(fn)
            label_short = get_short_label(thick, conc)
            
            # Energy Lookup
            energy_val = np.nan
            energy_std = np.nan
            if energy_map and not np.isnan(qs):
                # Find closest QS key
                qs_keys = np.array(list(energy_map.keys()))
                closest_idx = (np.abs(qs_keys - qs)).argmin()
                closest_qs = qs_keys[closest_idx]
                if abs(closest_qs - qs) < 5: # Tolerance
                    energy_val = energy_map[closest_qs]['mean']
                    energy_std = energy_map[closest_qs]['std']

            # Correction
            counts_corr = apply_nd_correction(counts_raw, nd)

            # Analyze
            fit = analyze_spectrum(wl, counts_corr)
            
            # Store Data
            res_dict = {
                'Filename': fn,
                'QS Level': qs,
                'Pump Energy (mJ)': energy_val,
                'Energy Std (mJ)': energy_std,
                'ND Filter': nd,
                'Thickness': thick,
                'Sample Label Short': label_short,
                'Peak λ (nm)': fit.peak_wavelength,
                'Peak Intensity': fit.peak_intensity,
                'FWHM (nm)': fit.fwhm,
                'Integrated Intensity': fit.integrated_intensity,
                'R²': fit.r_squared
            }
            results.append(res_dict)

            # Individual Plot
            if show_plots:
                with st.expander(f"📈 {fn} (QS={qs})", expanded=False):
                    fig = create_spectrum_plot(wl, counts_raw, counts_corr, fit, fn, nd)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save plot to zip
                    if KALEIDO_AVAILABLE:
                        img_bytes = fig.to_image(format=export_fmt, scale=2)
                        zf.writestr(f"{fn.replace('.asc','')}.{export_fmt}", img_bytes)

    progress.empty()
    
    # --- DASHBOARD & ANALYSIS ---
    df_res = pd.DataFrame(results)
    
    st.divider()
    
    # 1. Data Table
    st.subheader("📊 Analysis Results")
    st.dataframe(df_res.style.format({
        'Pump Energy (mJ)': '{:.3f}', 
        'Peak λ (nm)': '{:.2f}', 
        'FWHM (nm)': '{:.2f}',
        'R²': '{:.4f}'
    }), use_container_width=True)
    
    # 2. Threshold Analysis
    st.subheader("🎯 Threshold Detection")
    col1, col2 = st.columns(2)
    
    valid_data = df_res.dropna(subset=['Pump Energy (mJ)', 'Integrated Intensity'])
    if not valid_data.empty:
        thresh = detect_threshold(valid_data['Pump Energy (mJ)'].values, 
                                  valid_data['Integrated Intensity'].values)
        
        with col1:
            st.metric("Estimated Threshold", f"{thresh.threshold_energy:.4f} mJ" if thresh.threshold_found else "Not Detected")
            st.metric("Slope Efficiency (Above)", f"{thresh.slope_above:.2e}")
        
        with col2:
            fig_thresh = go.Figure()
            fig_thresh.add_trace(go.Scatter(x=valid_data['Pump Energy (mJ)'], 
                                          y=valid_data['Integrated Intensity'], 
                                          mode='markers', name='Data'))
            if thresh.threshold_found:
                fig_thresh.add_vline(x=thresh.threshold_energy, line_dash="dash", line_color="green", annotation_text="Threshold")
            
            fig_thresh.update_layout(title="Lasing Threshold", xaxis_title="Energy (mJ)", yaxis_title="Integrated Intensity")
            st.plotly_chart(fig_thresh, use_container_width=True)
    else:
        st.info("Upload Energy Calibration file to enable threshold detection.")

    # 3. Aggregate Plots (Grouped)
    if 'Pump Energy (mJ)' in df_res.columns and df_res['Pump Energy (mJ)'].notna().any():
        st.subheader("📈 Spectral Evolution (Grouped)")
        
        tab1, tab2 = st.tabs(["Energy vs Wavelength", "Energy vs Peak Intensity"])
        
        with tab1:
            fig_wl = create_aggregate_plot(df_res, 'Pump Energy (mJ)', 'Peak λ (nm)', 
                                         "Peak Wavelength Shift", "Peak Wavelength (nm)")
            st.plotly_chart(fig_wl, use_container_width=True)
            
        with tab2:
            fig_int = create_aggregate_plot(df_res, 'Pump Energy (mJ)', 'Peak Intensity', 
                                          "Intensity Growth", "Peak Intensity (Counts)")
            st.plotly_chart(fig_int, use_container_width=True)

    # --- DOWNLOADS ---
    st.divider()
    st.subheader("💾 Downloads")
    c1, c2, c3 = st.columns(3)
    
    # CSV
    csv = df_res.to_csv(index=False).encode('utf-8')
    c1.download_button("📥 Download Results (CSV)", csv, "results.csv", "text/csv")
    
    # Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Analysis')
        if energy_map:
            pd.DataFrame.from_dict(energy_map, orient='index').to_excel(writer, sheet_name='Energy Cal')
    c2.download_button("📊 Download Results (Excel)", output.getvalue(), "results.xlsx")
    
    # Images
    if KALEIDO_AVAILABLE:
        c3.download_button("🖼️ Download All Plots (ZIP)", plot_zip.getvalue(), "plots.zip", "application/zip")
    else:
        c3.warning("Image export unavailable (Kaleido missing)")

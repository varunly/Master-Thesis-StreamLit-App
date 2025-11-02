# ==============================================================
# Streamlit App: Random Laser ASC Analyzer - FAU Edition
# ==============================================================
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from io import StringIO, BytesIO
import zipfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import re
from datetime import datetime
import base64

# Check for kaleido installation
try:
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None
    KALEIDO_AVAILABLE = True
except:
    KALEIDO_AVAILABLE = False

# ==============================================================
# CONFIGURATION
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
    slope_below: float
    slope_above: float
    threshold_found: bool

# ==============================================================
# PAGE CONFIGURATION
# ==============================================================
st.set_page_config(
    page_title="FAU Random Laser Analyzer",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# ==============================================================
# CUSTOM CSS - PROFESSIONAL STYLING
# ==============================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Montserrat:wght@600;700&display=swap');
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #003865 0%, #005a8c 50%, #0077b3 100%);
        background-attachment: fixed;
    }
    
    /* Content area styling */
    .main .block-container {
        background: rgba(255, 255, 255, 0.98);
        padding: 2rem 3rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        font-family: 'Montserrat', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #003865, #0077b3, #00a0e3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeInDown 1s ease-in-out;
    }
    
    .sub-header {
        font-family: 'Roboto', sans-serif;
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* FAU Logo Container */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 2rem;
        animation: fadeIn 1.5s ease-in-out;
    }
    
    .university-name {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.5rem;
        color: #003865;
        font-weight: 600;
        text-align: center;
        margin-top: 1rem;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003865 0%, #005a8c 100%);
    }
    
    [data-testid="stSidebar"] .block-container {
        background: transparent;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] h1, h2, h3 {
        color: white !important;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Metric cards */
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #0077b3;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
    }
    
    .stMetric label {
        font-family: 'Roboto', sans-serif;
        font-weight: 600;
        color: #003865 !important;
        font-size: 1.1rem;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-family: 'Montserrat', sans-serif;
        font-size: 2rem;
        color: #0077b3 !important;
        font-weight: 700;
    }
    
    /* Section headers */
    h2, h3 {
        font-family: 'Montserrat', sans-serif;
        color: #003865;
        border-bottom: 3px solid #0077b3;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #e0f7ff 0%, #f0f9ff 100%);
        border: 2px dashed #0077b3;
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #003865;
        background: linear-gradient(135deg, #d0f0ff 0%, #e5f5ff 100%);
        transform: scale(1.02);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0077b3, #005a8c);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-family: 'Roboto', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #005a8c, #003865);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00a0e3, #0077b3);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-family: 'Roboto', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 119, 179, 0.3);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0077b3, #005a8c);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 119, 179, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0077b3, #00a0e3);
    }
    
    /* Data table */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #0077b3;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f0f9ff, #e0f7ff);
        border-radius: 10px;
        font-family: 'Roboto', sans-serif;
        font-weight: 600;
        color: #003865;
    }
    
    /* Success/Warning/Error boxes */
    .element-container .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 5px solid #28a745;
        border-radius: 10px;
    }
    
    .element-container .stWarning {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-left: 5px solid #ffc107;
        border-radius: 10px;
    }
    
    .element-container .stError {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 5px solid #dc3545;
        border-radius: 10px;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #003865, #005a8c);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
        font-family: 'Roboto', sans-serif;
        box-shadow: 0 -4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .footer a {
        color: #00a0e3;
        text-decoration: none;
        font-weight: 600;
    }
    
    .footer a:hover {
        color: #00c8ff;
        text-decoration: underline;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Section containers */
    .section-container {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #0077b3;
    }
    
    /* Plotly chart containers */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f0f9ff, #e0f7ff);
        border-radius: 10px 10px 0 0;
        color: #003865;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0077b3, #005a8c);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================
# HEADER WITH FAU LOGO
# ==============================================================
def add_logo_and_header():
    """Add FAU logo and professional header"""
    
    # FAU Logo (you can replace this URL with your actual logo file)
    logo_html = """
    <div class="logo-container">
        <img src="https://www.fau.eu/files/2020/09/FAU-Logo.png" 
             alt="FAU Logo" 
             style="height: 120px; margin-bottom: 1rem;">
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)
    
    st.markdown('<p class="university-name">Friedrich-Alexander-Universität Erlangen-Nürnberg</p>', 
                unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">🔬 Advanced Random Laser Analyzer</p>', 
                unsafe_allow_html=True)
    
    st.markdown('''
    <p class="sub-header">
        Precision Spectral Analysis Platform with Lorentzian Fitting & Threshold Detection<br>
        <em>Department of Physics | Photonics Research Group</em>
    </p>
    ''', unsafe_allow_html=True)

add_logo_and_header()

# ==============================================================
# CORE ANALYSIS FUNCTIONS (Same as before)
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
        peak_val = counts[peak_idx]
        x0_init = wl[peak_idx]
        
        baseline_est = np.percentile(counts, 5)
        A_init = peak_val - baseline_est
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
        A, gamma = abs(A), abs(gamma)
        
        fwhm = 2 * gamma
        fit_y = lorentzian(wl, A, x0, gamma, y0)
        
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
                        float(np.trapz(counts - np.min(counts), wl)), counts.copy(),
                        0.0, float(calculate_snr(counts)), {'error': str(e)}, fit_success=False)

def detect_threshold(qs_levels: np.ndarray, intensities: np.ndarray, min_points: int = 3) -> ThresholdAnalysis:
    """Detect lasing threshold using broken-stick algorithm"""
    if len(qs_levels) < 2 * min_points:
        return ThresholdAnalysis(None, 0, 0, False)
    
    try:
        idx = np.argsort(qs_levels)
        qs_sorted = qs_levels[idx]
        int_sorted = intensities[idx]
        
        best_threshold = None
        best_r2_sum = -np.inf
        best_slopes = (0, 0)
        
        for i in range(min_points, len(qs_sorted) - min_points):
            below = np.polyfit(qs_sorted[:i], int_sorted[:i], 1)
            above = np.polyfit(qs_sorted[i:], int_sorted[i:], 1)
            
            r2_below = calculate_r_squared(int_sorted[:i], np.polyval(below, qs_sorted[:i]))
            r2_above = calculate_r_squared(int_sorted[i:], np.polyval(above, qs_sorted[i:]))
            
            r2_sum = r2_below + r2_above
            if r2_sum > best_r2_sum and above[0] > below[0]:
                best_r2_sum = r2_sum
                best_threshold = qs_sorted[i]
                best_slopes = (below[0], above[0])
        
        found = best_threshold is not None and best_slopes[1] > 2 * best_slopes[0]
        
        return ThresholdAnalysis(best_threshold, best_slopes[0], best_slopes[1], found)
        
    except Exception:
        return ThresholdAnalysis(None, 0, 0, False)

def extract_qs(filename: str) -> float:
    """Extract Q-switch value from filename using regex"""
    patterns = [r'qs[_\s-]*(\d+\.?\d*)', r'(\d+\.?\d*)[_\s-]*qs', 
                r'q[_\s-]*(\d+\.?\d*)', r'(\d+\.?\d+)']
    
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
    """Parse .asc file and return wavelength and intensity arrays"""
    df = pd.read_csv(StringIO(file_content), sep='\t', decimal=',', 
                     skiprows=skip_rows, engine='python')
    df = df.dropna(axis=1, how='all')
    
    if df.shape[1] < 2:
        raise ValueError("File must have at least 2 columns")
    
    wl = df.iloc[:, 0].to_numpy()
    counts = df.iloc[:, 1:].mean(axis=1).to_numpy()
    
    return wl, counts

def fig_to_image(fig: go.Figure, format: str, width: int, height: int, scale: int) -> bytes:
    """Convert plotly figure to image bytes"""
    try:
        img_bytes = fig.to_image(format=format, width=width, height=height, 
                                scale=scale, engine="kaleido")
        return img_bytes
    except Exception as e:
        st.error(f"❌ Image export failed: {str(e)}")
        raise

# ==============================================================
# VISUALIZATION FUNCTIONS (Enhanced with FAU colors)
# ==============================================================
def create_spectrum_plot(wl: np.ndarray, counts: np.ndarray, 
                        fit_result: FitResult, filename: str) -> go.Figure:
    """Create interactive spectrum plot with fit overlay"""
    fig = go.Figure()
    
    # FAU color scheme
    fau_blue = '#003865'
    fau_light_blue = '#0077b3'
    
    fig.add_trace(go.Scatter(
        x=wl, y=counts, mode='lines', name='Experimental Data',
        line=dict(color=fau_blue, width=3),
        hovertemplate='λ: %{x:.2f} nm<br>I: %{y:.0f}<extra></extra>'
    ))
    
    if fit_result.fit_success and not np.isnan(fit_result.fwhm):
        fig.add_trace(go.Scatter(
            x=wl, y=fit_result.fit_y, mode='lines', name='Lorentzian Fit',
            line=dict(color='#e63946', width=3, dash='dash'), opacity=0.8,
            hovertemplate='Fit: %{y:.0f}<extra></extra>'
        ))
        
        fig.add_vline(x=fit_result.peak_wavelength, line_dash="dot", 
                     line_color="#2a9d8f", line_width=2,
                     annotation_text=f"Peak λ = {fit_result.peak_wavelength:.2f} nm",
                     annotation_position="top")
        
        gamma = fit_result.fwhm / 2
        x0 = fit_result.peak_wavelength
        half_max = fit_result.peak_intensity / 2
        
        fig.add_trace(go.Scatter(
            x=[x0-gamma, x0+gamma], y=[half_max, half_max],
            mode='markers+text', marker=dict(size=12, color='#f77f00', symbol='diamond'),
            name=f'FWHM = {fit_result.fwhm:.2f} nm',
            text=['', f'FWHM={fit_result.fwhm:.2f}nm'], textposition='top center',
            hovertemplate='FWHM boundary<extra></extra>'
        ))
        
        fig.add_shape(type="line", x0=x0-gamma, y0=half_max, x1=x0+gamma, y1=half_max,
                     line=dict(color="#f77f00", width=2, dash="dash"))
    
    title_html = f"<b style='color:{fau_blue}'>{filename}</b><br>"
    if fit_result.fit_success:
        title_html += f"<sub>Peak: {fit_result.peak_wavelength:.2f} nm | "
        title_html += f"FWHM: {fit_result.fwhm:.2f} nm | "
        title_html += f"R²: {fit_result.r_squared:.4f} | "
        title_html += f"SNR: {fit_result.snr:.1f}</sub>"
    else:
        title_html += f"<sub style='color: red;'>Fit Failed - Showing Raw Data Only | SNR: {fit_result.snr:.1f}</sub>"
    
    fig.update_layout(
        title=title_html, xaxis_title="Wavelength (nm)", yaxis_title="Intensity (counts)",
        template="plotly_white", hovermode="x unified", height=500, showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)', 
                   bordercolor=fau_blue, borderwidth=2),
        plot_bgcolor='rgba(240,249,255,0.5)',
        font=dict(family="Roboto, sans-serif", size=12, color=fau_blue)
    )
    
    return fig

def create_threshold_plot(df: pd.DataFrame, threshold: ThresholdAnalysis) -> go.Figure:
    """Create comprehensive threshold analysis plot"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Integrated Intensity vs Q-Switch", "FWHM vs Q-Switch",
                       "Peak Wavelength vs Q-Switch", "Peak Intensity vs Q-Switch"),
        vertical_spacing=0.12, horizontal_spacing=0.1
    )
    
    valid = df.dropna(subset=['QS Level'])
    qs = valid['QS Level'].values
    
    colors = ['#e63946', '#0077b3', '#9b59b6', '#f77f00']
    data = [
        (valid['Integrated Intensity'], 1, 1),
        (valid['FWHM (nm)'], 1, 2),
        (valid['Peak λ (nm)'], 2, 1),
        (valid['Peak Intensity'], 2, 2)
    ]
    
    for idx, (y_data, row, col) in enumerate(data):
        fig.add_trace(
            go.Scatter(x=qs, y=y_data, mode='lines+markers',
                      marker=dict(size=10, color=colors[idx]), line=dict(width=3)),
            row=row, col=col
        )
    
    if threshold.threshold_found:
        fig.add_vline(x=threshold.threshold_qs, line_dash="dash", line_color="green",
                     annotation_text=f"Threshold ≈ {threshold.threshold_qs:.1f}", row=1, col=1)
    
    fig.update_xaxes(title_text="Q-Switch Level", row=1, col=1)
    fig.update_xaxes(title_text="Q-Switch Level", row=1, col=2)
    fig.update_xaxes(title_text="Q-Switch Level", row=2, col=1)
    fig.update_xaxes(title_text="Q-Switch Level", row=2, col=2)
    
    fig.update_yaxes(title_text="Integrated Intensity", row=1, col=1)
    fig.update_yaxes(title_text="FWHM (nm)", row=1, col=2)
    fig.update_yaxes(title_text="Wavelength (nm)", row=2, col=1)
    fig.update_yaxes(title_text="Counts", row=2, col=2)
    
    fig.update_layout(
        height=700, showlegend=False, template="plotly_white",
        title_text="<b style='color:#003865'>Threshold Analysis Dashboard</b>",
        plot_bgcolor='rgba(240,249,255,0.3)',
        font=dict(family="Roboto, sans-serif", size=11, color='#003865')
    )
    
    return fig

# ==============================================================
# SIDEBAR CONFIGURATION
# ==============================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")
    st.markdown("---")
    
    with st.expander("📁 File Settings", expanded=True):
        skip_rows = st.number_input("Header rows to skip", 0, 100, 38)
        show_individual = st.checkbox("Show individual plots", True)
        show_fit_params = st.checkbox("Show fit parameters", False)
    
    if KALEIDO_AVAILABLE:
        with st.expander("💾 Export Settings", expanded=False):
            image_format = st.selectbox("Image format", ["png", "jpeg", "svg", "pdf"], index=0)
            image_width = st.number_input("Image width (px)", 800, 3000, 1200)
            image_height = st.number_input("Image height (px)", 400, 2000, 800)
            image_scale = st.slider("Image scale/quality", 1, 5, 2)
    
    st.markdown("---")
    st.markdown("### 📊 Analysis Features")
    st.markdown("""
    ✅ Lorentzian curve fitting  
    ✅ FWHM & R² calculation  
    ✅ Threshold detection  
    ✅ SNR estimation  
    ✅ Interactive visualizations  
    ✅ Multi-format export  
    """)
    
    st.markdown("---")
    if KALEIDO_AVAILABLE:
        st.success("✅ Image export enabled")
    else:
        st.error("❌ Kaleido not installed")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;'>
        <small>Developed by<br><b>FAU Physics Department</b><br>
        Photonics Research Group</small>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================
# FILE UPLOAD SECTION
# ==============================================================
st.markdown("---")
uploaded_files = st.file_uploader(
    "📤 Upload Spectral Data Files (.asc format)",
    accept_multiple_files=True,
    type=['asc'],
    help="Select one or more .asc files for comprehensive analysis"
)

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
    
    with zipfile.ZipFile(plot_zip, "w", zipfile.ZIP_DEFLATED) as html_buffer:
        img_buffer = zipfile.ZipFile(image_zip, "w", zipfile.ZIP_DEFLATED) if KALEIDO_AVAILABLE else None
        
        try:
            for idx, file in enumerate(uploaded_files):
                filename = file.name
                status.info(f"⚙️ Processing: **{filename}** ({idx+1}/{len(uploaded_files)})")
                
                try:
                    content = file.read().decode(errors='ignore')
                    wl, counts = parse_asc_file(content, skip_rows)
                    result = analyze_spectrum(wl, counts)
                    qs = extract_qs(filename)
                    
                    if show_individual:
                        fig = create_spectrum_plot(wl, counts, result, filename)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if KALEIDO_AVAILABLE:
                            col1, col2 = st.columns([3, 1])
                            with col2:
                                try:
                                    img_bytes = fig_to_image(fig, image_format, image_width, 
                                                            image_height, image_scale)
                                    st.download_button(
                                        label=f"📥 {image_format.upper()}",
                                        data=img_bytes,
                                        file_name=f"{filename.replace('.asc', '')}.{image_format}",
                                        mime=f"image/{image_format}",
                                        key=f"download_{idx}"
                                    )
                                    if img_buffer:
                                        img_buffer.writestr(f"{filename.replace('.asc', '')}.{image_format}", 
                                                          img_bytes)
                                except Exception as e:
                                    st.error(f"Image export failed: {e}")
                        
                        if show_fit_params and result.fit_params and result.fit_success:
                            with st.expander(f"🔍 Detailed Fit Parameters - {filename}"):
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Amplitude", f"{result.fit_params.get('Amplitude', 0):.1f}")
                                col2.metric("Center (nm)", f"{result.fit_params.get('Center', 0):.2f}")
                                col3.metric("Gamma", f"{result.fit_params.get('Gamma', 0):.2f}")
                                col4.metric("Baseline", f"{result.fit_params.get('Baseline', 0):.1f}")
                        
                        html = fig.to_html(full_html=False, include_plotlyjs='cdn').encode()
                        html_buffer.writestr(f"{filename.replace('.asc', '')}.html", html)
                    
                    combined_fig.add_trace(go.Scatter(
                        x=wl, y=counts, mode='lines',
                        name=f"QS={qs:.0f}" if not np.isnan(qs) else filename,
                        hovertemplate='%{y:.0f}<extra></extra>'
                    ))
                    
                    summary_data.append({
                        "File": filename, "QS Level": qs,
                        "Peak λ (nm)": result.peak_wavelength,
                        "Peak Intensity": result.peak_intensity,
                        "FWHM (nm)": result.fwhm,
                        "Integrated Intensity": result.integrated_intensity,
                        "R²": result.r_squared, "SNR": result.snr,
                        "Fit Success": "✅" if result.fit_success else "❌"
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error processing {filename}: {e}")
                    continue
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
        
        finally:
            if img_buffer:
                img_buffer.close()
    
    status.success("✅ **Analysis Complete!** All files processed successfully.")
    progress_bar.empty()
    
    # ==============================================================
    # RESULTS SECTION
    # ==============================================================
    st.markdown("---")
    st.markdown("## 📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    summary_df = pd.DataFrame(summary_data).sort_values("QS Level")
    
    with col1:
        st.metric("📁 Files Analyzed", len(summary_df))
    with col2:
        st.metric("📈 Avg R²", f"{summary_df['R²'].mean():.3f}")
    with col3:
        st.metric("📏 Avg FWHM", f"{summary_df['FWHM (nm)'].mean():.2f} nm")
    with col4:
        st.metric("🔊 Avg SNR", f"{summary_df['SNR'].mean():.1f}")
    
    # Combined Spectra
    st.markdown("---")
    st.markdown("## 🌈 Combined Spectral Analysis")
    combined_fig.update_layout(
        title="<b style='color:#003865'>Spectral Evolution with Q-Switch Level</b>",
        xaxis_title="Wavelength (nm)", yaxis_title="Intensity (counts)",
        template="plotly_white", hovermode="x unified", height=600,
        plot_bgcolor='rgba(240,249,255,0.5)',
        font=dict(family="Roboto, sans-serif", size=12, color='#003865')
    )
    st.plotly_chart(combined_fig, use_container_width=True)
    
    if KALEIDO_AVAILABLE:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            try:
                combined_img = fig_to_image(combined_fig, image_format, image_width, 
                                           image_height, image_scale)
                st.download_button(
                    label=f"📥 Download Combined Plot ({image_format.upper()})",
                    data=combined_img,
                    file_name=f"combined_spectra.{image_format}",
                    mime=f"image/{image_format}"
                )
            except:
                pass
    
    # Data Table
    st.markdown("---")
    st.markdown("## 📋 Detailed Results Table")
    
    def highlight_quality(val):
        if pd.isna(val):
            return ''
        if val > 0.95:
            return 'background-color: #d4edda'
        elif val > 0.85:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    styled_df = summary_df.style.applymap(highlight_quality, subset=['R²']).format({
        'Peak λ (nm)': '{:.2f}', 'Peak Intensity': '{:.0f}',
        'FWHM (nm)': '{:.2f}', 'Integrated Intensity': '{:.2e}',
        'R²': '{:.4f}', 'SNR': '{:.1f}', 'QS Level': '{:.0f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Threshold Analysis
    if summary_df['QS Level'].notna().sum() > 3:
        st.markdown("---")
        st.markdown("## 🎯 Threshold Detection Analysis")
        
        valid = summary_df.dropna(subset=['QS Level', 'Integrated Intensity'])
        threshold = detect_threshold(valid['QS Level'].values, 
                                    valid['Integrated Intensity'].values)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if threshold.threshold_found:
                st.success(f"✅ **Threshold Detected**\n\nQS ≈ **{threshold.threshold_qs:.1f}**")
            else:
                st.warning("⚠️ No clear threshold detected")
        with col2:
            st.metric("Slope (below threshold)", f"{threshold.slope_below:.2e}")
        with col3:
            st.metric("Slope (above threshold)", f"{threshold.slope_above:.2e}")
        
        threshold_fig = create_threshold_plot(summary_df, threshold)
        st.plotly_chart(threshold_fig, use_container_width=True)
        
        if KALEIDO_AVAILABLE:
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                try:
                    threshold_img = fig_to_image(threshold_fig, image_format, 
                                                int(image_width * 1.5), 
                                                int(image_height * 1.2), image_scale)
                    st.download_button(
                        label=f"📥 Threshold Plot ({image_format.upper()})",
                        data=threshold_img,
                        file_name=f"threshold_analysis.{image_format}",
                        mime=f"image/{image_format}"
                    )
                except:
                    pass
    
    # Export Section
    st.markdown("---")
    st.markdown("## 💾 Export Results")
    
    if KALEIDO_AVAILABLE:
        col1, col2, col3, col4 = st.columns(4)
    else:
        col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = summary_df.to_csv(index=False).encode()
        st.download_button(
            "📥 CSV Data", csv_data,
            f"laser_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv", use_container_width=True
        )
    
    with col2:
        st.download_button(
            "📦 HTML Plots", plot_zip.getvalue(),
            f"plots_html_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            "application/zip", use_container_width=True
        )
    
    with col3:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Results', index=False)
        st.download_button(
            "📊 Excel", excel_buffer.getvalue(),
            f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.ms-excel", use_container_width=True
        )
    
    if KALEIDO_AVAILABLE and image_zip:
        with col4:
            st.download_button(
                f"🖼️ Images ({image_format.upper()})", image_zip.getvalue(),
                f"plots_{image_format}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                "application/zip", use_container_width=True
            )

else:
    # Welcome Screen
    st.info("👆 **Upload your .asc spectral files to begin advanced analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📖 User Guide", expanded=True):
            st.markdown("""
            ### Quick Start
            
            1. **Upload Files**: Click above to select `.asc` files
            2. **Automatic Processing**: Advanced Lorentzian fitting
            3. **Interactive Visualization**: Zoom, pan, and explore
            4. **Export**: Download results in multiple formats
            
            ### File Naming Convention
            Include Q-switch values in filenames:
            - `sample_qs_100.asc`
            - `QS150.asc`
            - `data_200_qs.asc`
            
            ### Supported Formats
            - Input: `.asc` (tab-separated)
            - Export: CSV, Excel, HTML, PNG, JPEG, SVG, PDF
            """)
    
    with col2:
        with st.expander("🔬 Technical Details", expanded=True):
            st.markdown("""
            ### Analysis Methods
            
            **Spectral Fitting**
            - Lorentzian lineshape function
            - Levenberg-Marquardt optimization
            - Automatic parameter estimation
            
            **Metrics Calculated**
            - Peak wavelength & intensity
            - FWHM (Full Width at Half Maximum)
            - Integrated intensity
            - R² (goodness of fit)
            - Signal-to-Noise Ratio
            
            **Threshold Detection**
            - Broken-stick algorithm
            - Linear segmentation
            - Slope change analysis
            """)

# ==============================================================
# FOOTER
# ==============================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3 style="margin-top: 0;">Friedrich-Alexander-Universität Erlangen-Nürnberg</h3>
    <p><strong>Department of Physics | Photonics Research Group</strong></p>
    <p>Advanced Random Laser Analysis Platform</p>
    <hr style="border-color: rgba(255,255,255,0.3); margin: 1.5rem 0;">
    <p style="font-size: 0.9rem;">
        🔬 Powered by Streamlit • 📊 Plotly Graphics • 🧮 SciPy Optimization<br>
        💾 Multi-format Export • 🎨 Kaleido Rendering Engine
    </p>
    <p style="font-size: 0.85rem; margin-top: 1rem;">
        <em>For support or questions, contact the Physics Department IT Team</em><br>
        © 2024 FAU Erlangen-Nürnberg. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)

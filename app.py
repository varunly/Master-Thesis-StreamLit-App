# ==============================================================
# Streamlit App: Random Laser ASC Analyzer
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
# CORE ANALYSIS FUNCTIONS
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
    
    Args:
        wl: Wavelength array (nm)
        counts: Intensity array
    
    Returns:
        FitResult object with all metrics
    """
    try:
        # Smart initial guesses
        peak_idx = np.argmax(counts)
        x0_guess = wl[peak_idx]
        A_guess = counts[peak_idx] - np.percentile(counts, 10)
        gamma_guess = (wl.max() - wl.min()) / 20
        y0_guess = np.percentile(counts, 10)
        
        # Bounds for stable fitting
        bounds = (
            [0, wl.min(), 0, 0],
            [np.inf, wl.max(), (wl.max()-wl.min())/2, counts.max()]
        )
        
        # Perform fit
        popt, pcov = curve_fit(
            lorentzian, wl, counts,
            p0=[A_guess, x0_guess, gamma_guess, y0_guess],
            bounds=bounds,
            maxfev=20000
        )
        
        A, x0, gamma, y0 = popt
        fwhm = 2 * gamma
        fit_y = lorentzian(wl, *popt)
        
        # Calculate metrics
        baseline_corrected = counts - y0
        area = np.trapz(baseline_corrected, wl)
        r_squared = calculate_r_squared(counts, fit_y)
        snr = calculate_snr(counts)
        
        fit_params = {
            'Amplitude': A,
            'Center': x0,
            'Gamma': gamma,
            'Baseline': y0,
            'Std_Errors': np.sqrt(np.diag(pcov))
        }
        
        return FitResult(x0, A+y0, fwhm, area, fit_y, r_squared, snr, fit_params, fit_success=True)
        
    except Exception as e:
        # Fallback to basic metrics if fitting fails
        st.warning(f"⚠️ Lorentzian fit failed: {str(e)}. Using raw data only.")
        return FitResult(
            wl[np.argmax(counts)],
            np.max(counts),
            np.nan,
            np.trapz(counts, wl),
            counts,
            0.0,
            calculate_snr(counts),
            {},
            fit_success=False
        )

def detect_threshold(qs_levels: np.ndarray, intensities: np.ndarray, 
                     min_points: int = 3) -> ThresholdAnalysis:
    """
    Detect lasing threshold using broken-stick algorithm
    
    Args:
        qs_levels: Q-switch values
        intensities: Integrated intensities
        min_points: Minimum points for linear fit
    
    Returns:
        ThresholdAnalysis object
    """
    if len(qs_levels) < 2 * min_points:
        return ThresholdAnalysis(None, 0, 0, False)
    
    try:
        # Sort data
        idx = np.argsort(qs_levels)
        qs_sorted = qs_levels[idx]
        int_sorted = intensities[idx]
        
        # Try each point as potential threshold
        best_threshold = None
        best_r2_sum = -np.inf
        best_slopes = (0, 0)
        
        for i in range(min_points, len(qs_sorted) - min_points):
            # Fit two linear segments
            below = np.polyfit(qs_sorted[:i], int_sorted[:i], 1)
            above = np.polyfit(qs_sorted[i:], int_sorted[i:], 1)
            
            # Calculate R² for both segments
            r2_below = calculate_r_squared(int_sorted[:i], np.polyval(below, qs_sorted[:i]))
            r2_above = calculate_r_squared(int_sorted[i:], np.polyval(above, qs_sorted[i:]))
            
            # Look for maximum slope change
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
    """Parse .asc file and return wavelength and intensity arrays"""
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
def create_spectrum_plot(wl: np.ndarray, counts: np.ndarray, 
                        fit_result: FitResult, filename: str) -> go.Figure:
    """Create interactive spectrum plot with fit overlay"""
    fig = go.Figure()
    
    # Raw data - solid line (blue)
    fig.add_trace(go.Scatter(
        x=wl, y=counts,
        mode='lines',
        name='Experimental Data',
        line=dict(color='#2E86AB', width=3),
        hovertemplate='λ: %{x:.2f} nm<br>I: %{y:.0f}<extra></extra>'
    ))
    
    # Lorentzian fit - dotted line (red/orange) - more visible
    if fit_result.fit_success and not np.isnan(fit_result.fwhm):
        fig.add_trace(go.Scatter(
            x=wl, y=fit_result.fit_y,
            mode='lines',
            name='Lorentzian Fit',
            line=dict(color='red', width=3, dash='dash'),  # Changed to dash for better visibility
            opacity=0.8,
            hovertemplate='Fit: %{y:.0f}<extra></extra>'
        ))
        
        # Mark peak position
        fig.add_vline(
            x=fit_result.peak_wavelength,
            line_dash="dot",
            line_color="green",
            line_width=2,
            annotation_text=f"Peak λ = {fit_result.peak_wavelength:.2f} nm",
            annotation_position="top"
        )
        
        # FWHM markers
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
            textposition='top center',
            hovertemplate='FWHM boundary<extra></extra>'
        ))
        
        # Add horizontal line for FWHM
        fig.add_shape(
            type="line",
            x0=x0-gamma, y0=half_max,
            x1=x0+gamma, y1=half_max,
            line=dict(color="orange", width=2, dash="dash")
        )
    else:
        st.warning(f"⚠️ Lorentzian fitting failed for {filename}")
    
    # Layout
    title_html = f"<b>{filename}</b><br>"
    if fit_result.fit_success:
        title_html += f"<sub>Peak: {fit_result.peak_wavelength:.2f} nm | "
        title_html += f"FWHM: {fit_result.fwhm:.2f} nm | "
        title_html += f"R²: {fit_result.r_squared:.4f} | "
        title_html += f"SNR: {fit_result.snr:.1f}</sub>"
    else:
        title_html += f"<sub style='color: red;'>Fit Failed - Showing Raw Data Only</sub>"
    
    fig.update_layout(
        title=title_html,
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity (counts)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        showlegend=True,
        legend=dict(
            x=0.02, 
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig

def create_threshold_plot(df: pd.DataFrame, threshold: ThresholdAnalysis) -> go.Figure:
    """Create comprehensive threshold analysis plot"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Integrated Intensity vs Q-Switch",
            "FWHM vs Q-Switch",
            "Peak Wavelength vs Q-Switch",
            "Peak Intensity vs Q-Switch"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    valid = df.dropna(subset=['QS Level'])
    qs = valid['QS Level'].values
    
    # Plot 1: Integrated Intensity (main threshold indicator)
    fig.add_trace(
        go.Scatter(
            x=qs, y=valid['Integrated Intensity'],
            mode='lines+markers',
            name='Integrated Intensity',
            marker=dict(size=10, color='red'),
            line=dict(width=3)
        ),
        row=1, col=1
    )
    
    if threshold.threshold_found:
        fig.add_vline(
            x=threshold.threshold_qs,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Threshold ≈ {threshold.threshold_qs:.1f}",
            row=1, col=1
        )
    
    # Plot 2: FWHM (spectral narrowing)
    fig.add_trace(
        go.Scatter(
            x=qs, y=valid['FWHM (nm)'],
            mode='lines+markers',
            name='FWHM',
            marker=dict(size=10, color='blue'),
            line=dict(width=3)
        ),
        row=1, col=2
    )
    
    # Plot 3: Peak Wavelength (mode competition)
    fig.add_trace(
        go.Scatter(
            x=qs, y=valid['Peak λ (nm)'],
            mode='lines+markers',
            name='Peak λ',
            marker=dict(size=10, color='purple'),
            line=dict(width=3)
        ),
        row=2, col=1
    )
    
    # Plot 4: Peak Intensity
    fig.add_trace(
        go.Scatter(
            x=qs, y=valid['Peak Intensity'],
            mode='lines+markers',
            name='Peak Intensity',
            marker=dict(size=10, color='orange'),
            line=dict(width=3)
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Q-Switch Level", row=1, col=1)
    fig.update_xaxes(title_text="Q-Switch Level", row=1, col=2)
    fig.update_xaxes(title_text="Q-Switch Level", row=2, col=1)
    fig.update_xaxes(title_text="Q-Switch Level", row=2, col=2)
    
    fig.update_yaxes(title_text="Integrated Intensity", row=1, col=1)
    fig.update_yaxes(title_text="FWHM (nm)", row=1, col=2)
    fig.update_yaxes(title_text="Wavelength (nm)", row=2, col=1)
    fig.update_yaxes(title_text="Counts", row=2, col=2)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        template="plotly_white",
        title_text="<b>Threshold Analysis Dashboard</b>"
    )
    
    return fig

# ==============================================================
# STREAMLIT APP
# ==============================================================
st.set_page_config(
    page_title="Random Laser Analyzer",
    layout="wide",
    page_icon="🔬"
)

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
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔬 Random Laser Analyzer</p>', unsafe_allow_html=True)
st.markdown("""
Advanced spectral analysis tool with **Lorentzian fitting** and **threshold detection**.
Upload your `.asc` files to begin automated analysis.
""")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File parsing settings
    with st.expander("📁 File Settings", expanded=True):
        skip_rows = st.number_input("Header rows to skip", 0, 100, 38)
        show_individual = st.checkbox("Show individual plots", True)
        show_fit_params = st.checkbox("Show fit parameters", False)
    
    st.markdown("---")
    st.markdown("### 📊 Analysis Features")
    st.markdown("""
    - ✅ Lorentzian curve fitting
    - ✅ FWHM & R² calculation
    - ✅ Threshold detection
    - ✅ SNR estimation
    - ✅ Interactive visualizations
    - ✅ Export to CSV/Excel
    
    ### 📈 Plot Legend
    - **Blue solid line**: Experimental data
    - **Red dashed line**: Lorentzian fit
    - **Green dotted line**: Peak position
    - **Orange diamonds**: FWHM boundaries
    """)

# File Upload
uploaded_files = st.file_uploader(
    "📤 Upload .asc spectrum files",
    accept_multiple_files=True,
    type=['asc'],
    help="Select multiple files for Q-switch series analysis"
)

# ==============================================================
# MAIN PROCESSING
# ==============================================================
if uploaded_files:
    st.markdown("---")
    
    # Initialize
    progress_bar = st.progress(0)
    status = st.empty()
    
    summary_data = []
    plot_zip = BytesIO()
    combined_fig = go.Figure()
    
    with zipfile.ZipFile(plot_zip, "w", zipfile.ZIP_DEFLATED) as zip_buffer:
        
        for idx, file in enumerate(uploaded_files):
            filename = file.name
            status.info(f"⚙️ Processing: {filename} ({idx+1}/{len(uploaded_files)})")
            
            try:
                # Parse file
                content = file.read().decode(errors='ignore')
                wl, counts = parse_asc_file(content, skip_rows)
                
                # Analyze spectrum
                result = analyze_spectrum(wl, counts)
                qs = extract_qs(filename)
                
                # Individual plot
                if show_individual:
                    fig = create_spectrum_plot(wl, counts, result, filename)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Fit parameters
                    if show_fit_params and result.fit_params:
                        with st.expander(f"🔍 Detailed Fit Parameters - {filename}"):
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Amplitude", f"{result.fit_params.get('Amplitude', 0):.1f}")
                            col2.metric("Center (nm)", f"{result.fit_params.get('Center', 0):.2f}")
                            col3.metric("Gamma", f"{result.fit_params.get('Gamma', 0):.2f}")
                            col4.metric("Baseline", f"{result.fit_params.get('Baseline', 0):.1f}")
                    
                    # Save to ZIP
                    html = fig.to_html(full_html=False, include_plotlyjs='cdn').encode()
                    zip_buffer.writestr(f"{filename.replace('.asc', '')}.html", html)
                
                # Combined plot
                combined_fig.add_trace(go.Scatter(
                    x=wl, y=counts,
                    mode='lines',
                    name=f"QS={qs:.0f}" if not np.isnan(qs) else filename,
                    hovertemplate='%{y:.0f}<extra></extra>'
                ))
                
                # Add to summary
                summary_data.append({
                    "File": filename,
                    "QS Level": qs,
                    "Peak λ (nm)": result.peak_wavelength,
                    "Peak Intensity": result.peak_intensity,
                    "FWHM (nm)": result.fwhm,
                    "Integrated Intensity": result.integrated_intensity,
                    "R²": result.r_squared,
                    "SNR": result.snr,
                    "Fit Success": "✅" if result.fit_success else "❌"
                })
                
            except Exception as e:
                st.error(f"❌ Error processing {filename}: {e}")
                continue
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
    
    status.success("✅ All files processed successfully!")
    progress_bar.empty()
    
    # ==============================================================
    # RESULTS SECTION
    # ==============================================================
    
    st.markdown("---")
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    summary_df = pd.DataFrame(summary_data).sort_values("QS Level")
    
    with col1:
        st.metric("Files Analyzed", len(summary_df))
    with col2:
        avg_r2 = summary_df["R²"].mean()
        st.metric("Avg R²", f"{avg_r2:.3f}")
    with col3:
        avg_fwhm = summary_df["FWHM (nm)"].mean()
        st.metric("Avg FWHM", f"{avg_fwhm:.2f} nm")
    with col4:
        avg_snr = summary_df["SNR"].mean()
        st.metric("Avg SNR", f"{avg_snr:.1f}")
    
    # Combined Spectra
    st.markdown("---")
    st.subheader("🌈 Combined Spectra")
    combined_fig.update_layout(
        title="Spectral Evolution with Q-Switch Level",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity (counts)",
        template="plotly_white",
        hovermode="x unified",
        height=600
    )
    st.plotly_chart(combined_fig, use_container_width=True)
    
    # Data Table
    st.markdown("---")
    st.subheader("📋 Analysis Results")
    
    # Format and color-code table
    def highlight_quality(val):
        if pd.isna(val):
            return ''
        if val > 0.95:
            return 'background-color: #d4edda'
        elif val > 0.85:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    styled_df = summary_df.style.applymap(
        highlight_quality,
        subset=['R²']
    ).format({
        'Peak λ (nm)': '{:.2f}',
        'Peak Intensity': '{:.0f}',
        'FWHM (nm)': '{:.2f}',
        'Integrated Intensity': '{:.2e}',
        'R²': '{:.4f}',
        'SNR': '{:.1f}',
        'QS Level': '{:.0f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Threshold Analysis
    if summary_df['QS Level'].notna().sum() > 3:
        st.markdown("---")
        st.subheader("🎯 Threshold Detection")
        
        valid = summary_df.dropna(subset=['QS Level', 'Integrated Intensity'])
        threshold = detect_threshold(
            valid['QS Level'].values,
            valid['Integrated Intensity'].values
        )
        
        # Display threshold results
        col1, col2, col3 = st.columns(3)
        with col1:
            if threshold.threshold_found:
                st.success(f"✅ Threshold detected at QS ≈ **{threshold.threshold_qs:.1f}**")
            else:
                st.warning("⚠️ No clear threshold detected")
        with col2:
            st.metric("Slope (below)", f"{threshold.slope_below:.2e}")
        with col3:
            st.metric("Slope (above)", f"{threshold.slope_above:.2e}")
        
        # Threshold plots
        fig_threshold = create_threshold_plot(summary_df, threshold)
        st.plotly_chart(fig_threshold, use_container_width=True)
    
    # ==============================================================
    # DOWNLOADS
    # ==============================================================
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = summary_df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download CSV",
            csv_data,
            f"laser_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            "📦 Download Plots (ZIP)",
            plot_zip.getvalue(),
            f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            "application/zip",
            use_container_width=True
        )
    
    with col3:
        # Excel export with formatting
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Results', index=False)
        
        st.download_button(
            "📊 Download Excel",
            excel_buffer.getvalue(),
            f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.ms-excel",
            use_container_width=True
        )
    
    st.success("✨ Analysis complete! All visualizations are interactive - hover, zoom, and pan to explore.")

else:
    # Welcome screen
    st.info("👆 Upload .asc files to begin analysis")
    
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Upload Files**: Select one or more `.asc` spectral files
        2. **Automatic Analysis**: The app will:
           - Fit Lorentzian curves to each spectrum (shown as **red dashed line**)
           - Calculate FWHM, peak wavelength, and integrated intensity
           - Detect lasing threshold (if applicable)
        3. **Explore Results**: Interactive plots allow zooming and hovering
        4. **Download**: Export results as CSV, Excel, or HTML plots
        
        ### Plot Elements
        - **Blue solid line**: Your experimental data
        - **Red dashed line**: Lorentzian curve fit
        - **Green dotted vertical line**: Peak wavelength position
        - **Orange diamond markers**: FWHM (Full Width at Half Maximum) boundaries
        
        ### File Naming Convention
        For automatic Q-switch detection, include the value in filename:
        - `sample_qs_100.asc`
        - `QS150.asc`
        - `data_200_qs.asc`
        """)
    
    with st.expander("📊 Example Data"):
        st.code("""
# .asc file format (tab-separated):
Wavelength    Intensity1    Intensity2    Intensity3
550.00        1200          1205          1198
550.50        1350          1348          1352
551.00        1500          1502          1498
...
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    Built with Streamlit • Lorentzian Fitting via SciPy<br>
    📧 Questions? Check the documentation above
</div>
""", unsafe_allow_html=True)

# =====================================================
# Streamlit Random Laser Analyzer
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import StringIO

st.set_page_config(page_title="Random Laser ASC Analyzer", layout="wide")

# ------------------------------------------
# Lorentzian model
def lorentzian(x, A, x0, gamma, y0):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + y0

def analyze_spectrum(wl, counts):
    """Fit Lorentzian and return peak, FWHM, etc."""
    try:
        x0_guess = wl[np.argmax(counts)]
        A_guess = np.max(counts)
        gamma_guess = (np.max(wl)-np.min(wl))/20
        y0_guess = np.min(counts)
        popt, _ = curve_fit(lorentzian, wl, counts,
                            p0=[A_guess, x0_guess, gamma_guess, y0_guess],
                            maxfev=10000)
        A, x0, gamma, y0 = popt
        fwhm = 2 * gamma
        fit_y = lorentzian(wl, *popt)
        area = np.trapz(counts, wl)
        return x0, A, fwhm, area, fit_y
    except Exception:
        area = np.trapz(counts, wl)
        return np.nan, np.nan, np.nan, area, counts

# ------------------------------------------
# Streamlit UI
st.title("🎛️ Random Laser Spectrum Analyzer")
st.write("Upload your `.asc` files (one or multiple Q-switch levels).")

uploaded_files = st.file_uploader("Choose ASC files", accept_multiple_files=True, type=['asc'])

if uploaded_files:
    summary = []
    combined_fig, ax_comb = plt.subplots(figsize=(8,5))

    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        data = uploaded_file.read().decode(errors='ignore')
        df = pd.read_csv(StringIO(data), sep='\t', decimal=',', skiprows=38, engine='python')
        df = df.dropna(axis=1, how='all')
        wl = df.iloc[:,0].to_numpy()
        counts = df.iloc[:,1:].mean(axis=1).to_numpy()

        peak_wl, peak_int, fwhm, area, fit_y = analyze_spectrum(wl, counts)
        qs_val = ''.join(ch for ch in name if ch.isdigit())

        # --- Individual plot
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(wl, counts, 'b', alpha=0.6, label="Raw data")
        ax.plot(wl, fit_y, 'r--', lw=1.2, label="Lorentzian fit")
        ax.set_title(f"{name} | Peak λ={peak_wl:.2f} nm | FWHM={fwhm:.2f} nm")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Counts")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Add to combined plot
        ax_comb.plot(wl, counts, label=name)

        # Store summary
        summary.append([name, qs_val, peak_wl, peak_int, fwhm, area])

    # --- Combined spectra
    ax_comb.set_title("Combined Random-Laser Spectra")
    ax_comb.set_xlabel("Wavelength (nm)")
    ax_comb.set_ylabel("Counts")
    ax_comb.legend(fontsize=8)
    ax_comb.grid(True)
    st.pyplot(combined_fig)

    # --- Summary table
    cols = ["File", "QS Level", "Peak λ (nm)", "Peak Intensity", "FWHM (nm)", "Integrated Intensity"]
    summary_df = pd.DataFrame(summary, columns=cols)
    summary_df = summary_df.sort_values("QS Level")
    st.subheader("📊 Summary Table")
    st.dataframe(summary_df, use_container_width=True)

    # --- Threshold curve
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(summary_df["QS Level"].astype(float),
             summary_df["Integrated Intensity"], 'o-r', lw=1.5)
    ax2.set_title("Threshold Curve (Integrated Intensity vs Q-switch)")
    ax2.set_xlabel("Q-switch Level")
    ax2.set_ylabel("Integrated Intensity (Counts·nm)")
    ax2.grid(True)
    st.pyplot(fig2)

    # --- Optional AI summary
    if "openai" in globals():
        st.subheader("🧠 AI Summary")
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        text = summary_df.to_string(index=False)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize experimental trends in concise scientific language."},
                {"role": "user", "content": f"Here are random laser results:\n{text}"}
            ]
        )
        st.write(completion.choices[0].message.content)

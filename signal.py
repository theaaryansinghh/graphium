import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from scipy import signal
from scipy.optimize import minimize, differential_evolution
import plotly.graph_objects as go
import pandas as pd
import io
import logging

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================

def export_csv(data, filename="data.csv"):
    """Convert a pandas DataFrame to CSV bytes and provide a download button."""
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name=filename)

def annotate_peak(ax, freqs, fft_vals):
    """Annotate the peak frequency in the FFT plot."""
    half = len(freqs) // 2
    idx = np.argmax(np.abs(fft_vals[:half]))
    peak_freq = freqs[idx]
    peak_mag = np.abs(fft_vals[idx])
    ax.annotate(f"Peak: {peak_freq:.1f} Hz", xy=(peak_freq, peak_mag),
                xytext=(peak_freq, peak_mag * 1.1),
                arrowprops=dict(facecolor='white', shrink=0.05),
                color='white')
    return peak_freq, peak_mag

def safe_eval_function(func_str, x):
    """Safely evaluate a custom function provided as a string using numpy."""
    try:
        allowed_names = {"np": np, "x": x}
        return eval(func_str, {"__builtins__": None}, allowed_names)
    except Exception as e:
        st.error(f"Error evaluating function: {e}")
        return None

# =============================================================================
# Fourier Transform Visualizer
# =============================================================================
def fourier_transform_visualizer():
    st.header("Fourier Transform Visualizer")
    st.markdown("""
    **Time-Domain Signal and Its Fourier Transform**

    Adjust the frequency of a sine wave to see its time-domain signal and corresponding frequency spectrum.
    You can choose the unit for frequency.
    """)
    freq = st.slider("Frequency of Sine Wave", 1, 50, 5, key='ft_freq')
    unit = st.selectbox("Frequency Unit", ["Hz", "kHz"])
    unit_factor = 1 if unit == "Hz" else 1e-3

    duration = 1.0  # seconds
    sample_rate = 1000  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal_data = np.sin(2 * np.pi * freq * t)

    try:
        fft_values = fft(signal_data)
        freqs = fftfreq(len(t), 1 / sample_rate)
    except Exception as e:
        st.error(f"FFT computation error: {e}")
        return

    # Plot using Matplotlib
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(t, signal_data, color='cyan')
    ax[0].set_title("Time Domain Signal")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid(True)

    half = len(freqs) // 2
    ax[1].stem(freqs[:half] * unit_factor, np.abs(fft_values[:half]), linefmt="C1-", markerfmt="C1o")
    ax[1].set_title("Frequency Domain")
    ax[1].set_xlabel(f"Frequency ({unit})")
    ax[1].set_ylabel("Magnitude")
    ax[1].grid(True)
    try:
        annotate_peak(ax[1], freqs, fft_values)
    except Exception as e:
        logger.warning(f"Annotation error: {e}")

    st.pyplot(fig)
    # Export time-domain signal as CSV
    df = pd.DataFrame({"Time (s)": t, "Amplitude": signal_data})
    export_csv(df, "time_signal.csv")
    st.markdown("Use the download button above to export the time-domain signal data.")

# =============================================================================
# Spectrogram Visualizer
# =============================================================================
def spectrogram_visualizer():
    st.header("Spectrogram Visualizer")
    st.markdown("""
    **Spectrogram:** This plot shows how the frequency content of the signal changes over time.
    """)
    spec_freq = st.slider("Frequency of Sine Wave (for Spectrogram)", 1, 50, 5, key='spec_freq')
    duration = 1.0
    sample_rate = 1000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal_data_spec = np.sin(2 * np.pi * spec_freq * t)
    
    try:
        f, t_spec, Sxx = signal.spectrogram(signal_data_spec, sample_rate)
    except Exception as e:
        st.error(f"Spectrogram computation error: {e}")
        return

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    c = ax2.pcolormesh(t_spec, f, np.log(Sxx + 1e-10), shading='auto')
    ax2.set_title("Spectrogram (Log Scale)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    fig2.colorbar(c, ax=ax2)
    st.pyplot(fig2)

# =============================================================================
# Optimization Algorithm Visualizer
# =============================================================================
def optimization_visualizer():
    st.header("Optimization Algorithm Visualizer")
    st.markdown("""
    **Function Optimization:**
    
    Choose one of the predefined functions (or input your own) and see how different optimization algorithms 
    (Gradient Descent, SciPy's minimize, and Differential Evolution) find the minimum.
    """)
    
    function_choice = st.selectbox("Choose a function to optimize", ["Quadratic", "Sine", "Complex", "Custom"])
    
    if function_choice == "Quadratic":
        def quadratic(x):
            return x**2 + 3*x + 5
        func = quadratic
        func_str = "x**2 + 3*x + 5"
    elif function_choice == "Sine":
        def sine_function(x):
            return np.sin(x) + 0.5*x
        func = sine_function
        func_str = "sin(x) + 0.5*x"
    elif function_choice == "Complex":
        def complex_function(x):
            return np.sin(3*x) + 0.5*x**2 - 2*x + 4
        func = complex_function
        func_str = "sin(3*x) + 0.5*x**2 - 2*x + 4"
    else:
        func_str = st.text_input("Enter your function f(x):", "x**2 + 3*x + 5")
        func = lambda x: safe_eval_function(func_str, x)
        if func(1) is None:
            return
    
    st.markdown(f"**Function:** $f(x) = {func_str}$")
    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.array([func(x) for x in x_vals])
    
    # Plot function
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(x_vals, y_vals, label=f"f(x) = {function_choice}")
    ax3.set_title("Function to Optimize")
    ax3.set_xlabel("x")
    ax3.set_ylabel("f(x)")
    ax3.grid(True)
    
    st.sidebar.header("Gradient Descent Settings")
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
    iterations = st.sidebar.slider("Iterations", 10, 500, 50)
    
    def gradient_descent(f, x_start, lr, steps):
        x = x_start
        path = [x]
        for _ in range(steps):
            grad = (f(x + 1e-5) - f(x)) / 1e-5
            x = x - lr * grad
            path.append(x)
        return x, path
    
    try:
        x_min_gd, gd_path = gradient_descent(func, x_start=5, lr=learning_rate, steps=iterations)
    except Exception as e:
        st.error(f"Gradient descent error: {e}")
        return
    
    ax3.plot(gd_path, [func(x) for x in gd_path], 'r--', label="Gradient Descent Path")
    ax3.scatter(x_min_gd, func(x_min_gd), color='red', label="Gradient Descent Minimum")
    
    try:
        res = minimize(func, x0=0)
    except Exception as e:
        st.error(f"SciPy minimize error: {e}")
        return
    ax3.scatter(res.x, res.fun, color='blue', label="SciPy Minimize")
    
    try:
        res_ga = differential_evolution(func, bounds=[(-10, 10)])
    except Exception as e:
        st.error(f"Genetic Algorithm error: {e}")
        return
    ax3.scatter(res_ga.x, res_ga.fun, color='green', label="Genetic Algorithm")
    
    ax3.legend()
    st.pyplot(fig3)
    
    summary_data = pd.DataFrame({
        "Method": ["Gradient Descent", "SciPy Minimize", "Genetic Algorithm"],
        "x_min": [x_min_gd, res.x[0], res_ga.x[0]],
        "f(x_min)": [func(x_min_gd), res.fun, res_ga.fun]
    })
    st.table(summary_data)
    
    df_opt = pd.DataFrame({"x": x_vals, "f(x)": y_vals})
    export_csv(df_opt, "optimization_function.csv")

# =============================================================================
# Real-Time Time Evolution Visualizer
# =============================================================================
def time_evolution_visualizer():
    st.header("Time Evolution Visualizer")
    st.markdown(r"""
    **Time Evolution of a Signal:**

    This section shows how a sine wave with a time-dependent phase evolves.
    
    The signal is modeled as:
    
    $$
    y(t) = \sin\Bigl(2\pi f x + \phi(t)\Bigr),
    $$
    
    where we use a simple linear phase evolution: \(\phi(t) = t\).
    """)
    freq = st.slider("Wave Frequency", 1, 10, 2)
    t_val = st.slider("Time (t)", 0.0, 10.0, 0.0, 0.1)
    phase = t_val  # simple linear phase evolution
    x = np.linspace(0, 1, 500)
    y = np.sin(2 * np.pi * freq * x + phase)
    
    fig_anim = go.Figure()
    fig_anim.add_trace(go.Scatter(x=x, y=y, mode='lines', name="Signal"))
    fig_anim.update_layout(
        title=f"Signal at t = {t_val:.2f}",
        xaxis_title="x",
        yaxis_title="Amplitude",
        template="plotly_dark"
    )
    st.plotly_chart(fig_anim, use_container_width=True)

# =============================================================================
# Export Data Section
# =============================================================================
def export_wavefunction_data():
    st.header("Export Signal Data")
    st.markdown(r"""
    **Export Signal Data:**
    
    Choose a scenario and download the corresponding signal data (x, Re(signal), Im(signal)) as CSV.
    """)
    scenario = st.selectbox("Select Scenario", ["Gaussian", "Plane Wave", "Sinusoidal"])
    x = np.linspace(-10, 10, 500)
    if scenario == "Gaussian":
        signal_val = np.exp(-0.5 * x**2)
    elif scenario == "Plane Wave":
        k = 1.0
        signal_val = np.exp(1j * k * x)
    else:
        signal_val = np.sin(2 * np.pi * x / 5)
    
    df = pd.DataFrame({"x": x, "Re(signal)": signal_val.real, "Im(signal)": signal_val.imag})
    export_csv(df, f"{scenario}_signal.csv")
    
    fig_exp = go.Figure()
    fig_exp.add_trace(go.Scatter(x=x, y=signal_val.real, mode='lines', name="Re(signal)"))
    fig_exp.add_trace(go.Scatter(x=x, y=signal_val.imag, mode='lines', name="Im(signal)"))
    fig_exp.update_layout(template="plotly_dark", title=f"{scenario} Signal")
    st.plotly_chart(fig_exp, use_container_width=True)

# =============================================================================
# Main App Navigation
# =============================================================================
st.sidebar.title("Signal Processing & Optimization")
pages = [
    "Fourier Transform",
    "Spectrogram",
    "Optimization",
    "Time Evolution",
    "Export Data"
]
choice = st.sidebar.radio("Select a Page:", pages)

if choice == "Fourier Transform":
    fourier_transform_visualizer()
elif choice == "Spectrogram":
    spectrogram_visualizer()
elif choice == "Optimization":
    optimization_visualizer()
elif choice == "Time Evolution":
    time_evolution_visualizer()
elif choice == "Export Data":
    export_wavefunction_data()

# =============================================================================
# About This App
# =============================================================================
st.markdown("---")
st.markdown("## About This App")
st.markdown(r"""
This **Signal Processing & Optimization** app provides interactive visualizations for key concepts:

- **Fourier Transform Visualizer:** Explore time-domain signals and their frequency spectra. Adjust frequency and units, and export signal data.
- **Spectrogram Visualizer:** See how the frequency content of a signal changes over time.
- **Optimization Algorithm Visualizer:** Optimize a function using Gradient Descent, SciPy's minimize, and Differential Evolution, and compare the results.
- **Time Evolution Visualizer:** Observe the evolution of a time-varying signal with a time-dependent phase.
- **Export Data:** Download signal data as CSV for further analysis.

**Additional Features:**
- Interactive annotations (e.g., peak frequency markers).
- Custom function input for optimization.
- Detailed educational context and mathematical formulas.
- Export options for data and plots.
- Unit conversion options.

Enhance your learning experience in signal processing and optimization with this interactive app!
""")

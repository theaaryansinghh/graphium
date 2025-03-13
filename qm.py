import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
import time
import io
import pandas as pd

# =============================================================================
# Utility / Helper Functions
# =============================================================================

def wavefunction_to_csv(x, psi):
    """Convert wavefunction data (x, psi) into CSV bytes for downloading."""
    df = pd.DataFrame({"x": x, "Re(psi)": psi.real, "Im(psi)": psi.imag})
    return df.to_csv(index=False).encode('utf-8')

def numeric_energy(n, L):
    """For a particle in a 1D box, returns energy ~ n^2 (ignoring constants)."""
    return n**2

def piecewise_potential(x, left_height, center_height, right_height):
    """Define a piecewise potential with three segments."""
    V = np.zeros_like(x)
    for i, val in enumerate(x):
        if val < -1:
            V[i] = left_height
        elif val <= 1:
            V[i] = center_height
        else:
            V[i] = right_height
    return V

# =============================================================================
# Page Functions
# =============================================================================

def page_particle_in_a_box():
    st.title("Particle in a 1D Box")
    
    st.markdown(r"""
    ### Particle in a 1D Infinite Potential Well
    In an infinite potential well (box) of length **L**, the normalized wavefunctions are given by:
    
    $$
    \psi_n(x) = \sqrt{\frac{2}{L}}\,\sin\left(\frac{n\pi x}{L}\right),
    $$
    
    and the corresponding probability density is:
    
    $$
    |\psi_n(x)|^2 = \frac{2}{L}\,\sin^2\left(\frac{n\pi x}{L}\right).
    $$
    
    The energy levels scale as \(E_n \propto n^2\).
    """)
    
    # Energy level diagram
    st.subheader("Energy Level Diagram")
    L = st.slider("Box Length (L)", min_value=1, max_value=10, value=5, step=1)
    max_n = st.slider("Max Quantum Number for Diagram", min_value=1, max_value=10, value=5)
    energies = [numeric_energy(n, L) for n in range(1, max_n+1)]
    fig_diag = go.Figure()
    for i, E in enumerate(energies):
        fig_diag.add_trace(go.Scatter(
            x=[0, 1], y=[E, E],
            mode='lines', line=dict(width=3),
            name=f"n={i+1}"
        ))
    fig_diag.update_layout(
        title="Energy Levels (Schematic: E ∝ n²)",
        xaxis=dict(visible=False),
        yaxis_title="Energy",
        template="plotly_dark",
        height=300
    )
    st.plotly_chart(fig_diag, use_container_width=True)
    
    # Wavefunction plot
    st.subheader("Wavefunction & Probability Density")
    n = st.slider("Quantum Number (n)", min_value=1, max_value=5, value=1)
    x = np.linspace(0, L, 500)
    psi_n = np.sqrt(2 / L) * np.sin(n * np.pi * x / L)
    fig_wf = go.Figure(go.Scatter(
        x=x, y=psi_n**2,
        mode='lines', name=f"|ψ(x)|² for n={n}",
        line=dict(color="royalblue")
    ))
    fig_wf.update_layout(
        title="Probability Density |ψ(x)|²",
        xaxis_title="Position (x)",
        yaxis_title="|ψ(x)|²",
        template="plotly_dark"
    )
    st.plotly_chart(fig_wf, use_container_width=True)
    
    st.markdown(r"""
    **Additional Context:**
    
    The particle in a box is a fundamental model in quantum mechanics that shows how energy levels become quantized.
    As the quantum number **n** increases, the number of nodes in the wavefunction increases, and the energy rises roughly as \(n^2\).
    """)
    
def page_superposition_multiple_states():
    st.title("Multiple State Superposition & Interference")
    
    st.markdown(r"""
    ### Superposition of Quantum States
    When quantum states are superposed, the resulting wavefunction is the sum of the individual wavefunctions:
    
    $$
    \psi_{\text{total}}(x) = \psi_1(x) + \psi_2(x) + \psi_3(x).
    $$
    
    The interference pattern (constructive and destructive interference) reveals how phase and amplitude affect the total state.
    """)
    
    x = np.linspace(-10, 10, 1000)
    st.subheader("Wave 1")
    k1 = st.slider("k1 (wave number)", 1, 5, 1)
    amp1 = st.slider("Amplitude A₁", 0.0, 2.0, 1.0, 0.1, key="amp1")
    phase1 = st.slider("Phase φ₁ (radians)", 0.0, 2*np.pi, 0.0, 0.1, key="phase1")
    
    st.subheader("Wave 2")
    k2 = st.slider("k2 (wave number)", 1, 5, 2)
    amp2 = st.slider("Amplitude A₂", 0.0, 2.0, 1.0, 0.1, key="amp2")
    phase2 = st.slider("Phase φ₂ (radians)", 0.0, 2*np.pi, 0.0, 0.1, key="phase2")
    
    st.subheader("Wave 3")
    k3 = st.slider("k3 (wave number)", 1, 5, 3)
    amp3 = st.slider("Amplitude A₃", 0.0, 2.0, 1.0, 0.1, key="amp3")
    phase3 = st.slider("Phase φ₃ (radians)", 0.0, 2*np.pi, 0.0, 0.1, key="phase3")
    
    psi1 = amp1 * np.sin(k1*x + phase1)
    psi2 = amp2 * np.sin(k2*x + phase2)
    psi3 = amp3 * np.sin(k3*x + phase3)
    psi_total = psi1 + psi2 + psi3
    
    fig_sp = go.Figure()
    fig_sp.add_trace(go.Scatter(x=x, y=psi1, mode='lines', name="Wave 1"))
    fig_sp.add_trace(go.Scatter(x=x, y=psi2, mode='lines', name="Wave 2"))
    fig_sp.add_trace(go.Scatter(x=x, y=psi3, mode='lines', name="Wave 3"))
    fig_sp.add_trace(go.Scatter(x=x, y=psi_total, mode='lines', name="Total", line=dict(dash='dash')))
    fig_sp.update_layout(
        title="Multiple State Superposition",
        xaxis_title="Position (x)",
        yaxis_title="Amplitude",
        template="plotly_dark"
    )
    st.plotly_chart(fig_sp, use_container_width=True)
    
    st.markdown(r"""
    **Additional Context:**
    
    The relative phases and amplitudes of the individual wave functions determine the interference pattern.
    Adjust the sliders to see how constructive and destructive interference emerge.
    """)
    
def page_tunneling_editor():
    st.title("Quantum Tunneling: Custom Potential Editor")
    
    st.markdown(r"""
    ### Quantum Tunneling
    Quantum tunneling is a phenomenon where a particle can cross a potential barrier even if its energy is lower than the barrier's height.
    
    In this demonstration, you can adjust the potential heights in three regions. The wavefunction (a Gaussian) is shown for illustration.
    """)
    
    left_height = st.slider("Left Region Potential", 0.0, 20.0, 0.0, 1.0)
    center_height = st.slider("Center Region Potential", 0.0, 20.0, 10.0, 1.0)
    right_height = st.slider("Right Region Potential", 0.0, 20.0, 0.0, 1.0)
    
    x = np.linspace(-5, 5, 500)
    V = piecewise_potential(x, left_height, center_height, right_height)
    psi = np.exp(-0.5 * x**2)  # Gaussian wave packet (illustrative)
    
    fig_pe = go.Figure()
    fig_pe.add_trace(go.Scatter(x=x, y=psi, mode='lines', name="Wavefunction", line=dict(color="limegreen")))
    fig_pe.add_trace(go.Scatter(x=x, y=V, mode='lines', name="Potential", line=dict(color="red", dash='dash')))
    fig_pe.update_layout(
        title="Custom Potential & Wavefunction",
        xaxis_title="Position (x)",
        yaxis_title="Amplitude / Potential",
        template="plotly_dark"
    )
    st.plotly_chart(fig_pe, use_container_width=True)
    
    st.markdown(r"""
    **Additional Context:**
    
    In reality, solving the Schrödinger equation for tunneling requires numerical methods.
    This demonstration uses a fixed Gaussian wave packet to illustrate the concept of a potential barrier.
    """)
    
def page_harmonic_oscillator():
    st.title("1D Quantum Harmonic Oscillator")
    
    st.markdown(r"""
    ### Quantum Harmonic Oscillator
    The harmonic oscillator is a fundamental model in quantum mechanics. Its eigenfunctions involve Hermite polynomials:
    
    $$
    \psi_n(x) \propto H_n(\alpha x) \, e^{-\alpha^2 x^2/2},
    $$
    
    where $\alpha = \sqrt{\frac{m\omega}{\hbar}}$. Here, we provide a simplified demonstration.
    """)
    
    n = st.slider("Quantum Number (n)", 0, 5, 0)
    x = np.linspace(-4, 4, 400)
    if n == 0:
        psi = np.exp(-0.5 * x**2)
    elif n == 1:
        psi = x * np.exp(-0.5 * x**2)
    elif n == 2:
        psi = (x**2 - 1) * np.exp(-0.5 * x**2)
    else:
        psi = (x**n) * np.exp(-0.5 * x**2)
    
    prob = psi**2
    fig_ho = go.Figure(go.Scatter(x=x, y=prob, mode='lines', line=dict(color="blue")))
    fig_ho.update_layout(
        title=f"Harmonic Oscillator Probability Density (n={n})",
        xaxis_title="x",
        yaxis_title="|ψ(x)|²",
        template="plotly_dark"
    )
    st.plotly_chart(fig_ho, use_container_width=True)
    
    st.markdown(r"""
    **Additional Context:**
    
    The harmonic oscillator has equally spaced energy levels and wavefunctions that change shape with n.
    This demonstration uses a simplified version for visualization purposes.
    """)

def page_bloch_mixed_states():
    st.title("Bloch Sphere with Mixed States")
    
    st.markdown(r"""
    ### Bloch Sphere Representation
    A qubit's pure state is represented on the surface of the Bloch sphere. Mixed states (with lower purity)
    lie inside the sphere.
    
    The state vector for a pure state is:
    
    $$
    |\psi\rangle = \cos\!\Bigl(\frac{\theta}{2}\Bigr)|0\rangle + e^{i\phi}\,\sin\!\Bigl(\frac{\theta}{2}\Bigr)|1\rangle.
    $$
    
    The corresponding Bloch sphere coordinates are:
    
    $$
    x = \sin\theta\cos\phi,\quad y = \sin\theta\sin\phi,\quad z = \cos\theta.
    $$
    """)
    
    theta = st.slider("Theta (θ)", 0.0, np.pi, np.pi/4, 0.05)
    phi = st.slider("Phi (φ)", 0.0, 2*np.pi, np.pi/2, 0.05)
    purity = st.slider("Purity (0 = fully mixed, 1 = pure)", 0.0, 1.0, 1.0, 0.01)
    
    r_val = purity  # Pure state if r=1; mixed states have r<1
    x_state = r_val * np.sin(theta) * np.cos(phi)
    y_state = r_val * np.sin(theta) * np.sin(phi)
    z_state = r_val * np.cos(theta)
    
    fig = go.Figure()
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50))
    X_s = np.cos(u) * np.sin(v)
    Y_s = np.sin(u) * np.sin(v)
    Z_s = np.cos(v)
    fig.add_trace(go.Surface(x=X_s, y=Y_s, z=Z_s, colorscale='Blues', opacity=0.5, showscale=False))
    fig.add_trace(go.Scatter3d(
        x=[0, x_state], y=[0, y_state], z=[0, z_state],
        mode="lines+markers",
        line=dict(color="red", width=5),
        marker=dict(size=5, color="red"),
        name="State Vector"
    ))
    fig.update_layout(
        title="Bloch Sphere Representation",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(r"""
    **Additional Context:**
    
    By adjusting the purity slider, you can see how the state moves from the surface (pure state)
    to the center (completely mixed state).
    """)

def page_time_evolution():
    st.title("Time-Dependent Schrödinger Equation")
    
    st.markdown(r"""
    ### Time Evolution of a Wave Function
    The time-dependent Schrödinger equation governs the evolution of quantum states. A simple model is:
    
    $$
    \Psi(x,t) = \Psi(x,0) \exp\!\Bigl(-\frac{iEt}{\hbar}\Bigr).
    $$
    
    In this demonstration, a Gaussian wave packet is evolved in time.
    """)
    
    t = st.slider("Time (t)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    
    def schrodinger_evolution(x, t, k):
        return np.exp(-0.5 * x**2) * np.exp(-1j * k * x) * np.exp(-1j * t)
    
    x = np.linspace(-5, 5, 500)
    wavefunction = schrodinger_evolution(x, t, k=1.0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=wavefunction.real,
        mode='lines', name="Re(Ψ)", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=x, y=wavefunction.imag,
        mode='lines', name="Im(Ψ)", line=dict(color="red")
    ))
    fig.update_layout(
        title=f"Wave Function at t = {t:.2f}",
        xaxis_title="Position (x)",
        yaxis_title="Amplitude",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(r"""
    **Additional Context:**
    
    The time-dependent phase factor causes the wavefunction's complex parts to oscillate.
    This simple model illustrates how quantum states evolve in time.
    """)

def page_export_data():
    st.title("Export Wavefunction Data")
    
    st.markdown(r"""
    ### Exporting Wavefunction Data as CSV
    Choose a scenario and export the corresponding wavefunction data (x, Re(Ψ), Im(Ψ)) for further analysis.
    """)
    
    scenario = st.selectbox("Select Scenario", ["Gaussian", "Plane Wave", "Sinusoidal"])
    x = np.linspace(-10, 10, 500)
    if scenario == "Gaussian":
        psi = np.exp(-0.5*x**2)
    elif scenario == "Plane Wave":
        k = 1.0
        psi = np.exp(1j*k*x)
    else:
        psi = np.sin(2*np.pi*x/5)
    
    csv_bytes = wavefunction_to_csv(x, psi)
    st.download_button("Download Wavefunction CSV", data=csv_bytes, file_name=f"{scenario}_wave.csv")
    
    fig_exp = go.Figure()
    fig_exp.add_trace(go.Scatter(x=x, y=psi.real, mode='lines', name="Re(Ψ)"))
    fig_exp.add_trace(go.Scatter(x=x, y=psi.imag, mode='lines', name="Im(Ψ)"))
    fig_exp.update_layout(template="plotly_dark")
    st.plotly_chart(fig_exp, use_container_width=True)

def page_2D_infinite_well():
    st.title("2D Infinite Square Well")
    
    st.markdown(r"""
    ### 2D Infinite Square Well
    The 2D infinite square well has solutions:
    
    $$
    \psi_{n_x,n_y}(x,y) = \frac{2}{\sqrt{L_x L_y}} \sin\!\Bigl(\frac{n_x\pi x}{L_x}\Bigr) \sin\!\Bigl(\frac{n_y\pi y}{L_y}\Bigr).
    $$
    
    This demonstration plots the probability density for chosen quantum numbers.
    """)
    
    Lx = st.slider("Lx (well width in x)", 1, 10, 5)
    Ly = st.slider("Ly (well width in y)", 1, 10, 5)
    nx = st.slider("nₓ", 1, 5, 1)
    ny = st.slider("nᵧ", 1, 5, 1)
    
    x = np.linspace(0, Lx, 50)
    y = np.linspace(0, Ly, 50)
    X, Y = np.meshgrid(x, y)
    
    psi_2d = (2/np.sqrt(Lx*Ly)) * np.sin(nx*np.pi*X/Lx) * np.sin(ny*np.pi*Y/Ly)
    prob_2d = psi_2d**2
    
    fig_2d = go.Figure(data=[go.Surface(x=X, y=Y, z=prob_2d, colorscale="Viridis")])
    fig_2d.update_layout(
        title=f"2D Well Probability Density (nₓ={nx}, nᵧ={ny})",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="|ψ(x,y)|²"
        ),
        template="plotly_dark"
    )
    st.plotly_chart(fig_2d, use_container_width=True)

# =============================================================================
# Main App Navigation
# =============================================================================

st.sidebar.title("Quantum Mechanics")
pages = [
    "Particle in a 1D Box",
    "Multiple State Superposition",
    "Custom Potential Editor (Tunneling)",
    "1D Harmonic Oscillator",
    "Bloch Sphere (Mixed States)",
    "Time Evolution",
    "Export Wavefunction Data",
    "2D Infinite Square Well"
]
choice = st.sidebar.radio("Select a Page:", pages)

if choice == "Particle in a 1D Box":
    page_particle_in_a_box()
elif choice == "Multiple State Superposition":
    page_superposition_multiple_states()
elif choice == "Custom Potential Editor (Tunneling)":
    page_tunneling_editor()
elif choice == "1D Harmonic Oscillator":
    page_harmonic_oscillator()
elif choice == "Bloch Sphere (Mixed States)":
    page_bloch_mixed_states()
elif choice == "Time Evolution":
    page_time_evolution()
elif choice == "Export Wavefunction Data":
    page_export_data()
elif choice == "2D Infinite Square Well":
    page_2D_infinite_well()

# =============================================================================
# About This App
# =============================================================================

st.markdown("---")
st.markdown("## About This App")
st.markdown(r"""
This interactive **Quantum Mechanics** app is designed to help students and educators explore fundamental concepts through visualization.

**Features:**
- **Particle in a 1D Box:** Visualize energy levels and wavefunctions in an infinite potential well.
- **Multiple State Superposition:** See how different wavefunctions combine to create interference patterns.
- **Custom Potential Editor (Tunneling):** Adjust a piecewise potential barrier and observe a wavefunction.
- **1D Harmonic Oscillator:** Explore approximate wavefunctions of the harmonic oscillator.
- **Bloch Sphere (Mixed States):** Understand qubit states and how mixedness affects the state.
- **Time Evolution:** Observe the time-dependent behavior of a quantum wavefunction.
- **Export Wavefunction Data:** Download wavefunction data as CSV.
- **2D Infinite Square Well:** Visualize the probability density of a 2D quantum well.

**Further Reading & Resources:**
- [Quantum Mechanics by Griffiths](https://www.amazon.com/Introduction-Quantum-Mechanics-David-J-Griffiths/dp/0131118927)
- [MIT OpenCourseWare - Quantum Physics](https://ocw.mit.edu/courses/physics/8-04-quantum-physics-i-spring-2016/)
- [Wikipedia: Quantum Tunneling](https://en.wikipedia.org/wiki/Quantum_tunnelling)
- [Wikipedia: Bloch Sphere](https://en.wikipedia.org/wiki/Bloch_sphere)

This app is modularized into multiple pages, each focusing on a specific concept, and is designed to be both educational and interactive.
""")

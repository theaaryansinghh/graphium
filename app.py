import streamlit as st
import base64
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import streamlit.components.v1 as components

st.set_page_config(layout="wide")  # Full-width layout

def get_base64(file_path):
    """Read a file and return its Base64 encoded string."""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def home_page():
    # Read image files and convert to base64 strings
    img_data = get_base64("static/img.png")
    img2_data = get_base64("static/img2.png")
    
    # Read the index.html content
    with open("index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    
    # Replace image paths with inline base64 data
    html_content = html_content.replace(
        'src="/static/img.png"', 
        f'src="data:image/png;base64,{img_data}"'
    )
    html_content = html_content.replace(
        "background: url('img2.png')", 
        f"background: url('data:image/png;base64,{img2_data}')"
    )
    
    components.html(
        f"""
        <div style="width:100vw; height:100vh; overflow:hidden;">
            {html_content}
        </div>
        """, 
        height=800,  # Matches viewport height for full fit
        scrolling=False
    )

def lorenz_attractor():
    def lorenz(t, state, sigma=10, beta=8/3, rho=28):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    sol = solve_ivp(lorenz, [0, 50], [1, 1, 1], t_eval=np.linspace(0, 50, 10000))
    x, y, z = sol.y

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue', width=4)))
    fig.update_layout(
        width=1000, height=600,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
    )
    st.plotly_chart(fig, use_container_width=True)

def return_home():
    if st.button("Return to Homepage"):
        st.session_state.current_page = 'home'
        st.experimental_rerun()

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

if st.session_state.current_page == 'home':
    home_page()

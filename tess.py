import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

st.set_page_config(layout="wide")  # Enable full-width layout

# Function to visualize an even larger Lorenz attractor
def lorenz_attractor():
    def lorenz(t, state, sigma=10, beta=8/3, rho=28):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    
    sol = solve_ivp(lorenz, [0, 50], [1, 1, 1], t_eval=np.linspace(0, 50, 10000))
    x, y, z = sol.y
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue', width=4)))
    fig.update_layout(
        width=1800, height=1000,  # Make the graph significantly bigger
        margin=dict(l=0, r=0, t=0, b=0),  # Remove margins for full-width expansion
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )
    st.plotly_chart(fig, use_container_width=True)  # Use full container width


lorenz_attractor()
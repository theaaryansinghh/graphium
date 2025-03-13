import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from scipy.stats import norm, binom, poisson
from scipy.integrate import quad
from scipy.special import erf, gamma, beta
from streamlit_ace import st_ace  # For symbolic input editor
import pandas as pd
import requests
import re  # For regex-based preprocessing
import json  # For saving and loading sessions

# Import utility functions
from utils import (
    calculate_erf, calculate_gamma, calculate_beta,
    normal_pdf, binomial_pmf, poisson_pmf,
    symbolic_derivative, taylor_series_sin,
    integrate_function
)
from plotgraph import plot_graph
from d import threedplot
from eqsolv import eq_solver

# st.title("Mathematical Functions Calculator")


# st.header("Welcome to the Mathematical Functions Calculator!")
# st.write("You can now use all the features of the app.")

# Sidebar Menu
options = st.sidebar.radio(
    "Select a Tool:",
    [
        "Plot Graph",
        #"Interactive Tutorials",
        #"Save and Load Sessions",
        "3D Plot",
        "QUANTUM MECHANICS",
        "Equation Solver",
        "Matrix Operations",
        "Error Function",
        "Gamma Function",
        "Beta Function",
        "Normal Distribution",
        "Binomial Distribution",
        "Poisson Distribution",
        "Symbolic Differentiation",
        #"Integration Solver",
    ],
)

# About Section
st.sidebar.markdown("### About")
st.sidebar.markdown("""
**Aaryan Singh**  
Email: [theaaryansingh@outlook.com](mailto:theaaryansingh@outlook.com)  
LinkedIn: [aaryan-singhh](https://www.linkedin.com/in/aaryan-singhh)  
Instagram: [@theaaryansingh](https://instagram.com/theaaryansingh)   
""")

# Help Section
with st.sidebar.expander("Help"):
    st.write("""
   - I AM AARYAN.
    """)

# Plot Graph Section
if options == "Plot Graph":
    plot_graph()

if options == "Interactive Tutorials":
    st.header("Interactive Tutorials")

    st.write("""
    ### Welcome to the Interactive Tutorials!
    Here, you'll learn how to use the Mathematical Functions Calculator step by step.

    1. **Plot Graph**:
        - Enter an equation like `x^2 + y^2 - 1` or `sin(x)`.
        - Adjust sliders for additional variables (e.g., `a`, `b`).

    2. **3D Plot**:
        - Enter a 3D equation like `z = sin(x) + cos(y)`.
        - Visualize the 3D surface.

    3. **Equation Solver**:
        - Enter an equation like `x^2 - 4 = 0`.
        - Get the solution.

    4. **Matrix Operations**:
        - Enter a matrix and perform operations like determinant and inverse.

    5. **Save and Load Sessions**:
        - Save your current session (e.g., equation, slider values) as a JSON file.
        - Load a saved session from a JSON file.

    Explore the tools and have fun!
    """)

if options == "Save and Load Sessions":
    st.header("Save and Load Sessions")

    # Save session
    st.write("### Save Current Session")
    session_data = {
        "equation": st.session_state.get("equation", ""),
        "sliders": st.session_state.get("sliders", {}),
    }
    session_json = json.dumps(session_data)
    st.download_button(
        label="Download Session",
        data=session_json,
        file_name="session.json",
        mime="application/json",
    )

    # Load session
    st.write("### Load Session")
    uploaded_file = st.file_uploader("Upload a session file", type="json")
    if uploaded_file is not None:
        session_data = json.load(uploaded_file)
        st.session_state["equation"] = session_data.get("equation", "")
        st.session_state["sliders"] = session_data.get("sliders", {})
        st.success("Session loaded successfully!")



elif options == "3D Plot":
    threedplot()

elif options == "Equation Solver":
    eq_solver()

elif options == "Matrix Operations":
    st.header("Matrix Operations")

    # Input matrix size
    n = st.number_input("Enter the size of the matrix (n x n):", min_value=1, max_value=10, value=2, step=1)

    # Input matrix
    st.write(f"Enter a {n}x{n} matrix:")
    matrix = []
    for i in range(n):
        cols = st.columns(n)
        row = []
        for j in range(n):
            row.append(cols[j].number_input(f"A[{i+1},{j+1}]", value=1.0 if i == j else 0.0))
        matrix.append(row)

    matrix = np.array(matrix)

    # Operations
    operation = st.selectbox("Select operation:", ["Determinant", "Inverse", "Eigenvalues"])

    try:
        if operation == "Determinant":
            result = np.linalg.det(matrix)
            st.success(f"Determinant: {result}")
        elif operation == "Inverse":
            result = np.linalg.inv(matrix)
            st.success(f"Inverse:\n{result}")
        elif operation == "Eigenvalues":
            result = np.linalg.eigvals(matrix)
            st.success(f"Eigenvalues: {result}")
    except Exception as e:
        st.error(f"Error in matrix operation: {e}")

elif options == "Error Function":
    st.header("Error Function")
    x = st.number_input("Enter a value for x:", value=0.0, step=0.1)
    try:
        result = calculate_erf(x)
        st.success(f"The error function of {x} is: {result}")

        # Generate graph for the error function
        x_vals = np.linspace(-5, 5, 500)
        y_vals = erf(x_vals)

        fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode="lines", name="erf(x)"))
        fig.update_layout(
            title="Error Function",
            xaxis_title="x",
            yaxis_title="erf(x)",
        )
        st.plotly_chart(fig)

    except ValueError as e:
        st.error(str(e))

elif options == "Gamma Function":
    st.header("Gamma Function")
    x = st.number_input("Enter a value for x:", value=1.0, step=0.1)
    try:
        result = calculate_gamma(x)
        st.success(f"The gamma function of {x} is: {result}")
    except ValueError as e:
        st.error(str(e))

elif options == "Beta Function":
    st.header("Beta Function")
    x = st.number_input("Enter value for x:", value=1.0, step=0.1)
    y = st.number_input("Enter value for y:", value=1.0, step=0.1)
    try:
        result = calculate_beta(x, y)
        st.success(f"The beta function of {x} and {y} is: {result}")
    except ValueError as e:
        st.error(str(e))

elif options == "Normal Distribution":
    st.header("Normal Distribution PDF")
    x = st.slider("Enter a range for x:", -10.0, 10.0, (0.0, 1.0))
    mean = st.number_input("Enter mean:", value=0.0, step=0.1)
    stddev = st.number_input("Enter standard deviation:", value=1.0, step=0.1)
    try:
        x_vals = np.linspace(x[0], x[1], 500)
        y_vals = norm.pdf(x_vals, mean, stddev)
        fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode="lines", name="PDF"))
        fig.update_layout(
            title=f"Normal Distribution PDF (Mean={mean}, StdDev={stddev})",
            xaxis_title="x",
            yaxis_title="PDF",
        )
        st.plotly_chart(fig)
    except ValueError as e:
        st.error(str(e))

elif options == "Binomial Distribution":
    st.header("Binomial Distribution PMF")
    n = st.number_input("Enter the number of trials (n):", value=10, step=1, min_value=1)
    p = st.slider("Enter the probability of success (p):", 0.0, 1.0, 0.5)
    try:
        k_vals = np.arange(0, n + 1)
        pmf_vals = binom.pmf(k_vals, n, p)
        fig = go.Figure(data=go.Bar(x=k_vals, y=pmf_vals, name="PMF"))
        fig.update_layout(
            title=f"Binomial Distribution PMF (n={n}, p={p})",
            xaxis_title="Number of Successes (k)",
            yaxis_title="PMF",
        )
        st.plotly_chart(fig)
    except ValueError as e:
        st.error(str(e))

elif options == "Poisson Distribution":
    st.header("Poisson Distribution PMF")
    lam = st.number_input("Enter the average rate (lambda):", value=2.0, step=0.1)
    try:
        k_vals = np.arange(0, 20)
        pmf_vals = poisson.pmf(k_vals, lam)
        fig = go.Figure(data=go.Bar(x=k_vals, y=pmf_vals, name="PMF"))
        fig.update_layout(
            title=f"Poisson Distribution PMF (λ={lam})",
            xaxis_title="Number of Occurrences (k)",
            yaxis_title="PMF",
        )
        st.plotly_chart(fig)
    except ValueError as e:
        st.error(str(e))

elif options == "Symbolic Differentiation":
    st.header("Symbolic Differentiation")
    
    # Input expression from the user
    expression = st_ace(
        placeholder="Enter the expression to differentiate (in terms of x)",
        language="python",
        theme="monokai",
        value="x**2",
    )
    
    # Perform symbolic differentiation
    try:
        # Symbolic derivative
        result = symbolic_derivative(expression)
        latex_expression = sp.latex(sp.sympify(expression))
        latex_derivative = sp.latex(sp.sympify(result))

        # Display the results
        st.markdown(f"The derivative of:")
        st.latex(latex_expression)
        st.markdown("with respect to \(x\) is:")
        st.latex(latex_derivative)

        # Create numpy-compatible functions for plotting
        x_vals = np.linspace(-10, 10, 500)
        f_expr = sp.lambdify('x', sp.sympify(expression), 'numpy')
        f_prime_expr = sp.lambdify('x', sp.sympify(result), 'numpy')

        y_vals = f_expr(x_vals)

        # Add slider for x value
        x_slider = st.slider("Select x value to see the slope:", -10.0, 10.0, 0.0)

        # Compute the value of the derivative at the selected x value
        slope_at_x = f_prime_expr(x_slider)
        
        # Display the slope value
        st.write(f"The slope at x = {x_slider} is: {slope_at_x}")

        # Plot the function and its derivative along with the tangent line
        fig = go.Figure()

        # Plot original function
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='f(x)'))

        # Plot the tangent line at the selected x value
        tangent_line_x = np.array([x_slider - 5, x_slider + 5])  # Extend the tangent line range
        tangent_line_y = f_expr(x_slider) + slope_at_x * (tangent_line_x - x_slider)
        fig.add_trace(go.Scatter(x=tangent_line_x, y=tangent_line_y, mode='lines', name="Tangent Line",
                                line=dict(color='red', width=3)))

        # Add a marker at the selected x value on the function graph
        fig.add_trace(go.Scatter(
            x=[x_slider],
            y=[f_expr(x_slider)],
            mode="markers+text",
            marker=dict(size=10, color="red"),
            text=[f"Slope: {slope_at_x:.2f}"],
            textposition="top center",
            name=f"x = {x_slider}"
        ))

        # Update layout
        fig.update_layout(
            title="Function and Tangent Line",
            xaxis_title="x",
            yaxis_title="y",
        )
        
        # Display updated graph
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error in differentiation: {e}")

elif options == "Integration Solver":
    st.header("Numerical Integration Solver")
    
    # Expression input
    expression = st.text_input("Enter the function to integrate (e.g., sin(x))", "sin(x)")
    
    # Function for Integration
    a = st.number_input("Enter the lower bound:", value=0.0)
    b = st.number_input("Enter the upper bound:", value=np.pi)

    # Integration and plotting
    try:
        # Use sympy or numpy for handling sin(x) directly
        expression = expression.replace("sin(x)", "np.sin(x)").replace("cos(x)", "np.cos(x)")
        func = lambda x: eval(expression)
        result = integrate_function(func, a, b)
        st.success(f"Result of integration: {result}")

        # Plot the function
        x_vals = np.linspace(a, b, 500)
        y_vals = func(x_vals)
        
        fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Function"))
        
        # Shade the area under the curve
        fig.add_traces(go.Scatter(
            x=np.concatenate([x_vals, x_vals[::-1]]),
            y=np.concatenate([y_vals, np.zeros_like(y_vals)]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.3)',
            line=dict(color="rgba(255, 255, 255, 0)"),
            name="Shaded Area"
        ))

        fig.update_layout(
            title="Integration Area",
            xaxis_title="x",
            yaxis_title="f(x)",
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in integration: {e}")

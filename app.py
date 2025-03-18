import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import random
import cmath
import io
import logging
import requests
import re  # For regex-based preprocessing
import json  # For saving and loading sessions

# SciPy Libraries
from scipy.stats import norm, binom, poisson
from scipy.integrate import quad, solve_ivp
from scipy.special import erf, gamma, beta
from scipy.optimize import minimize, differential_evolution
from scipy import signal
from numpy.fft import fft, fftfreq

# Quantum Libraries
# from qiskit import QuantumCircuit, Aer, execute
# from qiskit.visualization import plot_bloch_multivector, plot_histogram

# Additional Libraries
from streamlit_ace import st_ace  # For symbolic input editor
import networkx as nx


# Import utility functions
from utils import (
    calculate_erf, calculate_gamma, calculate_beta,
    normal_pdf, binomial_pmf, poisson_pmf,
    symbolic_derivative, taylor_series_sin,
    integrate_function
)

# st.title("Mathematical Functions Calculator")


# st.header("Welcome to the Mathematical Functions Calculator!")
# st.write("You can now use all the features of the app.")

# Sidebar Menu
options = st.sidebar.radio(
    "Select a Tool:",
    [
        "Home page",
        "Plot Graph",
        #"Interactive Tutorials",
        #"Save and Load Sessions",
        "3D Plot",
        "Quantum Mechanics",
        #"Quantum Circuit",
        #"Quantum Entanglement Simulator",
        "Signal Processing and Optimization",
        "Interactive Graph Theory Visualization",
        "Mathematical Simulations with Interactivity",
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
# with st.sidebar.expander("Help"):
#     st.write("""
#    - I AM AARYAN.
#     """)

# Plot Graph Section
if options == "Plot Graph":
        
    # ---------------------------------------------------------
    # Make the page wide
    # ---------------------------------------------------------
    #st.set_page_config(layout="wide")

    # ---------------------------------------------------------
    # Session State Initialization
    # ---------------------------------------------------------
    if 'equation' not in st.session_state:
        st.session_state.equation = "x^2 + y^2 - 1"  # Default

    def update_equation():
        """
        Callback to store the text input into st.session_state.equation whenever it changes.
        """
        st.session_state.equation = st.session_state.equation_text_input

    # ---------------------------------------------------------
    # Page Title
    # ---------------------------------------------------------
    st.title("PLOT GRAPH")

    # ---------------------------------------------------------
    # Equation Input (placed above the graph)
    # ---------------------------------------------------------
    st.header("Enter Equation")
    st.text_input(
        label="Enter equation:",
        value=st.session_state.equation,
        key="equation_text_input",
        on_change=update_equation
    )

    # ---------------------------------------------------------
    # Equation Preprocessing & Parsing
    # ---------------------------------------------------------
    def preprocess_equation(equation: str) -> str:
        """
        Process the raw input to a sympy-compatible expression:
        - '^' -> '**'
        - Insert '*' for implied multiplication (e.g., 3x -> 3*x)
        - Convert |expr| to Abs(expr)
        - Replace standalone 'e' with exp(1)
        """
        try:
            # Replace '^' with '**'
            equation = equation.replace("^", "**")
            # Insert multiplication signs where needed
            equation = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', equation)
            equation = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', equation)
            # Replace absolute value notation with Sympy's Abs()
            equation = re.sub(r'\|(.*?)\|', r'Abs(\1)', equation)
            # Replace standalone 'e' with exp(1)
            equation = re.sub(r'\be\b', 'exp(1)', equation)
            return equation
        except Exception as error:
            st.error(f"Error processing equation: {error}")
            return None

    def parse_equation(equation: str):
        """
        Convert the preprocessed equation string into a Sympy expression.
        Allowed functions are defined for safe parsing.
        """
        try:
            processed_eq = preprocess_equation(equation)
            if processed_eq is None:
                return None
            
            # Define allowed symbols and functions
            x, y = sp.symbols('x y')
            allowed_functions = {
                'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                'log': sp.log, 'exp': sp.exp, 'sqrt': sp.sqrt,
                'pi': sp.pi, 'Abs': sp.Abs
            }
            
            # Parse the expression without auto-evaluation
            expr = sp.sympify(processed_eq, locals=allowed_functions, evaluate=False)
            
            # Check for collisions (e.g., using a function name as a variable)
            for symbol in expr.free_symbols:
                if str(symbol) in allowed_functions:
                    raise ValueError(
                        f"'{symbol}' is treated as a variable, but it conflicts with a function name."
                    )
            
            return expr
        except Exception as error:
            st.error(f"Invalid equation: {error}")
            return None

    # ---------------------------------------------------------
    # Parse the Equation and Plot
    # ---------------------------------------------------------
    expr = parse_equation(st.session_state.equation)

    if expr:
        try:
            # Gather free symbols (variables) from the expression
            variables = expr.free_symbols
            
            # Create sliders for extra variables (excluding x and y)
            slider_vars = {
                var: st.slider(
                    f"Value for {var}", -10.0, 10.0, 1.0, key=f"slider_{var}"
                )
                for var in variables if var not in [sp.Symbol('x'), sp.Symbol('y')]
            }
            
            # Substitute slider values into the expression
            expr_with_values = expr.subs(slider_vars)
            
            # --- Show the equation in LaTeX ---
            # If 'y' is in the expression, we interpret it as implicit: expr = 0
            # Otherwise, it's explicit: y = expr
            if sp.Symbol('y') in variables:
                # Implicit
                eq_latex = rf"{sp.latex(expr_with_values)} = 0"
            else:
                # Explicit
                eq_latex = rf"y = {sp.latex(expr_with_values)}"
            
            st.latex(eq_latex)  # Display the equation in LaTeX

            # Define sampling range for x (and y if needed)
            x_vals = np.linspace(-10, 10, 400)
            
            # If the expression contains y, treat it as an implicit equation F(x,y)=0
            if sp.Symbol('y') in variables:
                y_vals = np.linspace(-10, 10, 400)
                X, Y = np.meshgrid(x_vals, y_vals)
                
                # Convert the Sympy expression into a NumPy-callable function
                Z_func = sp.lambdify((sp.Symbol('x'), sp.Symbol('y')), expr_with_values, modules=["numpy"])
                Z = Z_func(X, Y)
                
                # Plot only the 0-level contour, with a red line and no fill
                fig = go.Figure(data=[go.Contour(
                    x=x_vals,
                    y=y_vals,
                    z=Z,
                    contours=dict(
                        start=0,
                        end=0,
                        size=1,
                        coloring='none',  # no fill
                        showlines=True
                    ),
                    line_color="red",
                    line_width=2,
                    showscale=False
                )])
            
            else:
                # If 'y' is not present, treat it as y = f(x)
                y_vals = sp.lambdify(sp.Symbol('x'), expr_with_values, modules=["numpy"])(x_vals)
                fig = go.Figure(data=go.Scatter(
                    x=x_vals, y=y_vals, mode="lines", name="Function",
                    line=dict(color='red', width=3)
                ))
            
            # Update plot layout (bigger figure)
            fig.update_layout(
                xaxis_title="x",
                yaxis_title="y",
                width=1000,
                height=700
            )
            
            # Show the figure, filling available width
            st.plotly_chart(fig, use_container_width=True)
            
            # ---------------------------------------------------------
            # Export Options
            # ---------------------------------------------------------
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Export PNG", key="export_png"):
                    fig.write_image("graph.png")
                    with open("graph.png", "rb") as file:
                        st.download_button("Download PNG", file, "graph.png", "image/png", key="download_png")
            with col2:
                if st.button("Export SVG", key="export_svg"):
                    fig.write_image("graph.svg")
                    with open("graph.svg", "rb") as file:
                        st.download_button("Download SVG", file, "graph.svg", "image/svg+xml", key="download_svg")
            with col3:
                if st.button("Export PDF", key="export_pdf"):
                    fig.write_image("graph.pdf")
                    with open("graph.pdf", "rb") as file:
                        st.download_button("Download PDF", file, "graph.pdf", "application/pdf", key="download_pdf")
        
        except Exception as error:
            st.error(f"Error in plotting: {error}")

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
    # ---------------------------------------------------------
    # Make page wide
    # ---------------------------------------------------------
    #st.set_page_config(layout="wide")

    # ---------------------------------------------------------
    # Helper: Insert Implicit Multiplication
    # ---------------------------------------------------------
    def insert_implicit_multiplication(eq_str: str) -> str:
        """
        Inserts multiplication symbols where needed.
        For example, turns "4x^2" into "4*x^2" and "2sin(x)" into "2*sin(x)".
        """
        # Insert * between a digit and a letter or an opening parenthesis.
        eq_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', eq_str)
        # Optionally, insert * between a closing parenthesis and a letter/digit:
        eq_str = re.sub(r'(\))([a-zA-Z(])', r'\1*\2', eq_str)
        return eq_str

    # ---------------------------------------------------------
    # Helper: Parse Full Equation
    # ---------------------------------------------------------
    def parse_full_equation(equation_str: str):
        """
        Parse the input equation string into a Sympy expression in the form F(x,y,z)=0.
        
        - If an '=' is present, the input is split into LHS and RHS and the final expression is LHS - RHS.
        - If no '=' is present:
            - For expressions involving only x and y, it is assumed that y = f(x), i.e. F(x,y) = y - f(x) = 0.
            - For expressions involving z, the equation is assumed to be implicit: F(x,y,z)=0.
        
        Returns a tuple (final_expr, display_str) where final_expr is the Sympy expression and
        display_str is a LaTeX string for display.
        """
        # First, insert implicit multiplication (e.g., turn "4x^2" into "4*x^2")
        eq_str = insert_implicit_multiplication(equation_str.strip())
        # Replace caret with power operator
        eq_str = eq_str.replace("^", "**")
        
        # Define allowed functions and symbols
        local_dict = {
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'log': sp.log, 'exp': sp.exp, 'sqrt': sp.sqrt,
            'Abs': sp.Abs
        }
        x, y, z = sp.symbols('x y z')
        
        if "=" in eq_str:
            parts = eq_str.split("=")
            if len(parts) != 2:
                st.error("Invalid equation: more than one '=' operator detected.")
                return None, None
            try:
                lhs = sp.sympify(parts[0].strip(), locals=local_dict)
                rhs = sp.sympify(parts[1].strip(), locals=local_dict)
            except Exception as e:
                st.error(f"Error parsing equation parts: {e}")
                return None, None
            final_expr = lhs - rhs
            display_str = sp.latex(lhs) + " = " + sp.latex(rhs)
            return final_expr, display_str
        else:
            try:
                expr = sp.sympify(eq_str, locals=local_dict)
            except Exception as e:
                st.error(f"Error parsing equation: {e}")
                return None, None
            if z in expr.free_symbols:
                # Implicit equation in 3 variables assumed to be f(x,y,z)=0
                final_expr = expr
                display_str = sp.latex(expr) + " = 0"
            else:
                # For only x and y, assume explicit: y = f(x)
                final_expr = y - expr
                display_str = "y = " + sp.latex(expr)
            return final_expr, display_str

    # ---------------------------------------------------------
    # Page Title & Instructions
    # ---------------------------------------------------------
    st.header("3D Graph Plotter")

    with st.expander("How to Use"):
        st.write(
            """
            - **Enter an Equation:**  
                You may include an equal sign (`=`) to specify exactly what the equation equals.  
                For example:
                - `4x^2 = y`  will be interpreted as y - 4*x**2 = 0.  
                - `x^2 + y^2 = z^2 + 1` will be interpreted as x^2 + y^2 - (z^2 + 1) = 0.
            - If you do **not** include an equal sign:  
                - For equations involving only `x` and `y`, the app assumes the equation is of the form **y = f(x)**.  
                - For equations involving `x`, `y`, and `z`, it assumes an implicit function **f(x, y, z) = 0**.
            
            - **Syntax Guidelines:**  
                - Use `**` for exponents (e.g., `x**2` for x²).  
                - You no longer need to explicitly insert `*` for multiplication; for example, you can enter `4x^2` and it will be interpreted as `4*x^2`.  
                - Standard math functions (like `sin`, `cos`, etc.) are available.
            
            - **Viewing & Exporting:**  
                - The 3D graph will update automatically based on your input.
                - Use the export button to save the graph as a PNG.
            """
        )

    # ---------------------------------------------------------
    # Equation Input (placed above the graph)
    # ---------------------------------------------------------
    equation_input = st.text_input(
        "Enter the 3D equation:",
        "(x^2)/4 + y^2 - (z^2)/2 - 1"  # Default example
    )

    final_expr, display_str = parse_full_equation(equation_input)

    if final_expr is not None:
        # Display the equation nicely using LaTeX
        st.latex(display_str)
        
        try:
            # Reduced data: smaller grid resolution and domain
            grid_points = 60
            domain_min, domain_max = -20, 20
            
            # Determine if this is a 3D implicit plot (involving z) or explicit (only x,y)
            x_sym, y_sym, z_sym = sp.symbols('x y z')
            if z_sym in final_expr.free_symbols:
                # 3D Implicit Isosurface Plot
                x_vals = np.linspace(domain_min, domain_max, grid_points)
                y_vals = np.linspace(domain_min, domain_max, grid_points)
                z_vals = np.linspace(domain_min, domain_max, grid_points)
                X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
                
                # Lambdify the expression for numerical evaluation
                f_func = sp.lambdify((x_sym, y_sym, z_sym), final_expr, 'numpy')
                F = f_func(X, Y, Z)
                
                # Plot 0-level isosurface using a colorscale (Inferno)
                fig = go.Figure(data=go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=F.flatten(),
                    isomin=0,
                    isomax=0,
                    surface_count=1,
                    colorscale="Inferno",
                    opacity=0.8,
                    showscale=False,
                    caps=dict(x_show=False, y_show=False, z_show=False)
                ))
            else:
                # Explicit Surface: y = f(x) interpreted as y - f(x)=0, so plot f(x) as a surface
                x_vals = np.linspace(domain_min, domain_max, grid_points)
                y_vals = np.linspace(domain_min, domain_max, grid_points)
                X, Y = np.meshgrid(x_vals, y_vals)
                
                f_func = sp.lambdify((x_sym, y_sym), final_expr, 'numpy')
                Z = f_func(X, Y)
                
                fig = go.Figure(data=[go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale="Viridis",
                    opacity=0.8,
                    showscale=False
                )])
            
            # Update layout with the reduced domain range
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title="x", range=[domain_min, domain_max]),
                    yaxis=dict(title="y", range=[domain_min, domain_max]),
                    zaxis=dict(title="z", range=[domain_min, domain_max])
                ),
                width=1000,
                height=700,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export Option
            if st.button("Export 3D Graph as PNG"):
                fig.write_image("3d_graph.png")
                st.success("3D Graph exported as 3d_graph.png")
        
        except Exception as e:
            st.error(f"Error in plotting: {e}")

    # Call the function to run the app

elif options == "Equation Solver":
    
    def export_csv(data, filename="data.csv"):
        """
        Convert a pandas DataFrame to CSV bytes and provide a download button.
        """
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Plot Data as CSV", data=csv, file_name=filename)

    def export_steps(text, filename="detailed_steps.txt"):
        """
        Provide a download button to export detailed steps as a text file.
        """
        return st.download_button("Download Detailed Steps", data=text, file_name=filename)

    def process_equation(eq):
        """
        Process the input equation string:
        - Replace caret (^) with Python’s power operator (**).
        - Insert multiplication symbols where needed (e.g., convert "4x^2" to "4*x^2").
        """
        eq = eq.replace("^", "**")
        # Insert multiplication between a digit and a letter or opening parenthesis (e.g., 4x -> 4*x)
        eq = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', eq)
        # Also, between a letter and a digit (e.g., x2 -> x*2)
        eq = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', eq)
        return eq

    def generate_steps(original, processed, expr):
        """
        Generate a text string containing detailed steps.
        """
        steps = f"Original Equation:\n{original}\n\n"
        steps += f"Processed Equation:\n{processed}\n\n"
        steps += f"Expression (f(x)=0):\n{sp.pretty(expr)}\n\n"
        try:
            simplified = sp.simplify(expr)
            steps += f"Simplified Expression:\n{sp.pretty(simplified)}\n\n"
        except Exception as e:
            steps += "Simplification failed.\n\n"
        try:
            factored = sp.factor(expr)
            steps += f"Factorized Expression:\n{sp.pretty(factored)}\n\n"
        except Exception as e:
            steps += "Factorization failed.\n\n"
        return steps

    # =============================================================================
    # Main App: Equation Solver
    # =============================================================================

    st.header("Equation Solver")

    st.markdown("""
    This app allows you to solve an equation (for example, `x^2 - 4 = 0`).  
    You can use the caret (`^`) for exponents and omit multiplication symbols (so you can type `4x^2` instead of `4*x^2`).  
    If an equal sign is included, the app solves the equation in the form *LHS - RHS = 0*.  
    You can also view detailed processing steps and download them.
    """)

    # Input Section
    equation_input = st.text_input("Enter the equation to solve (e.g., x^2 - 4 = 0):", "x^2 - 4 = 0")
    variable = st.text_input("Enter variable (default: x):", "x").strip()
    show_steps = st.checkbox("Show Detailed Steps", value=False)

    try:
        # Define the symbolic variable
        sym_var = sp.symbols(variable)

        # Process the input equation
        eq_processed = process_equation(equation_input.strip())

        # Parse the equation: if "=" exists, split into LHS and RHS; otherwise, assume f(x)=0.
        if "=" in eq_processed:
            parts = eq_processed.split("=")
            if len(parts) != 2:
                st.error("Invalid equation: more than one '=' sign detected.")
                raise Exception("Invalid equation format.")
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            expr = sp.sympify(f"({lhs}) - ({rhs})", locals={"E": sp.E, "pi": sp.pi})
        else:
            expr = sp.sympify(eq_processed, locals={"E": sp.E, "pi": sp.pi})

        # If detailed steps are enabled, show original, processed, simplified, and factorized forms.
        steps_text = ""
        if show_steps:
            st.markdown("#### Detailed Steps")
            st.markdown("**Original Equation:**")
            st.code(equation_input)
            st.markdown("**Processed Equation:**")
            st.code(eq_processed)
            st.markdown("**Expression (f(x)=0):**")
            st.latex(sp.latex(expr))
            try:
                simplified_expr = sp.simplify(expr)
                st.markdown("**Simplified Expression:**")
                st.latex(sp.latex(simplified_expr))
            except Exception as e:
                st.warning("Simplification failed.")
            try:
                factored_expr = sp.factor(expr)
                st.markdown("**Factorized Expression:**")
                st.latex(sp.latex(factored_expr))
            except Exception as e:
                st.warning("Factorization failed.")

            steps_text = generate_steps(equation_input, eq_processed, expr)
            export_steps(steps_text)

        # Solve the equation symbolically
        solutions = sp.solve(expr, sym_var, dict=True)

        real_solutions = []
        complex_solutions = []
        for sol in solutions:
            sol_val = sol[sym_var].evalf()
            if sol_val.is_real:
                real_solutions.append(float(sol_val))
            else:
                complex_solutions.append(sol_val)

        # If no real solutions, try numerical solving as fallback.
        if not real_solutions:
            x0 = st.number_input("Enter initial guess for numerical solution:", value=0.0)
            try:
                numerical_sol = sp.nsolve(expr, sym_var, x0)
                if numerical_sol.is_real:
                    real_solutions.append(float(numerical_sol))
                else:
                    complex_solutions.append(numerical_sol)
            except Exception as e:
                st.error(f"Numerical solving failed: {e}")

        st.markdown("### Equation")
        st.latex(f"{sp.latex(expr)} = 0")

        if real_solutions or complex_solutions:
            st.markdown("### Solutions")
            if real_solutions:
                for i, sol in enumerate(real_solutions, 1):
                    st.success(f"Real Solution {i}: {sol:.4f}")
            if complex_solutions:
                st.warning("Complex Solutions Found:")
                for i, sol in enumerate(complex_solutions, 1):
                    st.info(f"Complex Solution {i}: {sol}")

            # Plotting: Set domain that includes solutions
            domain_min = min(-10, *real_solutions) - 1
            domain_max = max(10, *real_solutions) + 1
            x_vals = np.linspace(domain_min, domain_max, 400)
            y_func = sp.lambdify(sym_var, expr, "numpy")
            try:
                y_vals = y_func(x_vals)
            except Exception as e:
                st.error(f"Error evaluating function for plotting: {e}")
                y_vals = np.zeros_like(x_vals)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=f"${sp.latex(expr)}$"))
            for sol in real_solutions:
                fig.add_trace(go.Scatter(
                    x=[sol], y=[0], mode="markers+text",
                    marker=dict(color="red", size=10),
                    text=[f"Root: {sol:.2f}"],
                    textposition="top center",
                    name=f"Solution: {sol:.2f}"
                ))
            fig.update_layout(
                title=f"Graph of ${sp.latex(expr)} = 0$",
                xaxis_title="x",
                yaxis_title="f(x)",
                xaxis=dict(zeroline=True),
                yaxis=dict(zeroline=True)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Export plot data as CSV
            df_plot = pd.DataFrame({"x": x_vals, "f(x)": y_vals})
            export_csv(df_plot, "equation_plot_data.csv")
        else:
            st.warning("No solutions found.")

    except Exception as e:
        st.error(f"Error in solving equation: {str(e)}")

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

elif options == "Quantum Mechanics":

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

    import streamlit as st

    st.title("Quantum Mechanics")

    # Dropdown Menu for Navigation
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

    choice = st.selectbox("Select a Page:", pages)

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

elif options == "Signal Processing and Optimization":
    
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
    import streamlit as st

    st.title("Signal Processing & Optimization")

    pages = [
        "Fourier Transform",
        "Spectrogram",
        "Optimization",
        "Time Evolution",
        "Export Data"
    ]

    choice = st.selectbox("Select a Page:", pages)

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

elif options == "Interactive Graph Theory Visualization":

    # Set full-width layout
    st.set_page_config(layout="wide")

    # Function to generate different types of graphs
    def generate_graph(graph_type, num_nodes, prob, edges_per_node):
        if graph_type == "Erdős–Rényi Random Graph":
            G = nx.erdos_renyi_graph(num_nodes, prob)
        elif graph_type == "Barabási–Albert Scale-Free Graph":
            G = nx.barabasi_albert_graph(num_nodes, edges_per_node)
        elif graph_type == "Grid Graph":
            size = int(num_nodes ** 0.5)
            G = nx.grid_2d_graph(size, size)
        else:
            G = nx.erdos_renyi_graph(num_nodes, 0.3)  # Default fallback
        return G

    # Function to visualize the graph interactively
    def plot_graph(G, layout_type, node_color, edge_color):
        pos = getattr(nx, layout_type + '_layout')(G) if hasattr(nx, layout_type + '_layout') else nx.spring_layout(G)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color=edge_color), mode='lines', hoverinfo='none')
        
        node_x, node_y = [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers',
            marker=dict(size=10, color=node_color, line=dict(width=2, color='black')),
            text=[f'Node {n}' for n in G.nodes()], hoverinfo='text'
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            width=None,  # Let it auto-adjust to full width
            height=750  # Slightly larger height for better appearance
        )
        st.plotly_chart(fig, use_container_width=True)

    # Streamlit UI
    st.title("Interactive Graph Theory Visualization")

    graph_type = st.selectbox("Choose Graph Type", [
        "Erdős–Rényi Random Graph", "Barabási–Albert Scale-Free Graph", "Grid Graph"
    ])
    num_nodes = st.slider("Number of Nodes", min_value=5, max_value=100, value=20)
    prob = st.slider("Edge Probability (for Erdős–Rényi)", min_value=0.01, max_value=1.0, value=0.3) if graph_type == "Erdős–Rényi Random Graph" else None
    edges_per_node = st.slider("Edges per Node (for Barabási–Albert)", min_value=1, max_value=10, value=2) if graph_type == "Barabási–Albert Scale-Free Graph" else None

    layout_type = st.selectbox("Graph Layout", ["spring", "circular", "random"])
    node_color = st.color_picker("Pick Node Color", "#1f77b4")
    edge_color = st.color_picker("Pick Edge Color", "#888888")

    G = generate_graph(graph_type, num_nodes, prob, edges_per_node)
    plot_graph(G, layout_type, node_color, edge_color)

elif options == "Mathematical Simulations with Interactivity":

    # Styling
    sns.set_style("darkgrid")
    st.title("Mathematical Simulations with Interactivity")

    # Gradient Descent
    def gradient_descent(f, grad_f, lr, epochs, x0):
        x = x0
        history = [x]
        for _ in range(epochs):
            x -= lr * grad_f(x)
            history.append(x)
        return history

    def f(x):
        return x**2

    def grad_f(x):
        return 2*x

    # Simulated Annealing
    def simulated_annealing(f, x0, temp, cooling, iterations):
        x = x0
        history = [x]
        for i in range(iterations):
            new_x = x + np.random.uniform(-1, 1)
            delta = f(new_x) - f(x)
            if delta < 0 or np.exp(-delta / temp) > np.random.rand():
                x = new_x
            history.append(x)
            temp *= cooling
        return history

    # Genetic Algorithm
    def genetic_algorithm(f, population_size, generations, mutation_rate):
        population = np.random.uniform(-10, 10, population_size)
        for _ in range(generations):
            fitness = np.array([f(x) for x in population])
            parents = population[np.argsort(fitness)[:population_size // 2]]
            offspring = parents + np.random.uniform(-1, 1, parents.shape) * mutation_rate
            population = np.concatenate((parents, offspring))
        return min(population, key=f)

    # Differential Equation Solver
    def solve_ode(y0, t_span, func):
        sol = solve_ivp(func, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 100))
        return sol.t, sol.y[0]

    def simple_ode(t, y):
        return -y + np.sin(t)

    # Vector Field Plotter
    def plot_vector_field():
        fig, ax = plt.subplots()
        x, y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        u, v = -y, x
        ax.quiver(x, y, u, v)
        ax.set_title("Vector Field")
        st.pyplot(fig)

    # Complex Analysis Visualization
    def plot_complex_function():
        fig, ax = plt.subplots()
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j*Y
        W = np.exp(Z)
        ax.contourf(X, Y, np.abs(W), cmap='coolwarm')
        ax.set_title("Complex Function Visualization: e^z")
        st.pyplot(fig)

    # UI Controls
    option = st.selectbox("Choose a simulation", [
        "Gradient Descent", "Simulated Annealing", "Genetic Algorithm",
        "Differential Equation Solver", "Vector Field", "Complex Analysis"])

    if option == "Gradient Descent":
        st.write("Gradient Descent is an optimization algorithm used to minimize functions by iteratively moving in the direction of the negative gradient.")
        lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        epochs = st.slider("Epochs", 10, 100, 50, 5)
        x0 = st.number_input("Initial x", value=5.0)
        history = gradient_descent(f, grad_f, lr, epochs, x0)
        fig, ax = plt.subplots()
        ax.plot(history, [f(x) for x in history], 'ro-')
        ax.set_title("Gradient Descent Optimization")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("f(x)")
        st.pyplot(fig)

    elif option == "Simulated Annealing":
        st.write("Simulated Annealing is a probabilistic optimization technique inspired by the annealing process in metallurgy, used to find global minima of functions.")
        temp = st.slider("Initial Temperature", 1, 200, 100, 5)
        cooling = st.slider("Cooling Rate", 0.5, 1.0, 0.9, 0.01)
        iterations = st.slider("Iterations", 10, 200, 100, 10)
        x0 = st.number_input("Initial x", value=5.0)
        history = simulated_annealing(f, x0, temp, cooling, iterations)
        fig, ax = plt.subplots()
        ax.plot(history, [f(x) for x in history], 'bo-')
        ax.set_title("Simulated Annealing")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("f(x)")
        st.pyplot(fig)

    elif option == "Genetic Algorithm":
        st.write("Genetic Algorithm is an evolutionary optimization method that mimics natural selection to find the optimal solution.")
        pop_size = st.slider("Population Size", 5, 50, 10, 1)
        generations = st.slider("Generations", 10, 100, 50, 5)
        mutation_rate = st.slider("Mutation Rate", 0.01, 1.0, 0.1, 0.01)
        result = genetic_algorithm(f, pop_size, generations, mutation_rate)
        st.write(f"Genetic Algorithm found minimum at x = {result}")

    elif option == "Differential Equation Solver":
        st.write("This solver numerically solves ordinary differential equations (ODEs) using SciPy's solve_ivp method.")
        t_span = (0, st.slider("Time Span (end)", 1, 20, 10, 1))
        y0 = st.number_input("Initial Condition y0", value=1.0)
        t, y = solve_ode([y0], t_span, simple_ode)
        fig, ax = plt.subplots()
        ax.plot(t, y)
        ax.set_title("Differential Equation Solution")
        ax.set_xlabel("Time")
        ax.set_ylabel("y(t)")
        st.pyplot(fig)

    elif option == "Vector Field":
        st.write("A vector field is a mathematical representation of vector quantities at different points in space, often used in physics and engineering.")
        plot_vector_field()

    elif option == "Complex Analysis":
        st.write("This visualization demonstrates complex functions by mapping them onto a 2D plane using contour plots.")
        plot_complex_function()

elif options == "Quantum Circuit":
    def create_quantum_circuit(gates):
        qc = QuantumCircuit(1)  # Single qubit circuit
        for gate in gates:
            if gate == "Hadamard":
                qc.h(0)
            elif gate == "X (Pauli-X)":
                qc.x(0)
            elif gate == "Y (Pauli-Y)":
                qc.y(0)
            elif gate == "Z (Pauli-Z)":
                qc.z(0)
            elif gate == "S":
                qc.s(0)
            elif gate == "T":
                qc.t(0)
        return qc

    def simulate_circuit(qc):
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        result = job.result()
        statevector = result.get_statevector()
        return statevector

    def measure_circuit(qc):
        qc.measure_all()
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1024)
        result = job.result()
        counts = result.get_counts()
        return counts

    def main():
        st.title("Quantum Circuit Simulator")
        
        gate_options = ["Hadamard", "X (Pauli-X)", "Y (Pauli-Y)", "Z (Pauli-Z)", "S", "T"]
        selected_gates = st.multiselect("Select quantum gates to apply:", gate_options)
        
        if st.button("Simulate"): 
            qc = create_quantum_circuit(selected_gates)
            statevector = simulate_circuit(qc)
            counts = measure_circuit(qc)
            
            st.subheader("Quantum Circuit")
            st.text(qc.draw())
            
            st.subheader("State Vector Visualization")
            fig, ax = plt.subplots()
            plot_bloch_multivector(statevector, ax=ax)
            st.pyplot(fig)
            
            st.subheader("Measurement Probabilities")
            fig, ax = plt.subplots()
            plot_histogram(counts, ax=ax)
            st.pyplot(fig)
            
    if __name__ == "__main__":
        main()

elif options == "Quantum Entanglement Simulator":
    def create_entangled_pair():
        qc = QuantumCircuit(2, 2)
        qc.h(0)  # Hadamard on qubit 0
        qc.cx(0, 1)  # CNOT gate entangles qubits
        return qc

    def measure_qubits(qc, basis):
        if basis == 'Z':  # Standard Z-basis measurement
            pass  # No change needed
        elif basis == 'X':  # X-basis measurement
            qc.h([0, 1])  # Apply Hadamard to change basis
        
        qc.measure([0, 1], [0, 1])
        simulator = Aer.get_backend('qasm_simulator')
        result = execute(qc, simulator, shots=1024).result()
        counts = result.get_counts()
        return counts

    def main():
        st.title("Quantum Entanglement Simulator")
        st.write("This simulation demonstrates entangled qubits and Bell test results.")
        
        # User selects measurement basis
        basis = st.radio("Select measurement basis:", ('Z', 'X'))
        
        # Create entangled qubit pair
        qc = create_entangled_pair()
        
        # Measure in selected basis
        if st.button("Run Measurement"):
            results = measure_qubits(qc, basis)
            st.write("Measurement Outcomes:")
            st.bar_chart(results)
            
            # Checking for Bell Test violations
            if basis == 'X':
                st.write("In the X-basis, a strong correlation indicates quantum entanglement.")
            else:
                st.write("In the Z-basis, classical correlations can still appear.")

    if __name__ == "__main__":
        main()

elif options == "Home page":
    
    import streamlit as st

    # 1. Must be the first Streamlit command:
    #st.set_page_config(page_title="Graphium", layout="wide")

    # 2. Custom CSS to hide the default header/menu and footer
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}     /* Hide the hamburger menu */
    header {visibility: hidden;}        /* Hide the default header */
    footer {visibility: hidden;}        /* Hide the 'Made with Streamlit' footer */
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # 3. Main Page Content
    st.markdown(
        """
        <h1 style="text-align: center; font-family: 'Times New Roman', Times, serif; margin-top: 20px; font-size: 3.5rem;">
        Graphium
        </h1>
        <h2 style="text-align: center; color: #ccc; margin-bottom: 2rem; font-size: 1.75rem;">
        Where Mathematics Meets Visualization
        </h2>
        <p style="color: #ddd; font-size: 1.1rem; max-width: 800px; margin: auto; line-height: 1.6;">
        Graphium is an advanced mathematical visualization platform designed for students,
        researchers, and professionals. It bridges the gap between abstract mathematical concepts
        and interactive, dynamic representations.
        </p>

        <h3 style="color: #fff; margin-top: 2rem; font-size: 1.8rem;">
        Why Graphium?
        </h3>
        <ul style="color: #ddd; font-size: 1.1rem; line-height: 1.6;">
        <li><span style="color: #4CAF50; font-weight: 600;">Intuitive Visualizations:</span> Explore mathematical structures with clarity.</li>
        <li><span style="color: #4CAF50; font-weight: 600;">Advanced Computation:</span> Leverage real-time computation for precise results.</li>
        <li><span style="color: #4CAF50; font-weight: 600;">Multi-Disciplinary Utility:</span> Useful for Physics, Engineering, Economics, and more.</li>
        <li><span style="color: #4CAF50; font-weight: 600;">Customization & Interactivity:</span> Modify parameters and observe instant updates.</li>
        </ul>

        <h3 style="color: #fff; margin-top: 2rem; font-size: 1.8rem;">
        Key Features
        </h3>
        <ul style="color: #ddd; font-size: 1.1rem; line-height: 1.6;">
        <li><span style="color: #4CAF50; font-weight: 600;">Graph Plotting:</span> 2D & 3D function plotting with smooth rendering.</li>
        <li><span style="color: #4CAF50; font-weight: 600;">Quantum Mechanics Simulations:</span> Visualize wavefunctions, probability densities, and quantum states.</li>
        <li><span style="color: #4CAF50; font-weight: 600;">Signal Processing & Optimization:</span> Fourier transforms, spectrograms, and optimization techniques.</li>
        <li><span style="color: #4CAF50; font-weight: 600;">Higher-Dimensional Representations:</span> Tesseracts, fractals, and chaotic systems brought to life.</li>
        </ul>
        <p style="color: #ddd; font-size: 1.1rem; max-width: 800px; margin: auto; line-height: 1.6;">
        Graphium transforms complex mathematical concepts into tangible, interactive experiences,
        making learning and research more insightful than ever.
        </p>

        <hr style="margin-top: 3rem; border: none; border-top: 1px solid #333;" />
        <p style="text-align: center; color: #aaa; margin-top: 1rem;">
        Graphium – A new dimension to mathematics.
        </p>
        """,
        unsafe_allow_html=True
    )

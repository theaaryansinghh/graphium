import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import re

def plot_graph():
    # ---------------------------------------------------------
    # Make the page wide
    # ---------------------------------------------------------
    st.set_page_config(layout="wide")

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

plot_graph()

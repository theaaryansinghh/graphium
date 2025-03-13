import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import re


def threedplot():
    # ---------------------------------------------------------
    # Make page wide
    # ---------------------------------------------------------
    st.set_page_config(layout="wide")

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
threedplot()

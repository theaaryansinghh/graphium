import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import re
import pandas as pd

# =============================================================================
# Helper Functions
# =============================================================================

def eq_solver():

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

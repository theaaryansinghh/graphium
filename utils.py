import numpy as np
import sympy as sp
from scipy.stats import norm, binom, poisson
from scipy.integrate import quad
from scipy.special import erf, gamma, beta
from typing import List, Tuple
import math

# Error Function Calculation
def calculate_erf(x: float) -> float:
    try:
        return float(erf(x))
    except (ValueError, TypeError):
        raise ValueError("Invalid input for error function calculation")

# Gamma Function
def calculate_gamma(x: float) -> float:
    try:
        return float(gamma(x))
    except (ValueError, TypeError):
        raise ValueError("Invalid input for gamma function calculation")

# Beta Function
def calculate_beta(x: float, y: float) -> float:
    try:
        return float(beta(x, y))
    except (ValueError, TypeError):
        raise ValueError("Invalid input for beta function calculation")

# Normal Distribution PDF
def normal_pdf(x: float, mean: float, stddev: float) -> float:
    try:
        return float(norm.pdf(x, mean, stddev))
    except (ValueError, TypeError):
        raise ValueError("Invalid input for normal distribution")

# Binomial Distribution PMF
def binomial_pmf(k: int, n: int, p: float) -> float:
    try:
        return float(binom.pmf(k, n, p))
    except (ValueError, TypeError):
        raise ValueError("Invalid input for binomial distribution")

# Poisson Distribution PMF
def poisson_pmf(k: int, lam: float) -> float:
    try:
        return float(poisson.pmf(k, lam))
    except (ValueError, TypeError):
        raise ValueError("Invalid input for Poisson distribution")

# Numerical Integration (Trapezoidal Rule)
def integrate_function(func, a: float, b: float) -> float:
    result, _ = quad(func, a, b)
    return result

# Symbolic Derivatives using SymPy
def symbolic_derivative(expression: str) -> str:
    x = sp.symbols('x')
    func = sp.sympify(expression)
    derivative = sp.diff(func, x)
    return str(derivative)

# Fourier Transform
def fourier_transform(expression: str) -> str:
    x = sp.symbols('x')
    func = sp.sympify(expression)
    transform = sp.fourier_transform(func, x, sp.symbols('k'))
    return str(transform)

# Linear Regression
def linear_regression(x_vals: List[float], y_vals: List[float]) -> Tuple[float, float]:
    A = np.vstack([x_vals, np.ones(len(x_vals))]).T
    m, b = np.linalg.lstsq(A, y_vals, rcond=None)[0]
    return m, b

# Taylor Series Expansion for Sin Function
def taylor_series_sin(x: float, terms: int = 5) -> float:
    result = 0
    for n in range(terms):
        result += ((-1) ** n) * (x ** (2 * n + 1)) / math.factorial(2 * n + 1)
    return result

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
from scipy.integrate import solve_ivp
import random
import cmath

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

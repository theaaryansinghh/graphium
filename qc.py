import streamlit as st
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import matplotlib.pyplot as plt

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

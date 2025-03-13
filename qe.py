import streamlit as st
import numpy as np
from qiskit import QuantumCircuit, Aer, execute

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

import streamlit as st
import networkx as nx
import plotly.graph_objects as go

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

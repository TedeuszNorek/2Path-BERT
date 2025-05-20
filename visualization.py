import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

def create_relationship_graph_figure(
    graph: nx.DiGraph,
    community_map: Optional[Dict[str, int]] = None,
    highlight_nodes: Optional[List[str]] = None,
    highlight_edges: Optional[List[Tuple[str, str]]] = None,
    title: str = "Semantic Relationship Graph"
) -> go.Figure:
    """
    Create a Plotly figure for visualizing relationship graphs.
    
    Args:
        graph: NetworkX DiGraph object
        community_map: Dict mapping nodes to community IDs
        highlight_nodes: List of nodes to highlight
        highlight_edges: List of edges (as tuples) to highlight
        title: Title for the figure
        
    Returns:
        Plotly figure object
    """
    # Set default values if None is provided
    if community_map is None:
        community_map = {}
    
    if highlight_nodes is None:
        highlight_nodes = []
    
    if highlight_edges is None:
        highlight_edges = []
    
    # Create a position attribute if it doesn't exist
    if not any('pos' in graph.nodes[node] for node in graph.nodes):
        pos = nx.spring_layout(graph, seed=42)
        nx.set_node_attributes(graph, {node: position for node, position in pos.items()}, 'pos')
    
    # Get node positions
    pos = {node: data.get('pos', [0, 0]) for node, data in graph.nodes(data=True)}
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    # Assign colors based on communities
    default_color = 0
    
    # Calculate node sizes based on degree
    degrees = dict(graph.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_size, max_size = 10, 30
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        # Size based on degree
        size = min_size + (degrees[node] / max_degree) * (max_size - min_size)
        node_size.append(size)
        
        # Color based on community or default
        color = community_map.get(node, default_color)
        node_color.append(color)
    
    # Highlight specific nodes if requested
    highlighted_x = []
    highlighted_y = []
    highlighted_text = []
    
    for node in highlight_nodes:
        if node in graph.nodes():
            x, y = pos[node]
            highlighted_x.append(x)
            highlighted_y.append(y)
            highlighted_text.append(node)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    # Create highlighted edge traces
    highlighted_edge_x = []
    highlighted_edge_y = []
    highlighted_edge_text = []
    
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_data = graph.get_edge_data(edge[0], edge[1])
        predicate = edge_data.get('predicate', '')
        confidence = edge_data.get('confidence', 0.0)
        polarity = edge_data.get('polarity', 'neutral')
        
        # Edge label
        edge_label = f"{edge[0]} --[{predicate}]--> {edge[1]}<br>Confidence: {confidence:.2f}<br>Polarity: {polarity}"
        
        # Check if this is a highlighted edge
        if edge in highlight_edges:
            highlighted_edge_x.extend([x0, x1, None])
            highlighted_edge_y.extend([y0, y1, None])
            highlighted_edge_text.append(edge_label)
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(edge_label)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, 
        line=dict(width=0.8, color='#cccccc'),
        hoverinfo='text',
        text=edge_text,
        mode='lines',
        name='Relationships'
    ))
    
    # Add highlighted edges
    if highlighted_edge_x:
        fig.add_trace(go.Scatter(
            x=highlighted_edge_x, y=highlighted_edge_y, 
            line=dict(width=2, color='red'),
            hoverinfo='text',
            text=highlighted_edge_text,
            mode='lines',
            name='Highlighted Relationships'
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='Viridis',
            line_width=2,
            line=dict(color='white')
        ),
        name='Entities'
    ))
    
    # Add highlighted nodes
    if highlighted_x:
        fig.add_trace(go.Scatter(
            x=highlighted_x, y=highlighted_y,
            mode='markers',
            hoverinfo='text',
            text=highlighted_text,
            marker=dict(
                size=node_size,
                color='red',
                symbol='star',
                line_width=2
            ),
            name='Highlighted Entities'
        ))
    
    # Add arrows for directed edges
    arrow_x = []
    arrow_y = []
    
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Calculate arrow position (80% along the edge)
        dx = x1 - x0
        dy = y1 - y0
        arrow_x.append(x0 + 0.8 * dx)
        arrow_y.append(y0 + 0.8 * dy)
    
    # Update layout
    fig.update_layout(
        title=title,
        title_font_size=16,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_heatmap_figure(
    matrix: np.ndarray,
    labels: List[str],
    title: str = "Relationship Adjacency Matrix"
) -> go.Figure:
    """
    Create a heatmap figure for visualizing relationship matrices.
    
    Args:
        matrix: 2D numpy array containing the adjacency matrix
        labels: List of entity labels
        title: Title for the figure
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        hoverongaps=False,
        colorbar=dict(title="Confidence")
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="Target Entity"),
        yaxis=dict(title="Source Entity"),
    )
    
    return fig

def create_bar_chart(
    labels: List[str],
    values: List[float],
    title: str = "Distribution",
    x_title: str = "Category",
    y_title: str = "Count"
) -> go.Figure:
    """
    Create a bar chart figure.
    
    Args:
        labels: List of category labels
        values: List of corresponding values
        title: Title for the figure
        x_title: X-axis title
        y_title: Y-axis title
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure(data=go.Bar(
        x=labels,
        y=values,
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title),
    )
    
    return fig

def create_sankey_diagram(
    graph: nx.DiGraph,
    weight_attr: str = 'weight',
    title: str = "Relationship Flow Diagram"
) -> go.Figure:
    """
    Create a Sankey diagram for visualizing relationship flows.
    
    Args:
        graph: NetworkX DiGraph object
        weight_attr: Edge attribute to use for flow weights
        title: Title for the figure
        
    Returns:
        Plotly figure object
    """
    # Extract nodes
    nodes = list(graph.nodes())
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    # Prepare Sankey data
    sources = []
    targets = []
    values = []
    labels = []
    
    # Add nodes to labels
    for node in nodes:
        labels.append(node)
    
    # Add edges
    for source, target, data in graph.edges(data=True):
        sources.append(node_indices[source])
        targets.append(node_indices[target])
        
        # Get weight, default to 1 if not found
        weight = data.get(weight_attr, 1)
        values.append(weight)
    
    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="blue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    
    # Update layout
    fig.update_layout(
        title=title,
        font=dict(size=12)
    )
    
    return fig
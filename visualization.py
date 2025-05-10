import plotly.graph_objects as go
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple
import colorsys

def generate_color_palette(n: int) -> List[str]:
    """
    Generate a visually distinct color palette.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of hexadecimal color codes
    """
    colors = []
    for i in range(n):
        # Use HSV color space for even distribution
        h = i / n
        s = 0.7
        v = 0.95
        
        # Convert to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Convert to hex
        hex_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        colors.append(hex_color)
        
    return colors

def create_relationship_graph_figure(G: nx.DiGraph, 
                                   community_map: Dict[str, int] = None, 
                                   node_size_factor: float = 20,
                                   edge_width_factor: float = 2,
                                   highlight_nodes: List[str] = None,
                                   highlight_edges: List[Tuple[str, str]] = None,
                                   title: str = "Semantic Relationship Graph") -> go.Figure:
    """
    Create a Plotly figure to visualize the relationship graph.
    
    Args:
        G: NetworkX DiGraph
        community_map: Optional mapping of nodes to communities
        node_size_factor: Factor to scale node sizes
        edge_width_factor: Factor to scale edge widths
        highlight_nodes: List of node IDs to highlight
        highlight_edges: List of edge tuples to highlight
        title: Figure title
        
    Returns:
        Plotly figure
    """
    # Get node positions - if not present, compute using spring layout
    positions = {}
    for node in G.nodes():
        if 'pos' in G.nodes[node]:
            positions[node] = G.nodes[node]['pos']
    
    if not positions:
        positions = nx.spring_layout(G, seed=42)
    
    # Get node sizes based on degree
    node_degrees = dict(G.degree())
    node_sizes = {node: node_size_factor * (1 + node_degrees.get(node, 0)) 
                 for node in G.nodes()}
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    # If we have community data, prepare colors
    if community_map:
        num_communities = max(community_map.values()) + 1
        colors = generate_color_palette(max(3, num_communities))
    else:
        colors = ['#636EFA']  # Default blue color
    
    # Create highlight set for O(1) lookups
    highlight_node_set = set(highlight_nodes) if highlight_nodes else set()
    
    # Collect node data
    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        
        # Generate node text with info
        text = f"<b>{node}</b><br>"
        text += f"Connections: {node_degrees.get(node, 0)}<br>"
        if 'count' in G.nodes[node]:
            text += f"Occurrences: {G.nodes[node]['count']}<br>"
        node_text.append(text)
        
        # Set node size
        node_size.append(node_sizes.get(node, node_size_factor))
        
        # Set node color based on community or highlight
        if node in highlight_node_set:
            node_color.append('#FF0000')  # Red for highlighted nodes
        elif community_map and node in community_map:
            community_id = community_map[node]
            node_color.append(colors[community_id % len(colors)])
        else:
            node_color.append(colors[0])
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color='#888'),
            symbol='circle'
        ),
        name='Entities'
    )
    
    # Prepare edge traces
    edge_traces = []
    
    # Group edges by predicate for legend
    predicates = set()
    for _, _, data in G.edges(data=True):
        if 'predicate' in data:
            predicates.add(data['predicate'])
    
    # Create a color for each predicate
    predicate_colors = {pred: color for pred, color in zip(predicates, generate_color_palette(len(predicates)))}
    
    # Create highlight set for edges
    highlight_edge_set = set(highlight_edges) if highlight_edges else set()
    
    # Create a trace for each predicate
    for predicate in predicates:
        edge_x = []
        edge_y = []
        edge_text = []
        edge_width = []
        
        for u, v, data in G.edges(data=True):
            if data.get('predicate') == predicate:
                # Get positions
                x0, y0 = positions[u]
                x1, y1 = positions[v]
                
                # For curved edges
                xmid = (x0 + x1) / 2
                ymid = (y0 + y1) / 2
                
                # Add a slight curve
                # Move midpoint perpendicular to edge direction
                dx = x1 - x0
                dy = y1 - y0
                edge_len = np.sqrt(dx*dx + dy*dy)
                if edge_len > 0:
                    # Normalize and rotate by 90 degrees
                    nx, ny = -dy/edge_len, dx/edge_len
                    xmid += nx * 0.03  # Small offset
                    ymid += ny * 0.03
                
                # Add points for curved line
                edge_x.extend([x0, xmid, x1, None])
                edge_y.extend([y0, ymid, y1, None])
                
                # Edge text
                confidence = data.get('confidence', 0.5)
                polarity = data.get('polarity', 'neutral')
                directness = data.get('directness', 'direct')
                
                text = f"<b>{predicate}</b><br>"
                text += f"Confidence: {confidence:.2f}<br>"
                text += f"Polarity: {polarity}<br>"
                text += f"Directness: {directness}<br>"
                if 'sentence' in data:
                    text += f"Context: {data['sentence']}"
                edge_text.append(text)
                
                # Edge width based on confidence and highlight status
                width = edge_width_factor * (0.5 + confidence)
                if (u, v) in highlight_edge_set:
                    width *= 2  # Double width for highlighted edges
                edge_width.append(width)
        
        if edge_x:  # Only create trace if there are edges
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(
                    width=edge_width[0],  # All edges in this trace have the same width
                    color=predicate_colors[predicate]
                ),
                hoverinfo='text',
                text=edge_text * 4,  # Repeat text for each segment
                mode='lines',
                name=predicate,
                opacity=0.8
            )
            edge_traces.append(edge_trace)
    
    # Build figure with all traces
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            titlefont=dict(size=16),
            showlegend=True,
            legend=dict(title="Predicates & Entities"),
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            plot_bgcolor='rgba(240,240,240,0.8)'
        )
    )
    
    # Add edge arrows
    for u, v, data in G.edges(data=True):
        # Get edge endpoints
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        
        # Calculate direction vector
        dx = x1 - x0
        dy = y1 - y0
        
        # Normalize
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            udx, udy = dx/length, dy/length
        else:
            continue  # Skip if points are identical
            
        # Calculate arrowhead position (pull back from endpoint)
        node_radius = node_sizes[v] / 1000  # Scale down
        ax = x1 - udx * node_radius
        ay = y1 - udy * node_radius
        
        # Add arrow annotation
        fig.add_annotation(
            x=ax, y=ay,
            ax=x1, ay=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=1,
            arrowcolor=predicate_colors.get(data.get('predicate', 'default'), '#888')
        )
    
    return fig

def create_heatmap_figure(adjacency_matrix: np.ndarray, 
                        labels: List[str],
                        title: str = "Relationship Adjacency Matrix") -> go.Figure:
    """
    Create a heatmap visualization of the adjacency matrix.
    
    Args:
        adjacency_matrix: Adjacency matrix as NumPy array
        labels: Node labels corresponding to matrix indices
        title: Figure title
        
    Returns:
        Plotly heatmap figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=adjacency_matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        hoverongaps=False,
        hoverinfo='text',
        text=[[f"{labels[i]} → {labels[j]}: {adjacency_matrix[i,j]:.2f}" 
              for j in range(len(labels))] 
              for i in range(len(labels))]
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=600,
        width=600,
        xaxis=dict(title='Target Entity'),
        yaxis=dict(title='Source Entity'),
        plot_bgcolor='rgba(240,240,240,0.8)'
    )
    
    return fig

def create_bar_chart(labels: List[str], values: List[float], 
                   title: str, xaxis_title: str, yaxis_title: str) -> go.Figure:
    """
    Create a bar chart visualization.
    
    Args:
        labels: Category labels
        values: Values for each category
        title: Chart title
        xaxis_title: X-axis title
        yaxis_title: Y-axis title
        
    Returns:
        Plotly bar chart figure
    """
    fig = go.Figure(data=go.Bar(
        x=labels,
        y=values,
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis=dict(title=xaxis_title),
        yaxis=dict(title=yaxis_title),
        height=400,
        plot_bgcolor='rgba(240,240,240,0.8)'
    )
    
    return fig

def create_sankey_diagram(G: nx.DiGraph, 
                        weight_attr: str = 'confidence',
                        title: str = "Relationship Flow Diagram") -> go.Figure:
    """
    Create a Sankey diagram of relationships.
    
    Args:
        G: NetworkX DiGraph
        weight_attr: Edge attribute to use as flow value
        title: Diagram title
        
    Returns:
        Plotly Sankey diagram figure
    """
    # Get all nodes
    all_nodes = list(G.nodes())
    
    if not all_nodes or not G.edges():
        # Return empty figure if no data
        return go.Figure()
        
    # Map node names to indices
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    # Prepare Sankey data
    node_labels = all_nodes
    sources = []
    targets = []
    values = []
    link_labels = []
    
    # Add edges
    for u, v, data in G.edges(data=True):
        sources.append(node_map[u])
        targets.append(node_map[v])
        
        # Use confidence or default value as flow
        values.append(data.get(weight_attr, 1.0))
        
        # Use predicate as link label
        link_labels.append(data.get('predicate', ''))
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="rgba(31, 119, 180, 0.8)"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            hovertemplate='%{source.label} → %{target.label}<br>Predicate: %{label}<br>Value: %{value:.2f}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=600,
        font=dict(size=10)
    )
    
    return fig

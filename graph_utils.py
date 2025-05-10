import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
import torch

def build_networkx_graph(relationships: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from relationship data.
    
    Args:
        relationships: List of relationship dictionaries
        
    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()
    
    # Add nodes and edges from relationships
    for rel in relationships:
        subject = rel.get("subject", "").strip()
        obj = rel.get("object", "").strip()
        predicate = rel.get("predicate", "").strip()
        
        # Skip invalid relationships
        if not subject or not obj or not predicate:
            continue
            
        # Add nodes if they don't exist
        if not G.has_node(subject):
            G.add_node(subject, type="subject", count=0)
        if not G.has_node(obj):
            G.add_node(obj, type="object", count=0)
            
        # Increment node counts
        G.nodes[subject]["count"] = G.nodes[subject].get("count", 0) + 1
        G.nodes[obj]["count"] = G.nodes[obj].get("count", 0) + 1
        
        # Add edge with attributes
        G.add_edge(subject, obj, 
                 predicate=predicate, 
                 polarity=rel.get("polarity", "neutral"),
                 directness=rel.get("directness", "direct"),
                 confidence=rel.get("confidence", 0.5),
                 sentence=rel.get("sentence", ""))
        
    return G

def get_graph_statistics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculate graph statistics.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary of graph statistics
    """
    if G.number_of_nodes() == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0,
            "avg_degree": 0,
            "max_in_degree": 0,
            "max_out_degree": 0,
            "most_central_nodes": [],
            "most_connected_nodes": []
        }
        
    # Basic statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Degree statistics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = dict(G.degree())
    
    avg_degree = sum(total_degrees.values()) / num_nodes
    max_in_degree = max(in_degrees.values()) if in_degrees else 0
    max_out_degree = max(out_degrees.values()) if out_degrees else 0
    
    # Centrality
    try:
        centrality = nx.betweenness_centrality(G)
        most_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    except:
        most_central = []
    
    # Most connected nodes
    most_connected = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "avg_degree": avg_degree,
        "max_in_degree": max_in_degree,
        "max_out_degree": max_out_degree,
        "most_central_nodes": most_central,
        "most_connected_nodes": most_connected
    }

def apply_rgcn_layout(G: nx.DiGraph, embeddings: np.ndarray, entity_map: Dict[str, int]) -> nx.DiGraph:
    """
    Apply RGCN embeddings as node positions in the graph.
    
    Args:
        G: NetworkX DiGraph
        embeddings: Node embeddings from RGCN
        entity_map: Mapping from entity strings to indices
        
    Returns:
        DiGraph with positions added
    """
    G_pos = G.copy()
    
    # For 2D visualization, use first two dimensions of embeddings
    if embeddings is not None and embeddings.shape[1] >= 2:
        # Scale coordinates to reasonable range
        x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
        y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
        
        x_range = max(1e-10, x_max - x_min)  # Avoid division by zero
        y_range = max(1e-10, y_max - y_min)
        
        # Apply positions to nodes
        for node in G_pos.nodes():
            if node in entity_map:
                idx = entity_map[node]
                if idx < len(embeddings):
                    # Scale to [0, 1] range
                    x = (embeddings[idx, 0] - x_min) / x_range
                    y = (embeddings[idx, 1] - y_min) / y_range
                    G_pos.nodes[node]['pos'] = (x, y)
    
    # If we don't have RGCN embeddings, use spring layout
    missing_pos = any('pos' not in G_pos.nodes[node] for node in G_pos.nodes())
    if missing_pos:
        pos = nx.spring_layout(G_pos)
        nx.set_node_attributes(G_pos, pos, 'pos')
    
    return G_pos

def get_community_structure(G: nx.DiGraph) -> Dict[str, int]:
    """
    Detect communities in the graph.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary mapping nodes to community IDs
    """
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    try:
        # Try Louvain method if available (community package)
        from community import best_partition
        communities = best_partition(G_undirected)
    except:
        # Fallback to connected components
        communities = {}
        for i, component in enumerate(nx.connected_components(G_undirected)):
            for node in component:
                communities[node] = i
    
    return communities

def find_important_paths(G: nx.DiGraph, 
                       top_n: int = 5, 
                       min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """
    Find important paths in the graph based on confidence scores.
    
    Args:
        G: NetworkX DiGraph
        top_n: Number of paths to return
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of path dictionaries
    """
    # Filter edges based on confidence
    high_conf_edges = [(u, v) for u, v, data in G.edges(data=True) 
                      if data.get('confidence', 0) >= min_confidence]
    
    # Create subgraph with high confidence edges
    high_conf_graph = G.edge_subgraph(high_conf_edges)
    
    # Find paths
    paths = []
    
    # First try to find paths of length 2 or more
    for source in high_conf_graph.nodes():
        for target in high_conf_graph.nodes():
            if source != target:
                try:
                    # Find shortest path
                    path = nx.shortest_path(high_conf_graph, source, target)
                    if len(path) > 1:
                        # Get path details
                        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                        edge_data = [G.get_edge_data(u, v) for u, v in edges]
                        
                        # Calculate average confidence
                        avg_confidence = sum(data.get('confidence', 0) for data in edge_data) / len(edge_data)
                        
                        paths.append({
                            'path': path,
                            'length': len(path) - 1,  # Number of edges
                            'avg_confidence': avg_confidence,
                            'edge_data': edge_data
                        })
                except nx.NetworkXNoPath:
                    pass
    
    # Sort by average confidence and length
    paths.sort(key=lambda x: (x['avg_confidence'], x['length']), reverse=True)
    
    # Return top N paths
    return paths[:top_n]

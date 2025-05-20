import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
import random

def build_networkx_graph(relationships: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from relationship data.
    
    Args:
        relationships: List of relationship dictionaries
        
    Returns:
        NetworkX DiGraph
    """
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for rel in relationships:
        subject = rel.get("subject", "")
        predicate = rel.get("predicate", "")
        obj = rel.get("object", "")
        confidence = rel.get("confidence", 0.5)
        polarity = rel.get("polarity", "neutral")
        directness = rel.get("directness", "direct")
        sentence = rel.get("sentence", "")
        
        # Skip empty subjects or objects
        if not subject or not obj:
            continue
        
        # Add nodes if they don't exist
        if not G.has_node(subject):
            G.add_node(subject)
        
        if not G.has_node(obj):
            G.add_node(obj)
        
        # Add edge with attributes
        G.add_edge(
            subject, 
            obj, 
            predicate=predicate,
            confidence=confidence,
            polarity=polarity,
            directness=directness,
            sentence=sentence
        )
    
    return G

def get_graph_statistics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculate graph statistics.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary of graph statistics
    """
    stats = {}
    
    # Basic statistics
    stats["num_nodes"] = G.number_of_nodes()
    stats["num_edges"] = G.number_of_edges()
    
    # Graph density (ratio of actual to possible edges)
    # For a directed graph, the maximum number of edges is n(n-1)
    if stats["num_nodes"] > 1:
        stats["density"] = stats["num_edges"] / (stats["num_nodes"] * (stats["num_nodes"] - 1))
    else:
        stats["density"] = 0.0
    
    # Average degree (average number of edges per node)
    if stats["num_nodes"] > 0:
        stats["avg_degree"] = stats["num_edges"] / stats["num_nodes"]
    else:
        stats["avg_degree"] = 0.0
    
    # Centrality measures (if graph is not empty)
    if stats["num_nodes"] > 0:
        try:
            # Degree centrality
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
            
            stats["top_central_nodes_in"] = sorted(
                in_degree_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            stats["top_central_nodes_out"] = sorted(
                out_degree_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        except:
            # Handle potential errors in centrality calculations
            stats["top_central_nodes_in"] = []
            stats["top_central_nodes_out"] = []
    
    return stats

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
    # Create a copy of the graph to avoid modifying the original
    G_with_pos = G.copy()
    
    # Scale embeddings to reasonable layout coordinates
    # Use first two dimensions of embeddings as x, y coordinates
    if embeddings.shape[1] >= 2:
        # Normalize to range [0, 1]
        min_vals = embeddings.min(axis=0)
        max_vals = embeddings.max(axis=0)
        range_vals = max_vals - min_vals
        
        # Avoid division by zero
        range_vals[range_vals == 0] = 1.0
        
        normalized_embeddings = (embeddings - min_vals) / range_vals
        
        # Scale to layout range
        scale = 10.0
        scaled_embeddings = normalized_embeddings * scale
        
        # Apply positions to nodes
        for node in G_with_pos.nodes():
            if node in entity_map:
                idx = entity_map[node]
                if idx < len(embeddings):
                    x, y = scaled_embeddings[idx, 0], scaled_embeddings[idx, 1]
                    G_with_pos.nodes[node]['pos'] = [x, y]
    
    # For nodes without positions, use spring layout
    nodes_without_pos = [node for node in G_with_pos.nodes() if 'pos' not in G_with_pos.nodes[node]]
    
    if nodes_without_pos:
        # Get existing positions as fixed positions for spring layout
        fixed_positions = {
            node: G_with_pos.nodes[node]['pos'] 
            for node in G_with_pos.nodes() 
            if 'pos' in G_with_pos.nodes[node]
        }
        
        # Use spring layout with fixed positions for remaining nodes
        if fixed_positions:
            remaining_pos = nx.spring_layout(
                G_with_pos.subgraph(nodes_without_pos),
                k=0.5,
                seed=42
            )
            
            # Add positions to graph
            for node, pos in remaining_pos.items():
                G_with_pos.nodes[node]['pos'] = pos
        else:
            # If no fixed positions, use spring layout for all
            pos = nx.spring_layout(G_with_pos, seed=42)
            nx.set_node_attributes(G_with_pos, {n: p for n, p in pos.items()}, 'pos')
    
    return G_with_pos

def get_community_structure(G: nx.DiGraph) -> Dict[str, int]:
    """
    Detect communities in the graph.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary mapping nodes to community IDs
    """
    # Convert directed graph to undirected for community detection
    G_undirected = G.to_undirected()
    
    # For very small graphs, assign all to one community
    if G_undirected.number_of_nodes() < 3:
        return {node: 0 for node in G.nodes()}
    
    # Try to use connected components
    communities = {}
    
    try:
        # Get connected components
        components = list(nx.connected_components(G_undirected))
        
        # Assign community IDs
        for i, component in enumerate(components):
            for node in component:
                communities[node] = i
                
    except Exception:
        # Fallback: assign random communities
        communities = {node: random.randint(0, 2) for node in G.nodes()}
    
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
    # Filter edges by confidence
    filtered_edges = [(u, v) for u, v, data in G.edges(data=True) 
                     if data.get('confidence', 0) >= min_confidence]
    
    # Create filtered graph
    filtered_G = G.edge_subgraph(filtered_edges).copy()
    
    # Find all simple paths of length >= 2 in the filtered graph
    all_paths = []
    
    # Limit search to avoid combinatorial explosion
    max_path_length = min(4, G.number_of_nodes() - 1)
    
    # Get nodes with high centrality as starting points
    if G.number_of_nodes() > 0:
        try:
            # Use degree centrality to find important nodes
            centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            start_nodes = [node for node, _ in sorted_nodes[:min(5, len(sorted_nodes))]]
        except:
            # Fallback to using all nodes
            start_nodes = list(filtered_G.nodes())
    else:
        start_nodes = []
    
    # Find paths from important nodes
    for start_node in start_nodes:
        for node in filtered_G.nodes():
            if node == start_node:
                continue
                
            try:
                # Find simple paths between the nodes
                for path in nx.all_simple_paths(
                    filtered_G, start_node, node, cutoff=max_path_length
                ):
                    if len(path) >= 3:  # At least 3 nodes (2 edges)
                        all_paths.append(path)
            except nx.NetworkXNoPath:
                continue
    
    # If no paths found, try with all nodes as starting points
    if not all_paths and filtered_G.number_of_nodes() > 0:
        for start_node in filtered_G.nodes():
            for node in filtered_G.nodes():
                if node == start_node:
                    continue
                    
                try:
                    # Find simple paths between the nodes
                    for path in nx.all_simple_paths(
                        filtered_G, start_node, node, cutoff=max_path_length
                    ):
                        if len(path) >= 3:  # At least 3 nodes (2 edges)
                            all_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
    
    # Calculate average confidence for each path
    path_scores = []
    
    for path in all_paths:
        # Get edge data for consecutive nodes in the path
        edge_data = []
        total_confidence = 0.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            data = G.get_edge_data(u, v)
            edge_data.append(data)
            total_confidence += data.get('confidence', 0.0)
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(edge_data) if edge_data else 0.0
        
        path_scores.append({
            'path': path,
            'avg_confidence': avg_confidence,
            'edge_data': edge_data
        })
    
    # Sort paths by average confidence
    sorted_paths = sorted(path_scores, key=lambda x: x['avg_confidence'], reverse=True)
    
    # Return top N paths
    return sorted_paths[:top_n]
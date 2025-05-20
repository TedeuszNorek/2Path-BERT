import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
import textwrap
from io import StringIO

# Import our custom modules
from bart_processor import BARTProcessor
from rgcn_model import RGCNProcessor
import graph_utils
from visualization import (
    create_relationship_graph_figure,
    create_heatmap_figure,
    create_bar_chart,
    create_sankey_diagram
)

# Import database modules
import db_models
import db_utils
from sqlalchemy.orm import Session
from contextlib import contextmanager

# Database session context manager
@contextmanager
def get_db():
    """Get a database session and handle closing it safely"""
    session = db_models.SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Semantic Relationship Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'rgcn_result' not in st.session_state:
    st.session_state.rgcn_result = None

# Create processors as singletons in session state
if 'bart_processor' not in st.session_state:
    st.session_state.bart_processor = BARTProcessor()
if 'rgcn_processor' not in st.session_state:
    st.session_state.rgcn_processor = RGCNProcessor()

# App header
st.title("🔍 Semantic Relationship Analyzer")
st.markdown("""
This application extracts and visualizes semantic relationships from text using BART and RGCN models.
Extract subject-predicate-object triples, analyze relationship patterns, and explore knowledge graphs.
""")

# Sidebar configuration
st.sidebar.header("Input Options")

# Input method selection
input_method = st.sidebar.radio(
    "Choose input method",
    ["Text Input", "File Upload", "Sample Text"]
)

# Sample texts
sample_texts = {
    "Scientific": """
    Climate change is causing global temperatures to rise. Rising temperatures lead to melting ice caps.
    Melting ice caps contribute to sea level rise. Sea level rise threatens coastal cities.
    Greenhouse gases trap heat in the atmosphere. Human activities produce greenhouse gases.
    Renewable energy can reduce carbon emissions. Carbon emissions accelerate climate change.
    """,
    
    "News": """
    The Federal Reserve raised interest rates yesterday. Higher interest rates may slow economic growth.
    The company announced a new product launch. The new product features advanced AI capabilities.
    Investors responded positively to the earnings report. The market showed strong performance this quarter.
    The president met with foreign leaders to discuss trade agreements. International relations influence economic policies.
    """,
    
    "Educational": """
    Photosynthesis converts light energy into chemical energy. Plants use chlorophyll to capture sunlight.
    The water cycle involves evaporation, condensation, and precipitation. Clouds form when water vapor condenses.
    Mitochondria are the powerhouse of the cell. DNA contains genetic instructions for development and functioning.
    The Pythagorean theorem relates the sides of a right triangle. Mathematics provides tools for solving complex problems.
    """
}

# Text input
text_for_analysis = ""

if input_method == "Text Input":
    text_for_analysis = st.sidebar.text_area(
        "Enter text to analyze",
        height=200,
        help="Input text you want to extract relationships from"
    )
    
elif input_method == "File Upload":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a text file",
        type=["txt", "md", "csv"],
        help="Upload a text file to extract relationships from"
    )
    
    if uploaded_file is not None:
        # Handle different file types
        if uploaded_file.name.endswith('.csv'):
            # For CSV, let user select the column
            df = pd.read_csv(uploaded_file)
            st.sidebar.subheader("CSV File Loaded")
            text_column = st.sidebar.selectbox(
                "Select text column",
                df.columns.tolist()
            )
            text_for_analysis = "\n".join(df[text_column].astype(str).tolist())
        else:
            # For other text files
            text_for_analysis = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        
        # Show a preview
        st.sidebar.subheader("Preview")
        st.sidebar.text(textwrap.shorten(text_for_analysis, width=300, placeholder="..."))

elif input_method == "Sample Text":
    sample_choice = st.sidebar.selectbox(
        "Select a sample text",
        list(sample_texts.keys())
    )
    text_for_analysis = sample_texts[sample_choice]
    st.sidebar.text_area("Sample Text Preview", text_for_analysis, height=150, disabled=True)

# Processing options
st.sidebar.header("Processing Options")

min_confidence = st.sidebar.slider(
    "Minimum confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Filter relationships with confidence below this threshold"
)

# Advanced options in expander
with st.sidebar.expander("Advanced Options"):
    visualization_type = st.selectbox(
        "Visualization type",
        ["Graph (RGCN Layout)", "Graph (Force Layout)", "Adjacency Matrix", "Sankey Diagram"]
    )
    
    show_negative = st.checkbox("Show negative relationships", value=True)
    show_indirect = st.checkbox("Show indirect relationships", value=True)
    
    highlight_entity = st.text_input(
        "Highlight entity (leave empty for none)",
        "",
        help="Enter an entity name to highlight in visualizations"
    )

# Process button
process_button = st.sidebar.button("Process Text", use_container_width=True)

# Main body of the app
if process_button and text_for_analysis:
    # Store the current text in session state
    st.session_state.current_text = text_for_analysis
    
    # Create processing container
    with st.spinner("Processing text..."):
        # Clear previous results
        st.session_state.processed_data = None
        st.session_state.graph = None
        st.session_state.rgcn_result = None
        
        try:
            # Step 1: Process text with BART
            start_time = time.time()
            processed_data = st.session_state.bart_processor.extract_relationships(text_for_analysis)
            bart_time = time.time() - start_time
            
            # Filter relationships by confidence
            filtered_relationships = [
                rel for rel in processed_data["relationships"]
                if rel["confidence"] >= min_confidence
            ]
            
            # Filter by polarity and directness if needed
            if not show_negative:
                filtered_relationships = [
                    rel for rel in filtered_relationships
                    if rel["polarity"] != "negative"
                ]
                
            if not show_indirect:
                filtered_relationships = [
                    rel for rel in filtered_relationships
                    if rel["directness"] != "indirect"
                ]
                
            # Update the processed data with filtered relationships
            processed_data["filtered_relationships"] = filtered_relationships
            
            # Step 2: Build graph
            start_time = time.time()
            graph = graph_utils.build_networkx_graph(filtered_relationships)
            graph_time = time.time() - start_time
            
            # Step 3: Process with RGCN
            start_time = time.time()
            rgcn_result = st.session_state.rgcn_processor.process_relationships(filtered_relationships)
            rgcn_time = time.time() - start_time
            
            # Step 4: Apply RGCN layout to graph
            graph_with_layout = graph_utils.apply_rgcn_layout(
                graph,
                rgcn_result["embeddings"],
                rgcn_result["entity_to_idx"]
            )
            
            # Store results in session state
            st.session_state.processed_data = processed_data
            st.session_state.graph = graph_with_layout
            st.session_state.rgcn_result = rgcn_result
            
            # Show timing info
            st.info(f"Processing completed: BART: {bart_time:.2f}s, Graph: {graph_time:.2f}s, RGCN: {rgcn_time:.2f}s")
            
        except Exception as e:
            st.error(f"Error processing text: {str(e)}")

# Display results if available
if st.session_state.processed_data and st.session_state.graph:
    # Get data from session state
    processed_data = st.session_state.processed_data
    graph = st.session_state.graph
    rgcn_result = st.session_state.rgcn_result
    
    # Setup main display area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Relationship Visualization")
        
        # Create highlight lists for visualization if an entity is specified
        highlight_nodes = []
        highlight_edges = []
        
        if highlight_entity and highlight_entity in graph.nodes():
            highlight_nodes = [highlight_entity]
            # Find edges connected to the highlighted entity
            highlight_edges = [
                (u, v) for u, v in graph.edges()
                if u == highlight_entity or v == highlight_entity
            ]
        
        # Get community structure
        communities = graph_utils.get_community_structure(graph)
        
        # Visualize based on selected type
        if visualization_type == "Graph (RGCN Layout)":
            # Create figure using RGCN layout
            fig = create_relationship_graph_figure(
                graph,
                community_map=communities,
                highlight_nodes=highlight_nodes,
                highlight_edges=highlight_edges,
                title="Semantic Relationship Graph (RGCN Layout)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif visualization_type == "Graph (Force Layout)":
            # Use force layout
            pos = nx.spring_layout(graph, seed=42)
            nx.set_node_attributes(graph, {node: position for node, position in pos.items()}, 'pos')
            
            fig = create_relationship_graph_figure(
                graph,
                community_map=communities,
                highlight_nodes=highlight_nodes,
                highlight_edges=highlight_edges,
                title="Semantic Relationship Graph (Force Layout)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif visualization_type == "Adjacency Matrix":
            # Create adjacency matrix visualization
            # Get entities
            entities = list(graph.nodes())
            n = len(entities)
            
            if n > 0:
                # Create matrix
                adj_matrix = np.zeros((n, n))
                entity_to_idx = {entity: i for i, entity in enumerate(entities)}
                
                # Fill matrix
                for u, v, data in graph.edges(data=True):
                    i, j = entity_to_idx[u], entity_to_idx[v]
                    adj_matrix[i, j] = data.get('confidence', 0.5)
                
                # Create heatmap
                fig = create_heatmap_figure(
                    adj_matrix,
                    entities,
                    title="Relationship Adjacency Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No relationships to visualize")
                
        elif visualization_type == "Sankey Diagram":
            # Create Sankey diagram
            fig = create_sankey_diagram(
                graph,
                weight_attr='confidence',
                title="Relationship Flow Diagram"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show statistics
        st.header("Statistics")
        
        # Graph statistics
        graph_stats = graph_utils.get_graph_statistics(graph)
        
        st.subheader("Graph Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        metrics_col1.metric("Entities", graph_stats["num_nodes"])
        metrics_col2.metric("Relationships", graph_stats["num_edges"])
        
        metrics_col1.metric("Avg. Connections", f"{graph_stats['avg_degree']:.2f}")
        metrics_col2.metric("Graph Density", f"{graph_stats['density']:.3f}")
        
        # Polarity and directness distributions
        if "statistics" in processed_data:
            stats = processed_data["statistics"]
            
            if "polarity_distribution" in stats:
                st.subheader("Polarity Distribution")
                pol_dist = stats["polarity_distribution"]
                
                # Create data for bar chart
                labels = list(pol_dist.keys())
                values = list(pol_dist.values())
                
                fig = create_bar_chart(
                    labels, values,
                    "Relationship Polarity",
                    "Polarity", "Count"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if "directness_distribution" in stats:
                st.subheader("Directness Distribution")
                dir_dist = stats["directness_distribution"]
                
                # Create data for bar chart
                labels = list(dir_dist.keys())
                values = list(dir_dist.values())
                
                fig = create_bar_chart(
                    labels, values,
                    "Relationship Directness",
                    "Directness", "Count"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Display extracted relationships
    st.header("Extracted Relationships")
    
    if "filtered_relationships" in processed_data:
        relationships = processed_data["filtered_relationships"]
        
        if relationships:
            # Create DataFrame for display
            df_data = []
            for rel in relationships:
                df_data.append({
                    "Subject": rel["subject"],
                    "Predicate": rel["predicate"],
                    "Object": rel["object"],
                    "Confidence": f"{rel['confidence']:.2f}",
                    "Polarity": rel["polarity"],
                    "Directness": rel["directness"]
                })
            
            rel_df = pd.DataFrame(df_data)
            
            # Display table with sorting
            st.dataframe(
                rel_df,
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        help="Confidence score of the extracted relationship",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "Polarity": st.column_config.SelectboxColumn(
                        "Polarity",
                        help="Sentiment polarity of the relationship",
                        options=["positive", "negative", "neutral"],
                        width="medium"
                    ),
                    "Directness": st.column_config.SelectboxColumn(
                        "Directness",
                        help="Whether the relationship is direct or indirect",
                        options=["direct", "indirect"],
                        width="medium"
                    )
                }
            )
        else:
            st.warning("No relationships found that meet the filtering criteria.")
    
    # Display important paths
    st.header("Important Relationship Paths")
    important_paths = graph_utils.find_important_paths(
        graph,
        top_n=5,
        min_confidence=min_confidence
    )
    
    if important_paths:
        for i, path_data in enumerate(important_paths):
            path = path_data["path"]
            edge_data = path_data["edge_data"]
            
            # Format path as readable text
            path_text = " → ".join(path)
            
            # Create expander for each path
            with st.expander(f"Path {i+1}: {path_text} (Confidence: {path_data['avg_confidence']:.2f})"):
                # Show edges in the path
                for j, (u, v) in enumerate(zip(path[:-1], path[1:])):
                    data = edge_data[j]
                    predicate = data.get("predicate", "")
                    confidence = data.get("confidence", 0)
                    polarity = data.get("polarity", "neutral")
                    directness = data.get("directness", "direct")
                    
                    st.markdown(f"**{u}** *({predicate})* → **{v}**")
                    st.markdown(f"Confidence: {confidence:.2f}, Polarity: {polarity}, Directness: {directness}")
                    if "sentence" in data:
                        st.markdown(f"*Context: \"{data['sentence']}\"*")
                    st.divider()
    else:
        st.info("No significant paths found in the relationship graph.")
    
    # RDF Triples
    st.header("Generated RDF Triples")
    
    if "rdf_triples" in processed_data:
        triples = processed_data["rdf_triples"]
        
        if triples:
            # Create DataFrame for display
            df_data = []
            for triple in triples:
                if len(triple) == 3:
                    subject, predicate, obj = triple
                    df_data.append({
                        "Subject": subject,
                        "Predicate": predicate,
                        "Object": obj
                    })
            
            if df_data:
                triple_df = pd.DataFrame(df_data)
                st.dataframe(triple_df, use_container_width=True)
            else:
                st.warning("No valid RDF triples found.")
        else:
            st.warning("No RDF triples were generated.")
    
# Display initial instructions if no data has been processed
if not st.session_state.processed_data:
    st.info("👈 Enter or upload text on the left sidebar and click 'Process Text' to analyze semantic relationships.")
    
    # Show some sample instructions
    with st.expander("How to use this application"):
        st.markdown("""
        ### Getting Started
        
        1. **Input Text**: Enter text directly, upload a file, or use one of the sample texts.
        2. **Set Options**: Adjust the confidence threshold and other filtering options.
        3. **Process**: Click the 'Process Text' button to extract relationships.
        4. **Explore Results**: View the visualizations, statistics, and extracted relationships.
        
        ### Understanding the Output
        
        - **Relationships**: Subject-predicate-object triples extracted from the text.
        - **Polarity**: Whether a relationship is positive, negative, or neutral.
        - **Directness**: Whether a relationship is stated directly or indirectly.
        - **Confidence**: The model's confidence in the extracted relationship.
        
        ### Visualization Types
        
        - **Graph (RGCN Layout)**: Shows entities and relationships using RGCN embeddings.
        - **Graph (Force Layout)**: Shows entities and relationships using force-directed layout.
        - **Adjacency Matrix**: Shows connections between entities as a heatmap.
        - **Sankey Diagram**: Shows relationship flows between entities.
        """)

# Footer
st.markdown("---")
st.markdown(
    "Powered by BART (semantic extraction) and RGCN (graph modeling) | "
    "Use this tool to analyze text and discover semantic relationship patterns."
)

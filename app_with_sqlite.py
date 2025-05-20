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

# Import simple database module
import db_simple as db

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

# Create tabs for main functionality and database explorer
main_tab, database_tab = st.tabs(["Analyzer", "Database Explorer"])

# Database Explorer tab content
with database_tab:
    st.header("Database Explorer")
    
    # Get data from database
    analysis_count = db.get_analysis_count()
    relationship_count = db.get_relationship_count()
    top_entities_df = db.get_top_entities(limit=20)
    recent_analyses_df = db.get_recent_analyses(limit=10)
    
    # Display database statistics
    st.subheader("Database Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Analyses", analysis_count)
    col2.metric("Total Relationships", relationship_count)
    
    # Display top entities
    if not top_entities_df.empty:
        st.subheader("Top Entities")
        
        # Create bar chart
        entity_chart = px.bar(
            top_entities_df, 
            x="name", 
            y="count",
            title="Most Common Entities in Knowledge Base",
            color="count",
            color_continuous_scale="Viridis"
        )
        entity_chart.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(entity_chart, use_container_width=True)
    
    # Recent analyses
    if not recent_analyses_df.empty:
        st.subheader("Recent Analyses")
        
        # Display table
        st.dataframe(
            recent_analyses_df.assign(
                text_length=recent_analyses_df['text'].str.len()
            )[['id', 'title', 'timestamp', 'text_length']].rename(
                columns={
                    'id': 'ID',
                    'title': 'Title',
                    'timestamp': 'Date',
                    'text_length': 'Text Length'
                }
            ),
            use_container_width=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", help="Analysis ID"),
                "Title": st.column_config.TextColumn("Title", help="Analysis title"),
                "Date": st.column_config.TextColumn("Date", help="Analysis date and time"),
                "Text Length": st.column_config.NumberColumn("Text Length", help="Length of analyzed text in characters")
            }
        )
        
        # Selected analysis details
        selected_analysis_id = st.selectbox(
            "Select an analysis to view details",
            options=recent_analyses_df['id'].tolist(),
            format_func=lambda x: f"#{x}: {recent_analyses_df[recent_analyses_df['id']==x]['title'].iloc[0] if not recent_analyses_df[recent_analyses_df['id']==x]['title'].isnull().iloc[0] else 'Untitled'}"
        )
        
        if selected_analysis_id:
            # Get analysis and its relationships
            analysis = db.get_analysis_by_id(selected_analysis_id)
            relationships_df = db.get_relationships_by_analysis_id(selected_analysis_id)
            
            if analysis:
                # Display analysis details
                st.subheader(f"Analysis #{analysis['id']}: {analysis['title'] or 'Untitled'}")
                st.caption(f"Created on {analysis['timestamp']}")
                
                # Display text in expander
                with st.expander("View Text"):
                    st.text_area("Analyzed Text", analysis['text'], height=150, disabled=True)
                
                if not relationships_df.empty:
                    # Display relationships
                    st.subheader("Extracted Relationships")
                    st.dataframe(
                        relationships_df,
                        use_container_width=True,
                        column_config={
                            "confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                help="Confidence score of the extracted relationship",
                                format="%.2f",
                                min_value=0,
                                max_value=1,
                            )
                        }
                    )
                    
                    # Create and display graph visualization
                    st.subheader("Relationship Graph")
                    
                    # Convert relationships to format needed for graph
                    rel_dicts = []
                    for _, row in relationships_df.iterrows():
                        rel_dict = {
                            "subject": row['subject'],
                            "predicate": row['predicate'],
                            "object": row['object'],
                            "confidence": row['confidence'],
                            "polarity": row['polarity'],
                            "directness": row['directness'],
                            "sentence": row['sentence']
                        }
                        rel_dicts.append(rel_dict)
                    
                    # Build graph
                    graph = graph_utils.build_networkx_graph(rel_dicts)
                    communities = graph_utils.get_community_structure(graph)
                    
                    # Create visualization
                    fig = create_relationship_graph_figure(
                        graph,
                        community_map=communities,
                        title=f"Relationship Graph for Analysis #{analysis['id']}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Option to delete the analysis
                    if st.button("Delete Analysis", use_container_width=True, type="primary"):
                        if db.delete_analysis(int(analysis['id'])):
                            st.success(f"Analysis #{analysis['id']} deleted successfully")
                            st.rerun()
                        else:
                            st.error("Failed to delete analysis")
                else:
                    st.info("No relationships found for this analysis.")
    else:
        st.info("No analyses found in the database. Process some text and save it to get started.")

# Sidebar configuration
st.sidebar.header("Input Options")

# Database stats in sidebar
st.sidebar.caption(f"Database: {analysis_count} analyses, {relationship_count} relationships")

# Database options in expander
with st.sidebar.expander("Saved Analyses"):
    recent_analyses_df = db.get_recent_analyses(limit=5)
    
    if not recent_analyses_df.empty:
        selected_analysis = st.selectbox(
            "Load saved analysis",
            options=[f"#{row['id']}: {row['title'] or 'Untitled'} ({row['timestamp']})" 
                    for _, row in recent_analyses_df.iterrows()],
            index=None,
            placeholder="Select an analysis..."
        )
        
        if selected_analysis:
            # Extract analysis ID from selection
            analysis_id = int(selected_analysis.split(":")[0][1:])
            
            # Load button
            if st.button("Load Selected Analysis", use_container_width=True):
                # Get analysis and relationships
                analysis = db.get_analysis_by_id(analysis_id)
                relationships_df = db.get_relationships_by_analysis_id(analysis_id)
                
                if analysis:
                    st.session_state.current_text = analysis['text']
                    
                    # Convert to the format expected by the app
                    rel_dicts = []
                    for _, row in relationships_df.iterrows():
                        rel_dict = {
                            "subject": row['subject'],
                            "predicate": row['predicate'],
                            "object": row['object'],
                            "confidence": row['confidence'],
                            "polarity": row['polarity'],
                            "directness": row['directness'],
                            "sentence": row['sentence']
                        }
                        rel_dicts.append(rel_dict)
                    
                    # Rerun to update the UI
                    st.rerun()
    else:
        st.caption("No saved analyses found")

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

# Add title field and save option
analysis_title = st.sidebar.text_input("Analysis Title (for saving)", "", help="Optional title when saving this analysis")

# Process and save options
col1, col2 = st.sidebar.columns(2)
process_button = col1.button("Process Text", use_container_width=True)
save_button = col2.button("Save to DB", use_container_width=True, disabled=(not st.session_state.processed_data))

# Main content tab
with main_tab:
    # Handle save button click
    if save_button and st.session_state.processed_data:
        with st.spinner("Saving analysis to database..."):
            # Save analysis with title if provided
            title = analysis_title if analysis_title else None
            analysis_id = db.save_analysis(
                text=st.session_state.current_text,
                relationships=st.session_state.processed_data.get("filtered_relationships", []),
                title=title
            )
            
            # Show success message
            st.success(f"Analysis saved to database with ID: {analysis_id}")
            
            # Reset title field
            analysis_title = ""
            
            # Force rerun to update database stats
            st.rerun()

    # Handle process button click
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
        important_paths = graph_utils.find_important_paths(graph, top_n=3, min_confidence=min_confidence)
        
        if important_paths:
            for i, path_data in enumerate(important_paths):
                path = path_data["path"]
                avg_confidence = path_data["avg_confidence"]
                edge_data = path_data["edge_data"]
                
                # Create path description
                path_str = " → ".join(path)
                
                with st.expander(f"Path {i+1}: {path_str} (Confidence: {avg_confidence:.2f})"):
                    # Show edges in the path
                    for j in range(len(path) - 1):
                        subj = path[j]
                        obj = path[j + 1]
                        data = edge_data[j]
                        
                        st.write(f"**{subj}** ─[*{data.get('predicate', '')}*]→ **{obj}**")
                        st.caption(f"Confidence: {data.get('confidence', 0):.2f} | Polarity: {data.get('polarity', 'neutral')} | Directness: {data.get('directness', 'direct')}")
                        if 'sentence' in data:
                            st.caption(f"Context: '{data['sentence']}'")
                        st.divider()
        else:
            st.info("No significant paths found with the current confidence threshold.")
        
        # Display RDF triples
        if "rdf_triples" in processed_data:
            with st.expander("View RDF Triples"):
                triples = processed_data["rdf_triples"]
                
                triple_df = pd.DataFrame([
                    {"Subject": s, "Predicate": p, "Object": o}
                    for s, p, o in triples
                ])
                
                st.dataframe(triple_df, use_container_width=True)
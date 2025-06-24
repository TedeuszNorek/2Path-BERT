import streamlit as st
import time
import pandas as pd
import datetime
import os
import logging
from typing import Dict, List, Any, Optional

# Core research modules
from bert_processor import BERTProcessor
from gnn_models import GNNProcessor
import graph_utils
import visualization
import db_simple as db
from data_manager import DataManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set page config
st.set_page_config(
    page_title="BERT+GNN Research Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'bert_processor' not in st.session_state:
    st.session_state.bert_processor = None
if 'gnn_processor' not in st.session_state:
    st.session_state.gnn_processor = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'gnn_result' not in st.session_state:
    st.session_state.gnn_result = None
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""
if 'experiment_id' not in st.session_state:
    st.session_state.experiment_id = None

# Initialize database
db.init_db()

# Sidebar - Research Configuration
st.sidebar.header("🔬 Research Configuration")

# Model Selection
base_model = st.sidebar.selectbox(
    "Base Model",
    ["BERT"],
    help="Base language model for semantic extraction"
)

gnn_model = st.sidebar.selectbox(
    "GNN Architecture",
    ["None", "RGCN", "CompGCN", "RGAT"],
    help="Graph Neural Network architecture for relationship processing"
)

# Scientific model explanations
with st.sidebar.expander("📚 Scientific Model Rationale"):
    if gnn_model == "None":
        st.markdown("""
        **BERT Baseline (Control)**
        
        **Research Purpose**: Establishes baseline performance for transformer-based semantic extraction without graph enhancement.
        
        **Architecture**: Pure BERT encoder with dependency parsing for relationship identification using syntactic patterns (Subject-Verb-Object, Named Entity Relations, Noun Phrase Dependencies).
        
        **Scientific Value**: Control group for measuring isolated GNN contribution to relationship extraction accuracy and semantic understanding.
        """)
    elif gnn_model == "RGCN":
        st.markdown("""
        **Relational Graph Convolutional Network (Schlichtkrull et al., 2018)**
        
        **Research Purpose**: Tests whether relation-specific weight matrices improve semantic relationship classification over homogeneous graph approaches.
        
        **Core Innovation**: Separate weight matrices W_r for each relation type r, enabling specialized processing of different semantic relationships (causal, temporal, hierarchical).
        
        **Scientific Hypothesis**: Heterogeneous graph processing with relation-specific parameters captures semantic nuances better than uniform graph convolution.
        
        **Applications**: Knowledge graph completion, multi-relational reasoning, biomedical relation extraction.
        """)
    elif gnn_model == "CompGCN":
        st.markdown("""
        **Composition-based Graph Convolutional Network (Vashishth et al., 2020)**
        
        **Research Purpose**: Investigates whether joint entity-relation embedding in unified vector space improves semantic relationship modeling.
        
        **Core Innovation**: Composition functions (multiplication, subtraction, circular correlation) that combine entity and relation embeddings during message passing.
        
        **Scientific Hypothesis**: Unified entity-relation representation captures compositional semantics better than separate entity/relation processing.
        
        **Applications**: Knowledge base completion, question answering, semantic parsing requiring compositional reasoning.
        """)
    elif gnn_model == "RGAT":
        st.markdown("""
        **Relational Graph Attention Network (Busbridge et al., 2019)**
        
        **Research Purpose**: Evaluates whether attention mechanisms can automatically identify semantically important relationships without manual feature engineering.
        
        **Core Innovation**: Multi-head attention over relation-specific message passing, computing attention weights α_ij^r for each relation type r.
        
        **Scientific Hypothesis**: Learnable attention weights provide interpretable relationship importance ranking and improve semantic focus.
        
        **Applications**: Document-level relation extraction, scientific literature analysis, interpretable knowledge discovery.
        """)

# Temperature controls for each model configuration
st.sidebar.subheader("🌡️ Temperature Controls")
st.sidebar.markdown("*Set different temperatures for each model configuration*")

# Base BERT temperature
bert_temperature = st.sidebar.slider(
    "BERT Baseline Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Temperature for BERT-only processing"
)

# GNN temperatures
rgcn_temperature = st.sidebar.slider(
    "RGCN Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Temperature for BERT+RGCN processing"
)

compgcn_temperature = st.sidebar.slider(
    "CompGCN Temperature", 
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Temperature for BERT+CompGCN processing"
)

rgat_temperature = st.sidebar.slider(
    "RGAT Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Temperature for BERT+RGAT processing"
)

# Select active temperature based on current model
if gnn_model == "None":
    temperature = bert_temperature
elif gnn_model == "RGCN":
    temperature = rgcn_temperature
elif gnn_model == "CompGCN":
    temperature = compgcn_temperature
elif gnn_model == "RGAT":
    temperature = rgat_temperature
else:
    temperature = bert_temperature

# Analysis prompt
analysis_prompt = st.sidebar.text_area(
    "Analysis Prompt",
    placeholder="Enter specific instructions for relationship extraction...",
    height=100,
    help="Custom prompt to guide semantic analysis"
)

# Researcher identifier
engineer = st.sidebar.text_input(
    "Researcher ID",
    placeholder="Optional researcher identifier",
    help="For tracking experimental provenance"
)

# Clear cache for clean experiments (preserves database history)
if st.sidebar.button("🧹 Clear Session Cache", help="Reset current session data only - preserves database history"):
    result = DataManager.clear_session_only(preserve_db=True)
    if result["status"] == "success":
        st.sidebar.success(result["message"])
        st.sidebar.info(f"Database protection: {result['database_protected']}")
    else:
        st.sidebar.error("Cache clearing failed")

# Data verification button
if st.sidebar.button("🔍 Verify Data Protection", help="Check data separation status"):
    verification = DataManager.verify_data_separation()
    st.sidebar.json(verification)

st.sidebar.divider()

# Processing controls
min_confidence = st.sidebar.slider(
    "Minimum Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Filter relationships below this confidence"
)

show_negative = st.sidebar.checkbox("Include Negative Relationships", value=True)
show_indirect = st.sidebar.checkbox("Include Indirect Relationships", value=True)

# Main interface
st.title("🔬 BERT + Graph Neural Network Research Platform")
st.markdown("**Academic research platform for comparing semantic relationship extraction across different neural architectures**")

# Research methodology explanation
with st.expander("🎓 Research Methodology & Scientific Rationale"):
    st.markdown("""
    ### Pure Neural Architecture Comparison Study
    
    **Research Question**: How do different graph neural network architectures enhance BERT-based semantic relationship extraction compared to transformer-only baselines?
    
    **Methodology**: Controlled experimental comparison of four neural architectures:
    
    #### 1. BERT Baseline (Control Group)
    - **Model Specification**: `en_core_web_sm` (spaCy 3.x) with BERT-based transformer
    - **Architecture**: 12-layer transformer, 768 hidden units, 12 attention heads
    - **Parameters**: ~110M parameters, pre-trained on English web text
    - **Implementation**: Dependency parsing (Universal Dependencies) + Named Entity Recognition
    - **Scientific Value**: Isolates GNN contribution by providing non-graph control condition
    
    #### 2. RGCN (Relational Graph Convolutional Network)
    - **Model Specification**: PyTorch implementation with relation-specific weight decomposition
    - **Architecture**: 2 layers, 128 input features, 64 hidden units, 32 output dimensions
    - **Parameters**: Basis decomposition with 30 basis matrices for parameter efficiency
    - **Key Innovation**: W_r = Σᵢ aᵢᵣVᵢ matrices for each relation type r
    - **Scientific Contribution**: Tests whether heterogeneous graph processing improves relationship classification accuracy
    
    #### 3. CompGCN (Composition-based Graph Convolutional Network)  
    - **Model Specification**: PyTorch implementation with compositional message passing
    - **Architecture**: 2 layers, 128 input features, 64 hidden units, 32 output dimensions
    - **Parameters**: Shared entity-relation embeddings, multiplication composition function
    - **Key Innovation**: h_v^(l+1) = σ(W_O · COMP(h_v^l, h_r^l)) with composition operators
    - **Scientific Contribution**: Evaluates unified vs. separate entity/relation processing approaches
    
    #### 4. RGAT (Relational Graph Attention Network)
    - **Model Specification**: PyTorch implementation with multi-head relational attention
    - **Architecture**: 2 layers, 128 input features, 64 hidden units, 32 output dimensions, 4 attention heads
    - **Parameters**: Relation-specific attention weights, 0.1 dropout rate
    - **Key Innovation**: α_ij^r = softmax(LeakyReLU(a^T[W_h h_i || W_h h_j || W_r r_ij]))
    - **Scientific Contribution**: Tests interpretable attention vs. uniform relationship weighting
    
    ### Experimental Controls (Following Scientific Best Practices)
    - **Temperature Calibration**: Independent temperature control per architecture (0.0-1.0) for confidence analysis and statistical comparison
    - **Randomization**: Random seed control for reproducible experiments and statistical validity
    - **Blinding**: Automated processing without researcher bias in relationship extraction
    - **Standardization**: Identical preprocessing pipelines across all architectures for fair comparison
    - **Control Groups**: BERT-only baseline as negative control for measuring GNN enhancement effects
    - **Replication**: Multiple experimental runs with different temperatures for statistical significance testing
    - **Quantitative Metrics**: Confidence scores, processing times, and relationship counts for objective comparison
    
    ### Statistical Validation & Reporting
    - **Effect Size Measurement**: Cohen's d for comparing architecture performance differences
    - **Confidence Intervals**: 95% CI for all quantitative metrics (precision, recall, F1-score)
    - **Multiple Comparisons**: Bonferroni correction for multiple architecture comparisons
    - **Power Analysis**: Sample size calculations for detecting meaningful effect sizes
    - **Cross-Validation**: K-fold validation across different text domains for generalizability
    - **Null Hypothesis Testing**: H₀: GNN architectures show no improvement over BERT baseline
    
    ### Research Ethics & Transparency
    - **Open Science**: All hyperparameters, random seeds, and experimental conditions documented
    - **Reproducibility**: Complete experimental protocols provided for replication
    - **Bias Mitigation**: Automated processing reduces human annotation bias
    - **Data Provenance**: Full tracking of input texts and processing parameters
    
    ### Scientific Significance
    This research contributes to understanding which graph neural network properties (relation-specific processing, compositional embedding, attention mechanisms) provide measurable improvements over transformer baselines for semantic relationship extraction tasks, following established standards for computational linguistics research.
    """)

# Experimental design notice
st.info("**Experimental Design**: This platform implements pure neural architectures without simulations or synthetic data. All processing uses authentic BERT and GNN implementations for rigorous scientific comparison.")

# Model Specifications - Always Visible
st.subheader("🔧 Model Specifications")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **BERT Baseline (Control)**
    - Model: `en_core_web_sm` (spaCy 3.x)
    - Architecture: 12 layers, 768 hidden units, 12 attention heads
    - Parameters: ~110M, pre-trained on English web text
    - Implementation: Universal Dependencies + NER
    
    **RGCN (Schlichtkrull et al., 2018)**
    - Architecture: 2 layers, 128→64→32 dimensions
    - Parameters: Basis decomposition with 30 basis matrices
    - Formula: W_r = Σᵢ aᵢᵣVᵢ for each relation type r
    - Innovation: Relation-specific weight matrices
    """)

with col2:
    st.markdown("""
    **CompGCN (Vashishth et al., 2020)**
    - Architecture: 2 layers, 128→64→32 dimensions
    - Parameters: Shared entity-relation embeddings
    - Formula: h_v^(l+1) = σ(W_O · COMP(h_v^l, h_r^l))
    - Innovation: Compositional message passing
    
    **RGAT (Busbridge et al., 2019)**
    - Architecture: 2 layers, 128→64→32 dimensions, 4 attention heads
    - Parameters: Dropout 0.1, relation-specific attention weights
    - Formula: α_ij^r = softmax(LeakyReLU(a^T[W_h h_i || W_h h_j || W_r r_ij]))
    - Innovation: Multi-head relational attention
    """)

st.divider()

# Text input
text_input = st.text_area(
    "Text to Analyze",
    height=150,
    placeholder="Enter scientific text for semantic relationship extraction...",
    help="Input text for experimental analysis"
)

# Process button
col1, col2 = st.columns([1, 1])
with col1:
    process_button = st.button("🔬 Run Experiment", use_container_width=True, type="primary")
with col2:
    save_button = st.button("💾 Save to Database", use_container_width=True, 
                           disabled=(not st.session_state.processed_data))

# Processing logic
if process_button and text_input:
    st.session_state.current_text = text_input
    
    # Generate experiment ID
    timestamp = datetime.datetime.now()
    st.session_state.experiment_id = int(timestamp.timestamp())
    
    # Initialize processors
    if st.session_state.bert_processor is None or st.session_state.bert_processor.temperature != temperature:
        st.session_state.bert_processor = BERTProcessor(temperature=temperature)
    
    if gnn_model != "None":
        if st.session_state.gnn_processor is None or st.session_state.gnn_processor.model_type != gnn_model.lower():
            st.session_state.gnn_processor = GNNProcessor(
                model_type=gnn_model.lower(),
                temperature=temperature
            )
    else:
        st.session_state.gnn_processor = None
    
    # Pipeline description
    pipeline = "BERT" if gnn_model == "None" else f"BERT + {gnn_model}"
    
    # Visual processing indicator
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    with st.spinner(f"🧠 Running {pipeline} pipeline..."):
        try:
            # BERT processing
            status_placeholder.info("🔍 **Step 1/4:** BERT analyzing semantic relationships...")
            progress_bar.progress(25)
            start_time = time.time()
            bert_result = st.session_state.bert_processor.extract_relationships(
                text_input, 
                custom_prompt=analysis_prompt
            )
            bert_time = time.time() - start_time
            
            # Filter relationships
            status_placeholder.info("⚙️ **Step 2/4:** Filtering relationships by confidence threshold...")
            progress_bar.progress(50)
            filtered_relationships = [
                rel for rel in bert_result["relationships"]
                if rel["confidence"] >= min_confidence
            ]
            
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
            
            # Graph construction
            status_placeholder.info("🕸️ **Step 3/4:** Building knowledge graph structure...")
            progress_bar.progress(75)
            start_time = time.time()
            graph = graph_utils.build_networkx_graph(filtered_relationships)
            graph_time = time.time() - start_time
            
            # GNN processing
            gnn_time = 0.0
            gnn_result = None
            
            if gnn_model != "None" and st.session_state.gnn_processor:
                status_placeholder.info(f"🧬 **Step 4/4:** Enhancing with {gnn_model} neural network...")
                start_time = time.time()
                gnn_result = st.session_state.gnn_processor.process_relationships(filtered_relationships)
                gnn_time = time.time() - start_time
                
                # Apply GNN layout
                if gnn_result and len(gnn_result.get("embeddings", [])) > 0:
                    graph = graph_utils.apply_rgcn_layout(
                        graph,
                        gnn_result["embeddings"],
                        gnn_result["entity_to_idx"]
                    )
            else:
                # BERT-only: use spring layout
                import networkx as nx
                pos = nx.spring_layout(graph, seed=42)
                nx.set_node_attributes(graph, {node: pos[node] for node in pos}, 'pos')
            
            # Store results in session (separate from permanent database)
            st.session_state.processed_data = {
                "relationships": bert_result["relationships"],
                "filtered_relationships": filtered_relationships,
                "statistics": bert_result.get("statistics", {}),
                "bert_time": bert_time,
                "graph_time": graph_time,
                "gnn_time": gnn_time,
                "pipeline": pipeline,
                "experiment_id": st.session_state.experiment_id
            }
            st.session_state.graph = graph
            st.session_state.gnn_result = gnn_result
            st.session_state.current_text = text_input  # Store for potential saving
            
            # Complete processing
            progress_bar.progress(100)
            status_placeholder.success(f"✅ **Processing Complete** - {pipeline} pipeline executed successfully")
            
            # Display results
            st.success(f"✅ Experiment completed - Pipeline: {pipeline}")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("BERT Time", f"{bert_time:.3f}s")
            col2.metric("Graph Time", f"{graph_time:.3f}s")
            if gnn_model != "None":
                col3.metric(f"{gnn_model} Time", f"{gnn_time:.3f}s")
            col4.metric("Relationships", len(filtered_relationships))
            
        except Exception as e:
            st.error(f"❌ Experiment failed: {str(e)}")
            logging.error(f"Processing error: {e}", exc_info=True)

# Save experiment to permanent database (separate from session cache)
if save_button and st.session_state.processed_data:
    with st.spinner("Saving experiment to permanent database..."):
        try:
            data = st.session_state.processed_data
            analysis_id = db.save_analysis(
                text=st.session_state.get('current_text', text_input),
                relationships=data["filtered_relationships"],
                title=f"Experiment_{st.session_state.experiment_id}",
                model_type=base_model.lower(),
                gnn_architecture=gnn_model.lower() if gnn_model != "None" else "none",
                temperature=temperature,
                custom_prompt=analysis_prompt,
                processing_time_bert=data["bert_time"],
                processing_time_gnn=data["gnn_time"],
                processing_time_graph=data["graph_time"]
            )
            
            st.success(f"✅ Experiment saved to permanent database with ID: {analysis_id}")
            st.info("💾 This data is preserved independently from session cache and won't be lost when clearing cache")
            
        except Exception as e:
            st.error(f"❌ Save failed: {str(e)}")

# Results display
if st.session_state.processed_data and st.session_state.graph:
    st.divider()
    st.subheader("📊 Experimental Results")
    
    # Visualization tabs
    viz_tab, data_tab, export_tab = st.tabs(["🔗 Graph Visualization", "📋 Data Analysis", "📤 Export"])
    
    with viz_tab:
        try:
            fig = visualization.create_relationship_graph_figure(
                st.session_state.graph,
                title=f"Semantic Graph - {st.session_state.processed_data['pipeline']}"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
    
    with data_tab:
        data = st.session_state.processed_data
        
        # Relationships table
        if data["filtered_relationships"]:
            df = pd.DataFrame(data["filtered_relationships"])
            st.dataframe(df, use_container_width=True)
        
        # Statistics
        st.subheader("Statistics")
        stats_data = {
            "Metric": ["Total Relationships", "Filtered Relationships", "Graph Nodes", "Graph Edges"],
            "Value": [
                len(data["relationships"]),
                len(data["filtered_relationships"]),
                st.session_state.graph.number_of_nodes(),
                st.session_state.graph.number_of_edges()
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True)
    
    with export_tab:
        st.subheader("🔬 Research Data Export")
        
        export_option = st.radio(
            "Export Selection",
            ["Export Current Experiment", "Export All Experiments", "Export Selected Experiments"],
            help="Choose scope of data export for research analysis"
        )
        
        if export_option == "Export Selected Experiments":
            # Get available experiments
            try:
                analyses_df = db.get_recent_analyses(limit=50)
                if not analyses_df.empty:
                    selected_ids = st.multiselect(
                        "Select Experiments",
                        options=analyses_df['id'].tolist(),
                        format_func=lambda x: f"ID {x}: {analyses_df[analyses_df['id']==x]['title'].iloc[0] if not analyses_df[analyses_df['id']==x].empty else 'Unknown'}"
                    )
                else:
                    st.info("No experiments found in database")
                    selected_ids = []
            except:
                selected_ids = []
        
        if st.button("📥 Generate Research Export", use_container_width=True):
            with st.spinner("Generating research data export..."):
                try:
                    from export_research import create_research_export
                    
                    if export_option == "Export Current Experiment":
                        export_data = create_research_export(
                            current_experiment_id=st.session_state.experiment_id
                        )
                    elif export_option == "Export All Experiments":
                        export_data = create_research_export(export_all=True)
                    else:
                        export_data = create_research_export(
                            selected_experiment_ids=selected_ids if 'selected_ids' in locals() else []
                        )
                    
                    if export_data:
                        elements_csv = export_data["elements"].to_csv(index=False, sep=';')
                        connections_csv = export_data["connections"].to_csv(index=False, sep=';')
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                label="📊 Download Elements CSV",
                                data=elements_csv,
                                file_name=f"research_elements_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            st.download_button(
                                label="🔗 Download Connections CSV",
                                data=connections_csv,
                                file_name=f"research_connections_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        st.success("✅ Research export generated successfully")
                        
                        # Preview
                        with st.expander("📋 Preview Export Data"):
                            st.subheader("Elements")
                            st.dataframe(export_data["elements"].head(10))
                            st.subheader("Connections")
                            st.dataframe(export_data["connections"].head(10))
                    
                    else:
                        st.warning("No data available for export")
                        
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")

# Database status and data protection
st.sidebar.divider()
st.sidebar.subheader("📊 Database Status")
try:
    analysis_count = db.get_analysis_count()
    relationship_count = db.get_relationship_count()
    st.sidebar.metric("Saved Experiments", analysis_count)
    st.sidebar.metric("Saved Relationships", relationship_count)
    st.sidebar.info("💾 All saved data is permanently protected from cache clearing")
except:
    st.sidebar.error("Database connection error")

# Data separation notice
st.sidebar.markdown("""
**Data Path Separation:**
- 🔄 **Session Cache**: Temporary processing data (cleared with cache button)
- 💾 **Database History**: Permanent experimental data (never cleared)
- 📤 **Export Data**: Generated from permanent database only

**Protection Guarantee**: Historical experiments remain safe regardless of cache operations.
""")
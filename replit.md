# BERT+GNN Research Platform

## Overview

This is a scientific research platform that combines BERT-based text processing with Graph Neural Networks (GNNs) for semantic relationship extraction and analysis. The platform provides an interactive Streamlit interface for researchers to experiment with different model configurations, extract relationships from text, and export research data for academic analysis.

## System Architecture

### Frontend Architecture
- **Streamlit-based web interface** providing an interactive research environment
- Multi-page layout with sidebar configuration panel
- Real-time visualization of graphs and relationship networks
- Export functionality for research data in CSV format

### Backend Architecture
- **BERT Processor**: Uses spaCy's BERT-based models for semantic relationship extraction
- **GNN Models**: Implements multiple Graph Neural Network architectures:
  - RGCN (Relational Graph Convolutional Networks)
  - CompGCN (Composition-based Graph Convolutional Networks)
  - RGAT (Relational Graph Attention Networks)
- **Graph Processing**: NetworkX-based graph construction and analysis
- **Visualization Engine**: Plotly-based interactive graph visualizations

## Key Components

### 1. Text Processing Pipeline
- **BERT-based extraction** using spaCy's English language model
- Temperature-controlled processing for varying extraction diversity
- Relationship extraction with confidence scoring
- Polarity and directness classification

### 2. Graph Neural Networks
- **RGCN Layer**: Handles heterogeneous relationship types with basis decomposition
- **Multi-relational processing**: Supports different edge types in knowledge graphs
- **Attention mechanisms**: RGAT implementation for weighted relationship importance
- **Compositional embeddings**: CompGCN for joint entity-relation representation

### 3. Data Storage
- **SQLite database** for experiment persistence
- Tables for analyses, relationships, and entities
- Experiment tracking with model configurations and performance metrics
- Simple database interface without external dependencies

### 4. Research Export System
- **Academic CSV format** with elements and connections structure
- Experiment comparison across different model configurations
- Temperature study exports for parameter analysis
- Research data aggregation and statistics

## Data Flow

1. **Input Processing**: Text input → BERT relationship extraction → Structured relationships
2. **Graph Construction**: Relationships → NetworkX graph → Community detection
3. **GNN Enhancement**: Graph + Model selection → Enhanced embeddings → Improved relationships
4. **Visualization**: Processed data → Plotly visualizations → Interactive interface
5. **Export**: Research data → CSV format → Academic analysis

## External Dependencies

### Core Libraries
- **Streamlit**: Web interface framework
- **spaCy**: BERT-based NLP processing (en_core_web_sm model)
- **PyTorch**: Neural network implementations for GNN models
- **NetworkX**: Graph construction and analysis
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation and export
- **SQLite**: Local database storage

### Model Dependencies
- spaCy English model (en_core_web_sm) for BERT-based processing
- PyTorch CPU version for GNN computations
- No external API dependencies or cloud services required

## Deployment Strategy

### Replit Configuration
- **Python 3.11** runtime environment
- **Streamlit server** on port 5000 with autoscale deployment
- UV package manager for dependency resolution
- PyTorch CPU-only index for efficient installation

### Database Setup
- Automatic SQLite database initialization on first run
- Local file storage in `semantic_analyzer.db`
- No external database configuration required

### Model Loading
- Automatic spaCy model download if not present
- Local PyTorch model initialization
- Error handling for missing dependencies

## Recent Changes

- June 25, 2025: All export options thoroughly tested and validated working correctly
- June 25, 2025: Fixed critical "bert_result" undefined errors across all GNN models
- June 25, 2025: Replaced confusing confidence scores with clear quality indicators (Strong/Medium/Weak)
- June 25, 2025: Added openpyxl dependency for Excel export functionality
- June 24, 2025: Implemented data path separation with protected database history
- June 24, 2025: Added DataManager class for secure session/database separation
- June 24, 2025: Fixed temperature 0.0 division by zero error
- June 24, 2025: Added visual processing indicators with progress bar and step-by-step status
- June 21, 2025: Set default temperature to 0.0 for maximum confidence
- June 18, 2025: Added individual temperature controls for each model (0.0-1.0 range)
- June 18, 2025: Enhanced scientific methodology with complete model specifications
- June 18, 2025: Fixed export functionality to include authentic experimental data

## Changelog

- June 18, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.
Interface preference: Single text analysis field only, no optional prompt field.
Quality preference: Replace confusing confidence scores with simple quality indicators (strong/medium/weak).
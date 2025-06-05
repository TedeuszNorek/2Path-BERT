#!/usr/bin/env python3
"""
Comprehensive debugging script for GNN architectures and graph generation
"""

import sys
import traceback
import time
import networkx as nx
import pandas as pd
from typing import Dict, List, Any

# Import all our modules
from bert_processor import BERTProcessor
from gnn_models import GNNProcessor
import graph_utils
import visualization
import db_simple as db

def test_bert_processor():
    """Test BERT processor functionality"""
    print("=" * 60)
    print("TESTING BERT PROCESSOR")
    print("=" * 60)
    
    try:
        processor = BERTProcessor(temperature=1.0)
        
        test_text = """
        The researcher conducted experiments on machine learning algorithms.
        Neural networks showed better performance than traditional methods.
        Deep learning models require large datasets for training.
        """
        
        print(f"Input text: {test_text[:100]}...")
        
        start_time = time.time()
        result = processor.extract_relationships(test_text)
        processing_time = time.time() - start_time
        
        print(f"Processing time: {processing_time:.3f}s")
        print(f"Relationships found: {len(result['relationships'])}")
        
        if result['relationships']:
            print("\nSample relationships:")
            for i, rel in enumerate(result['relationships'][:3]):
                print(f"{i+1}. {rel['subject']} -> {rel['predicate']} -> {rel['object']} (conf: {rel['confidence']:.3f})")
        
        metrics = processor.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
        print("✓ BERT processor test PASSED")
        return True, result
        
    except Exception as e:
        print(f"✗ BERT processor test FAILED: {str(e)}")
        traceback.print_exc()
        return False, None

def test_gnn_processor(gnn_type: str, relationships: List[Dict[str, Any]]):
    """Test GNN processor functionality"""
    print(f"\n{'=' * 60}")
    print(f"TESTING {gnn_type.upper()} PROCESSOR")
    print("=" * 60)
    
    try:
        processor = GNNProcessor(
            model_type=gnn_type.lower(),
            temperature=1.0
        )
        
        print(f"GNN Type: {gnn_type}")
        print(f"Input relationships: {len(relationships)}")
        
        start_time = time.time()
        result = processor.process_relationships(relationships)
        processing_time = time.time() - start_time
        
        print(f"Processing time: {processing_time:.3f}s")
        print(f"Embeddings generated: {len(result['embeddings'])}")
        print(f"Entity mappings: {len(result['entity_to_idx'])}")
        print(f"Relation mappings: {len(result['relation_to_idx'])}")
        
        metrics = processor.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
        print(f"✓ {gnn_type.upper()} processor test PASSED")
        return True, result
        
    except Exception as e:
        print(f"✗ {gnn_type.upper()} processor test FAILED: {str(e)}")
        traceback.print_exc()
        return False, None

def test_graph_generation(relationships: List[Dict[str, Any]]):
    """Test graph generation functionality"""
    print(f"\n{'=' * 60}")
    print("TESTING GRAPH GENERATION")
    print("=" * 60)
    
    try:
        print(f"Building graph from {len(relationships)} relationships...")
        
        start_time = time.time()
        graph = graph_utils.build_networkx_graph(relationships)
        graph_time = time.time() - start_time
        
        print(f"Graph building time: {graph_time:.3f}s")
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")
        
        # Test different layouts
        layouts = {
            "spring": lambda g: nx.spring_layout(g, seed=42),
            "circular": lambda g: nx.circular_layout(g),
            "random": lambda g: nx.random_layout(g, seed=42)
        }
        
        for layout_name, layout_func in layouts.items():
            try:
                start_time = time.time()
                pos = layout_func(graph)
                layout_time = time.time() - start_time
                print(f"✓ {layout_name} layout: {layout_time:.3f}s")
            except Exception as e:
                print(f"✗ {layout_name} layout failed: {str(e)}")
        
        print("✓ Graph generation test PASSED")
        return True, graph
        
    except Exception as e:
        print(f"✗ Graph generation test FAILED: {str(e)}")
        traceback.print_exc()
        return False, None

def test_visualization(graph, relationships: List[Dict[str, Any]]):
    """Test visualization functionality"""
    print(f"\n{'=' * 60}")
    print("TESTING VISUALIZATION")
    print("=" * 60)
    
    try:
        # Test different visualization types
        viz_tests = [
            ("Graph", lambda: visualization.create_graph_visualization(graph)),
            ("Adjacency Matrix", lambda: visualization.create_adjacency_matrix(graph)),
            ("Bar Chart", lambda: visualization.create_bar_chart(relationships)),
            ("Sankey", lambda: visualization.create_sankey_diagram(relationships))
        ]
        
        for viz_name, viz_func in viz_tests:
            try:
                start_time = time.time()
                fig = viz_func()
                viz_time = time.time() - start_time
                
                if fig is not None:
                    print(f"✓ {viz_name} visualization: {viz_time:.3f}s")
                else:
                    print(f"? {viz_name} visualization returned None")
                    
            except Exception as e:
                print(f"✗ {viz_name} visualization failed: {str(e)}")
        
        print("✓ Visualization test COMPLETED")
        return True
        
    except Exception as e:
        print(f"✗ Visualization test FAILED: {str(e)}")
        traceback.print_exc()
        return False

def test_database_operations(relationships: List[Dict[str, Any]]):
    """Test database operations"""
    print(f"\n{'=' * 60}")
    print("TESTING DATABASE OPERATIONS")
    print("=" * 60)
    
    try:
        # Initialize database
        db.init_db()
        print("✓ Database initialized")
        
        # Save analysis
        analysis_id = db.save_analysis(
            text="Test analysis for debugging",
            relationships=relationships,
            title="Debug Test",
            model_type='bert',
            gnn_architecture='rgcn',
            temperature=1.0,
            custom_prompt="Debug test prompt",
            processing_time_bert=0.1,
            processing_time_gnn=0.05,
            processing_time_graph=0.02
        )
        print(f"✓ Analysis saved with ID: {analysis_id}")
        
        # Test retrieval
        analysis = db.get_analysis_by_id(analysis_id)
        print(f"✓ Analysis retrieved: {analysis}")
        
        # Test relationships retrieval
        rel_df = db.get_relationships_by_analysis_id(analysis_id)
        print(f"✓ Relationships retrieved: {len(rel_df)} rows")
        
        # Test counts
        analysis_count = db.get_analysis_count()
        rel_count = db.get_relationship_count()
        print(f"✓ Database counts - Analyses: {analysis_count}, Relationships: {rel_count}")
        
        print("✓ Database operations test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Database operations test FAILED: {str(e)}")
        traceback.print_exc()
        return False

def run_comprehensive_debug():
    """Run comprehensive debugging of all components"""
    print("🔧 STARTING COMPREHENSIVE DEBUGGING")
    print("=" * 80)
    
    results = {}
    
    # Test 1: BERT Processor
    bert_success, bert_result = test_bert_processor()
    results['BERT'] = bert_success
    
    if not bert_success:
        print("🚨 BERT processor failed - stopping tests")
        return results
    
    relationships = bert_result['relationships'][:10]  # Use first 10 relationships
    
    # Test 2: GNN Processors
    gnn_types = ['rgcn', 'compgcn', 'rgat']
    for gnn_type in gnn_types:
        gnn_success, gnn_result = test_gnn_processor(gnn_type, relationships)
        results[gnn_type.upper()] = gnn_success
    
    # Test 3: Graph Generation
    graph_success, graph = test_graph_generation(relationships)
    results['Graph Generation'] = graph_success
    
    # Test 4: Visualization
    if graph_success:
        viz_success = test_visualization(graph, relationships)
        results['Visualization'] = viz_success
    
    # Test 5: Database Operations
    db_success = test_database_operations(relationships)
    results['Database'] = db_success
    
    # Summary
    print(f"\n{'=' * 80}")
    print("🔍 DEBUGGING SUMMARY")
    print("=" * 80)
    
    for component, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{component:20} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - System is fully functional!")
    else:
        print("⚠️  Some tests failed - check errors above")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_debug()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)
#!/usr/bin/env python3
"""
Comprehensive test suite for BERT+GNN research platform
"""

import sys
import time
import traceback
from typing import Dict, List, Any

def test_bert_extraction():
    """Test BERT relationship extraction"""
    print("Testing BERT extraction...")
    
    try:
        from bert_processor import BERTProcessor
        
        processor = BERTProcessor(temperature=1.0)
        
        test_texts = [
            "Scientists conducted experiments on neural networks.",
            "Machine learning algorithms process large datasets efficiently.",
            "Researchers analyzed the performance of deep learning models."
        ]
        
        for i, text in enumerate(test_texts):
            result = processor.extract_relationships(text)
            relationships = result.get('relationships', [])
            print(f"Text {i+1}: {len(relationships)} relationships extracted")
            
            if relationships:
                sample = relationships[0]
                print(f"  Sample: {sample['subject']} -> {sample['predicate']} -> {sample['object']}")
        
        print("✓ BERT extraction working")
        return True, relationships[:5] if relationships else []
        
    except Exception as e:
        print(f"✗ BERT extraction failed: {e}")
        traceback.print_exc()
        return False, []

def test_gnn_architecture(gnn_type: str, relationships: List[Dict[str, Any]]):
    """Test specific GNN architecture"""
    print(f"Testing {gnn_type.upper()}...")
    
    try:
        from gnn_models import GNNProcessor
        
        processor = GNNProcessor(model_type=gnn_type.lower(), temperature=1.0)
        
        start_time = time.time()
        result = processor.process_relationships(relationships)
        processing_time = time.time() - start_time
        
        embeddings = result.get('embeddings', [])
        entities = result.get('entity_to_idx', {})
        relations = result.get('relation_to_idx', {})
        
        print(f"  Processing time: {processing_time:.3f}s")
        print(f"  Embeddings shape: {len(embeddings)} x {len(embeddings[0]) if len(embeddings) > 0 else 0}")
        print(f"  Entities: {len(entities)}")
        print(f"  Relations: {len(relations)}")
        
        if len(embeddings) > 0 and len(entities) > 0:
            print(f"✓ {gnn_type.upper()} working correctly")
            return True
        else:
            print(f"? {gnn_type.upper()} returned empty results")
            return False
            
    except Exception as e:
        print(f"✗ {gnn_type.upper()} failed: {e}")
        traceback.print_exc()
        return False

def test_graph_generation(relationships: List[Dict[str, Any]]):
    """Test graph construction and visualization"""
    print("Testing graph generation...")
    
    try:
        import graph_utils
        import visualization
        import networkx as nx
        
        # Build graph
        graph = graph_utils.build_networkx_graph(relationships)
        print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Test layouts
        layouts = ['spring', 'circular']
        for layout_name in layouts:
            if layout_name == 'spring':
                pos = nx.spring_layout(graph, seed=42)
            else:
                pos = nx.circular_layout(graph)
            print(f"  {layout_name} layout: {len(pos)} positions")
        
        # Test visualization
        fig = visualization.create_relationship_graph_figure(graph)
        if fig:
            print("  Visualization created successfully")
        
        print("✓ Graph generation working")
        return True
        
    except Exception as e:
        print(f"✗ Graph generation failed: {e}")
        traceback.print_exc()
        return False

def test_database_operations():
    """Test database functionality"""
    print("Testing database operations...")
    
    try:
        import db_simple as db
        
        # Initialize
        db.init_db()
        print("  Database initialized")
        
        # Test save
        test_relationships = [
            {
                'subject': 'test_subject',
                'predicate': 'test_predicate', 
                'object': 'test_object',
                'confidence': 0.9,
                'polarity': 'positive',
                'directness': 'direct',
                'sentence': 'Test sentence'
            }
        ]
        
        analysis_id = db.save_analysis(
            text="Test analysis text",
            relationships=test_relationships,
            title="Test Analysis",
            model_type='bert',
            gnn_architecture='rgcn',
            temperature=1.0,
            custom_prompt="Test prompt",
            processing_time_bert=0.1,
            processing_time_gnn=0.05,
            processing_time_graph=0.02
        )
        
        print(f"  Analysis saved with ID: {analysis_id}")
        
        # Test retrieval
        retrieved = db.get_analysis_by_id(analysis_id)
        print(f"  Analysis retrieved: {bool(retrieved)}")
        
        relationships_df = db.get_relationships_by_analysis_id(analysis_id)
        print(f"  Relationships retrieved: {len(relationships_df)}")
        
        print("✓ Database operations working")
        return True
        
    except Exception as e:
        print(f"✗ Database operations failed: {e}")
        traceback.print_exc()
        return False

def test_export_functionality():
    """Test research export functionality"""
    print("Testing export functionality...")
    
    try:
        from export_research import create_research_export
        
        # Test export all
        export_data = create_research_export(export_all=True)
        
        elements_df = export_data.get('elements')
        connections_df = export_data.get('connections')
        
        print(f"  Elements exported: {len(elements_df) if elements_df is not None else 0}")
        print(f"  Connections exported: {len(connections_df) if connections_df is not None else 0}")
        
        if elements_df is not None and connections_df is not None:
            print("✓ Export functionality working")
            return True
        else:
            print("? Export returned empty data")
            return False
            
    except Exception as e:
        print(f"✗ Export functionality failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests comprehensively"""
    print("=" * 60)
    print("BERT+GNN RESEARCH PLATFORM COMPREHENSIVE TEST")
    print("=" * 60)
    
    results = {}
    
    # Test 1: BERT extraction
    bert_success, sample_relationships = test_bert_extraction()
    results['BERT'] = bert_success
    print("-" * 40)
    
    if not bert_success:
        print("❌ BERT failed - cannot continue with GNN tests")
        return results
    
    # Test 2: GNN architectures
    gnn_types = ['rgcn', 'compgcn', 'rgat']
    for gnn_type in gnn_types:
        gnn_success = test_gnn_architecture(gnn_type, sample_relationships)
        results[gnn_type.upper()] = gnn_success
        print("-" * 40)
    
    # Test 3: Graph generation
    graph_success = test_graph_generation(sample_relationships)
    results['Graph Generation'] = graph_success
    print("-" * 40)
    
    # Test 4: Database operations
    db_success = test_database_operations()
    results['Database'] = db_success
    print("-" * 40)
    
    # Test 5: Export functionality
    export_success = test_export_functionality()
    results['Export'] = export_success
    print("-" * 40)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for component, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{component:20} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL SYSTEMS OPERATIONAL - Research platform ready")
    else:
        print("⚠️  Some components need attention")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)
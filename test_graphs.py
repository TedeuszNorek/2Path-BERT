#!/usr/bin/env python3
"""
Simple test script to verify graph generation functionality
"""

def test_basic_functionality():
    """Test basic components without heavy dependencies"""
    print("Testing basic graph generation...")
    
    # Test networkx directly
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        pos = nx.spring_layout(G)
        print(f"Basic NetworkX test: {len(G.nodes())} nodes, {len(G.edges())} edges")
        print("NetworkX working correctly")
        return True
    except Exception as e:
        print(f"NetworkX error: {e}")
        return False

def test_visualization():
    """Test plotly visualization"""
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode='markers'))
        print("Basic Plotly test successful")
        return True
    except Exception as e:
        print(f"Plotly error: {e}")
        return False

def test_bert_minimal():
    """Test BERT processor with minimal overhead"""
    try:
        import spacy
        # Try to load the model
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Scientists study algorithms.")
        print(f"spaCy test: {len(doc)} tokens processed")
        return True
    except Exception as e:
        print(f"spaCy/BERT error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("GRAPH GENERATION DIAGNOSTICS")
    print("=" * 50)
    
    tests = [
        ("NetworkX", test_basic_functionality),
        ("Plotly", test_visualization),
        ("spaCy/BERT", test_bert_minimal)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"{test_name} test failed with error: {e}")
            results[test_name] = False
        print("-" * 30)
    
    print("\nTEST RESULTS:")
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\nAll core components working - graph generation should function")
    else:
        print("\nSome components failing - this explains graph generation issues")
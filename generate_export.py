#!/usr/bin/env python3
"""
Generate CSV/Excel exports from research experiments
"""

import pandas as pd
import datetime
import db_simple as db
from typing import Dict, List, Any, Optional

def generate_research_export():
    """Generate comprehensive research export with actual data"""
    
    print("Generating research export from database...")
    
    # Get all experiments from database
    try:
        analyses_df = db.get_recent_analyses(limit=1000)
        print(f"Found {len(analyses_df)} experiments in database")
        
        if analyses_df.empty:
            print("No experiments found. Running sample analysis...")
            # Generate sample data if no experiments exist
            from bert_processor import BERTProcessor
            from gnn_models import GNNProcessor
            
            bert = BERTProcessor(temperature=1.0)
            sample_text = """
            Scientists conducted comprehensive experiments on neural networks and machine learning algorithms.
            The research team analyzed the performance of deep learning models across various datasets.
            Graph neural networks showed promising results in relationship extraction tasks.
            Transformer architectures demonstrated superior semantic understanding capabilities.
            """
            
            # Run experiments with different configurations
            configurations = [
                ("BERT", "none"),
                ("BERT", "rgcn"), 
                ("BERT", "compgcn"),
                ("BERT", "rgat")
            ]
            
            for base_model, gnn_arch in configurations:
                print(f"Running {base_model} + {gnn_arch} experiment...")
                
                # BERT extraction
                result = bert.extract_relationships(sample_text)
                relationships = result['relationships'][:10]  # Limit for demo
                
                # GNN processing if not "none"
                if gnn_arch != "none":
                    gnn = GNNProcessor(model_type=gnn_arch, temperature=1.0)
                    gnn_result = gnn.process_relationships(relationships)
                
                # Save to database
                analysis_id = db.save_analysis(
                    text=sample_text.strip(),
                    relationships=relationships,
                    title=f"Sample_{base_model}_{gnn_arch}",
                    model_type=base_model.lower(),
                    gnn_architecture=gnn_arch,
                    temperature=1.0,
                    custom_prompt="Sample research experiment",
                    processing_time_bert=0.1,
                    processing_time_gnn=0.05 if gnn_arch != "none" else 0.0,
                    processing_time_graph=0.02
                )
                print(f"  Saved as experiment ID: {analysis_id}")
            
            # Refresh analyses after generating sample data
            analyses_df = db.get_recent_analyses(limit=1000)
        
        # Generate Elements CSV
        elements_data = []
        connections_data = []
        
        for _, analysis in analyses_df.iterrows():
            exp_id = analysis['id']
            
            # Get relationships for this experiment
            relationships_df = db.get_relationships_by_analysis_id(exp_id)
            
            if relationships_df.empty:
                continue
            
            # Extract metadata
            base_model = analysis.get('model_type', 'bert').upper()
            gnn_arch = analysis.get('gnn_architecture', 'none')
            model_config = gnn_arch.upper() if gnn_arch != 'none' else 'None'
            temperature = analysis.get('temperature', 1.0)
            analysis_prompt = analysis.get('custom_prompt', '')
            text_content = analysis.get('text', '')
            timestamp = analysis.get('timestamp', datetime.datetime.now().isoformat())
            
            # Collect unique entities
            subjects = set()
            objects = set()
            
            for _, rel in relationships_df.iterrows():
                if rel.get('subject'):
                    subjects.add(rel['subject'])
                if rel.get('object'):
                    objects.add(rel['object'])
            
            # Create Elements entries
            unique_entities = subjects.union(objects)
            for entity in unique_entities:
                elements_data.append({
                    'Label': entity,
                    'Type': '',
                    'Tags': '',
                    'Description': '',
                    'Trial ID': exp_id,
                    'Base Model': base_model,
                    'Model Configuration': model_config,
                    'Analysis prompt': analysis_prompt,
                    'Text': text_content,
                    'Temperature': temperature,
                    '[Engineer]': '',
                    'Timestamp': timestamp
                })
            
            # Create Connections entries
            for _, rel in relationships_df.iterrows():
                connections_data.append({
                    'From': rel.get('subject', ''),
                    'To': rel.get('object', ''),
                    'Direction': rel.get('polarity', ''),
                    'Label': '',
                    'Type': '',
                    'Tags': '',
                    'Description': '',
                    'Experiment ID': exp_id,
                    'Base Model': base_model,
                    'Model Configuration': model_config,
                    'Link': rel.get('predicate', ''),
                    'Analysis prompt': analysis_prompt,
                    'Text': text_content,
                    'Temperature': temperature,
                    '[Engineer]': '',
                    'Timestamp': timestamp
                })
        
        # Create DataFrames
        elements_df = pd.DataFrame(elements_data)
        connections_df = pd.DataFrame(connections_data)
        
        # Generate timestamps for filenames
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save CSV files
        elements_filename = f'research_elements_{timestamp_str}.csv'
        connections_filename = f'research_connections_{timestamp_str}.csv'
        
        elements_df.to_csv(elements_filename, index=False, sep=';', encoding='utf-8')
        connections_df.to_csv(connections_filename, index=False, sep=';', encoding='utf-8')
        
        print(f"\n✅ Export completed successfully!")
        print(f"📊 Elements CSV: {elements_filename} ({len(elements_df)} rows)")
        print(f"🔗 Connections CSV: {connections_filename} ({len(connections_df)} rows)")
        
        # Also save as Excel
        excel_filename = f'research_export_{timestamp_str}.xlsx'
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            elements_df.to_excel(writer, sheet_name='Elements', index=False)
            connections_df.to_excel(writer, sheet_name='Connections', index=False)
        
        print(f"📈 Excel file: {excel_filename}")
        
        # Display summary statistics
        print(f"\n📈 Export Statistics:")
        print(f"- Total experiments: {len(analyses_df)}")
        print(f"- Unique entities: {len(elements_df)}")
        print(f"- Total connections: {len(connections_df)}")
        
        if not elements_df.empty:
            print(f"- Model configurations: {list(elements_df['Model Configuration'].unique())}")
        
        # Preview data
        print(f"\n📋 Preview - Elements (first 5 rows):")
        print(elements_df.head().to_string(index=False))
        
        print(f"\n📋 Preview - Connections (first 5 rows):")
        print(connections_df.head().to_string(index=False))
        
        return {
            'elements_file': elements_filename,
            'connections_file': connections_filename,
            'excel_file': excel_filename,
            'elements_df': elements_df,
            'connections_df': connections_df
        }
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = generate_research_export()
    if result:
        print(f"\n🎉 Research export completed successfully!")
        print(f"Files generated: {result['elements_file']}, {result['connections_file']}, {result['excel_file']}")
    else:
        print("❌ Export failed")
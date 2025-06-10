import sqlite3
import pandas as pd
import datetime

def export_research_data():
    """Generate CSV exports from research database"""
    
    # Connect to database
    conn = sqlite3.connect('semantic_analyzer.db')
    
    try:
        # Get all analyses
        analyses_query = """
        SELECT id, text, title, model_type, gnn_architecture, temperature, 
               custom_prompt, timestamp, processing_time_bert, processing_time_gnn
        FROM analyses
        ORDER BY timestamp DESC
        """
        analyses_df = pd.read_sql_query(analyses_query, conn)
        
        # Get all relationships
        relationships_query = """
        SELECT r.*, a.model_type, a.gnn_architecture, a.temperature, 
               a.custom_prompt, a.text as analysis_text, a.timestamp
        FROM relationships r
        JOIN analyses a ON r.analysis_id = a.id
        ORDER BY a.timestamp DESC, r.id
        """
        relationships_df = pd.read_sql_query(relationships_query, conn)
        
        print(f"Found {len(analyses_df)} experiments and {len(relationships_df)} relationships")
        
        # Create Elements data
        elements_data = []
        connections_data = []
        
        for _, analysis in analyses_df.iterrows():
            exp_id = analysis['id']
            base_model = analysis['model_type'].upper() if analysis['model_type'] else 'BERT'
            gnn_arch = analysis['gnn_architecture'] if analysis['gnn_architecture'] else 'none'
            model_config = gnn_arch.upper() if gnn_arch != 'none' else 'None'
            
            # Get relationships for this experiment
            exp_relationships = relationships_df[relationships_df['analysis_id'] == exp_id]
            
            # Collect unique entities
            entities = set()
            for _, rel in exp_relationships.iterrows():
                if rel['subject']:
                    entities.add(rel['subject'])
                if rel['object']:
                    entities.add(rel['object'])
            
            # Add elements
            for entity in entities:
                elements_data.append({
                    'Label': entity,
                    'Type': '',
                    'Tags': '',
                    'Description': '',
                    'Trial ID': exp_id,
                    'Base Model': base_model,
                    'Model Configuration': model_config,
                    'Analysis prompt': analysis['custom_prompt'] or '',
                    'Text': analysis['text'] or '',
                    'Temperature': analysis['temperature'] or 1.0,
                    '[Engineer]': '',
                    'Timestamp': analysis['timestamp'] or ''
                })
            
            # Add connections
            for _, rel in exp_relationships.iterrows():
                connections_data.append({
                    'From': rel['subject'] or '',
                    'To': rel['object'] or '',
                    'Direction': rel['polarity'] or '',
                    'Label': '',
                    'Type': '',
                    'Tags': '',
                    'Description': '',
                    'Experiment ID': exp_id,
                    'Base Model': base_model,
                    'Model Configuration': model_config,
                    'Link': rel['predicate'] or '',
                    'Analysis prompt': analysis['custom_prompt'] or '',
                    'Text': analysis['text'] or '',
                    'Temperature': analysis['temperature'] or 1.0,
                    '[Engineer]': '',
                    'Timestamp': analysis['timestamp'] or ''
                })
        
        # Create DataFrames
        elements_df = pd.DataFrame(elements_data)
        connections_df = pd.DataFrame(connections_data)
        
        # Generate filenames
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        elements_file = f'research_elements_{timestamp}.csv'
        connections_file = f'research_connections_{timestamp}.csv'
        excel_file = f'research_export_{timestamp}.xlsx'
        
        # Save CSV files
        elements_df.to_csv(elements_file, index=False, sep=';', encoding='utf-8')
        connections_df.to_csv(connections_file, index=False, sep=';', encoding='utf-8')
        
        # Save Excel file
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            elements_df.to_excel(writer, sheet_name='Elements', index=False)
            connections_df.to_excel(writer, sheet_name='Connections', index=False)
        
        print(f"Generated files:")
        print(f"- {elements_file} ({len(elements_df)} elements)")
        print(f"- {connections_file} ({len(connections_df)} connections)")
        print(f"- {excel_file}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    export_research_data()
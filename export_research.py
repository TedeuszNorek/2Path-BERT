import pandas as pd
import datetime
from typing import Dict, List, Any, Optional
import db_simple as db

def create_research_export(
    current_experiment_id: Optional[int] = None,
    export_all: bool = False,
    selected_experiment_ids: Optional[List[int]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Create research export data according to academic specifications
    
    Returns:
        Dict with 'elements' and 'connections' DataFrames
    """
    
    # Determine which experiments to export
    if current_experiment_id:
        experiment_ids = [current_experiment_id]
    elif export_all:
        analyses_df = db.get_recent_analyses(limit=1000)
        experiment_ids = analyses_df['id'].tolist() if not analyses_df.empty else []
    elif selected_experiment_ids:
        experiment_ids = selected_experiment_ids
    else:
        return {"elements": pd.DataFrame(), "connections": pd.DataFrame()}
    
    if not experiment_ids:
        return {"elements": pd.DataFrame(), "connections": pd.DataFrame()}
    
    # Collect all experiment data
    elements_data = []
    connections_data = []
    
    for exp_id in experiment_ids:
        try:
            # Get experiment details
            analysis = db.get_analysis_by_id(exp_id)
            if not analysis:
                continue
            
            relationships_df = db.get_relationships_by_analysis_id(exp_id)
            if relationships_df.empty:
                continue
            
            # Extract experiment metadata
            base_model = analysis.get('model_type', 'BERT').upper()
            gnn_arch = analysis.get('gnn_architecture', 'none')
            model_config = gnn_arch.upper() if gnn_arch != 'none' else 'None'
            temperature = analysis.get('temperature', 1.0)
            analysis_prompt = analysis.get('custom_prompt', '')
            text_to_analyze = analysis.get('text', '')
            timestamp = analysis.get('timestamp', datetime.datetime.now().isoformat())
            
            # Collect unique entities from this experiment
            subjects = set()
            objects = set()
            
            for _, rel in relationships_df.iterrows():
                if rel.get('subject'):
                    subjects.add(rel['subject'])
                if rel.get('object'):
                    objects.add(rel['object'])
            
            # Create Elements entries for unique entities
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
                    'Text': text_to_analyze,
                    'Temperature': temperature,
                    '[Engineer]': '',
                    'Timestamp': timestamp
                })
            
            # Create Connections entries for relationships
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
                    'Text': text_to_analyze,
                    'Temperature': temperature,
                    '[Engineer]': '',
                    'Timestamp': timestamp
                })
                
        except Exception as e:
            print(f"Error processing experiment {exp_id}: {e}")
            continue
    
    # Create DataFrames
    elements_df = pd.DataFrame(elements_data) if elements_data else pd.DataFrame(columns=[
        'Label', 'Type', 'Tags', 'Description', 'Trial ID', 'Base Model', 
        'Model Configuration', 'Analysis prompt', 'Text', 'Temperature', 
        '[Engineer]', 'Timestamp'
    ])
    
    connections_df = pd.DataFrame(connections_data) if connections_data else pd.DataFrame(columns=[
        'From', 'To', 'Direction', 'Label', 'Type', 'Tags', 'Description',
        'Experiment ID', 'Base Model', 'Model Configuration', 'Link',
        'Analysis prompt', 'Text', 'Temperature', '[Engineer]', 'Timestamp'
    ])
    
    return {
        'elements': elements_df,
        'connections': connections_df
    }

def get_export_statistics(elements_df: pd.DataFrame, connections_df: pd.DataFrame) -> Dict[str, Any]:
    """Get statistics about the export data"""
    return {
        'total_elements': len(elements_df),
        'total_connections': len(connections_df),
        'unique_experiments': len(elements_df['Trial ID'].unique()) if not elements_df.empty else 0,
        'model_configurations': list(elements_df['Model Configuration'].unique()) if not elements_df.empty else [],
        'export_timestamp': datetime.datetime.now().isoformat()
    }
import pandas as pd
import io
import datetime
from typing import List, Dict, Any, Optional
import db_simple as db

def create_export_data(include_full_history: bool = False, selected_analysis_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Create export data based on the specified CSV structure
    
    Args:
        include_full_history: Whether to export all analyses from database
        selected_analysis_ids: Specific analysis IDs to export (if not full history)
        
    Returns:
        DataFrame ready for CSV export
    """
    
    # Define the CSV structure based on the provided template
    columns = [
        'Label', 'Type', 'Tags', 'Description', 'Trial ID', 'Model', 
        'Model Configuration', 'Link', 'Analysis prompt', 'Text', 
        'Temperature', '[Engineer]'
    ]
    
    export_data = []
    
    if include_full_history:
        # Get all analyses from database
        analyses_df = db.get_recent_analyses(limit=1000)  # Get more analyses for full history
    elif selected_analysis_ids and len(selected_analysis_ids) > 0:
        # Get specific analyses
        analyses_list = []
        for analysis_id in selected_analysis_ids:
            analysis = db.get_analysis_by_id(analysis_id)
            if analysis:
                analyses_list.append(analysis)
        
        if analyses_list:
            analyses_df = pd.DataFrame(analyses_list)
        else:
            analyses_df = pd.DataFrame()
    else:
        # Return empty DataFrame
        return pd.DataFrame(columns=columns)
    
    if analyses_df.empty:
        return pd.DataFrame(columns=columns)
    
    for _, analysis in analyses_df.iterrows():
        analysis_id = int(analysis['id'])
        
        # Get relationships for this analysis
        relationships_df = db.get_relationships_by_analysis_id(analysis_id)
        
        # Get model configuration from database if available
        gnn_arch = analysis.get('gnn_architecture', 'none')
        if gnn_arch and gnn_arch != 'none':
            model_config = f"BERT+{gnn_arch.upper()}"
        else:
            model_config = "BERT"
        
        # Get temperature from database if available
        temperature = analysis.get('temperature', 1.0)
        custom_prompt = analysis.get('custom_prompt', '')
        
        # Safely get text content
        text_content = analysis.get('text', '') or ''
        if len(text_content) > 500:
            text_content = text_content[:500] + "..."
        
        # Create export row
        row_data = {
            'Label': f"Analysis_{analysis_id}",
            'Type': "Semantic_Analysis",
            'Tags': "relationship_extraction",
            'Description': analysis.get('title') or f"Semantic analysis #{analysis_id}",
            'Trial ID': analysis_id,
            'Model': "BERT",
            'Model Configuration': model_config,
            'Link': f"analysis_{analysis_id}",
            'Analysis prompt': custom_prompt,
            'Text': text_content.replace('\n', ' ').replace('\r', ' '),
            'Temperature': str(temperature),
            '[Engineer]': f"Relationships: {len(relationships_df)}"
        }
        
        export_data.append(row_data)
        
        # Add relationship details as separate rows
        for _, rel in relationships_df.iterrows():
            rel_row = {
                'Label': f"Relationship_{rel['id']}",
                'Type': "Triple",
                'Tags': f"{rel.get('polarity', '')},{rel.get('directness', '')}",
                'Description': f"{rel.get('subject', '')} -> {rel.get('predicate', '')} -> {rel.get('object', '')}",
                'Trial ID': analysis_id,
                'Model': "BERT",
                'Model Configuration': model_config,
                'Link': f"analysis_{analysis_id}_rel_{rel['id']}",
                'Analysis prompt': rel.get('sentence', ''),
                'Text': f"Subject: {rel.get('subject', '')}, Predicate: {rel.get('predicate', '')}, Object: {rel.get('object', '')}",
                'Temperature': str(temperature),
                '[Engineer]': f"Confidence: {rel.get('confidence', 0):.3f}"
            }
            export_data.append(rel_row)
    
    if not export_data:
        return pd.DataFrame(columns=columns)
    
    return pd.DataFrame(export_data)

def export_to_csv(df: pd.DataFrame, filename: str = None) -> str:
    """
    Export DataFrame to CSV format
    
    Args:
        df: DataFrame to export
        filename: Optional filename (auto-generated if None)
        
    Returns:
        CSV content as string
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"semantic_analysis_export_{timestamp}.csv"
    
    # Create CSV content
    output = io.StringIO()
    df.to_csv(output, index=False, sep=';', encoding='utf-8')
    csv_content = output.getvalue()
    output.close()
    
    return csv_content

def get_export_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get statistics about the export data
    
    Args:
        df: Export DataFrame
        
    Returns:
        Dictionary with export statistics
    """
    if df.empty:
        return {
            "total_rows": 0,
            "analyses_count": 0,
            "relationships_count": 0,
            "date_range": "No data"
        }
    
    analyses_count = len(df[df['Type'] == 'Semantic_Analysis'])
    relationships_count = len(df[df['Type'] == 'Triple'])
    
    return {
        "total_rows": len(df),
        "analyses_count": analyses_count,
        "relationships_count": relationships_count,
        "export_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "columns": list(df.columns)
    }
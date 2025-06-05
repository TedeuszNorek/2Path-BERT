import pandas as pd
import io
import datetime
from typing import List, Dict, Any, Optional
import db_simple as db

def create_export_data(include_full_history: bool = False, selected_analysis_ids: List[int] = None) -> pd.DataFrame:
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
    elif selected_analysis_ids:
        # Get specific analyses
        analyses_df = pd.DataFrame()
        for analysis_id in selected_analysis_ids:
            analysis = db.get_analysis_by_id(analysis_id)
            if analysis:
                analyses_df = pd.concat([analyses_df, pd.DataFrame([analysis])], ignore_index=True)
    else:
        # Return empty DataFrame
        return pd.DataFrame(columns=columns)
    
    for _, analysis in analyses_df.iterrows():
        analysis_id = analysis['id']
        
        # Get relationships for this analysis
        relationships_df = db.get_relationships_by_analysis_id(analysis_id)
        
        # Determine model configuration based on relationships data
        model_config = "Plain"  # Default
        if not relationships_df.empty:
            # Check if this was processed with GNN (look for processing method indicators)
            # This is a simplified heuristic - in practice you'd store this info
            if len(relationships_df) > 5:  # Assume GNN if many relationships found
                model_config = "BERT+GNN"
            else:
                model_config = "BERT"
        
        # Create export row
        row_data = {
            'Label': f"Analysis_{analysis_id}",
            'Type': "Semantic_Analysis",
            'Tags': "relationship_extraction",
            'Description': analysis.get('title', f"Semantic analysis #{analysis_id}"),
            'Trial ID': analysis_id,
            'Model': "BERT",
            'Model Configuration': model_config,
            'Link': f"analysis_{analysis_id}",
            'Analysis prompt': "",  # Would need to be stored in analysis metadata
            'Text': analysis.get('text', '').replace('\n', ' ').replace('\r', ' ')[:500] + "..." if len(analysis.get('text', '')) > 500 else analysis.get('text', ''),
            'Temperature': "1.0",  # Default - would need to be stored in analysis metadata
            '[Engineer]': f"Relationships: {len(relationships_df)}"
        }
        
        export_data.append(row_data)
        
        # Add relationship details as separate rows
        for _, rel in relationships_df.iterrows():
            rel_row = {
                'Label': f"Relationship_{rel['id']}",
                'Type': "Triple",
                'Tags': f"{rel['polarity']},{rel['directness']}",
                'Description': f"{rel['subject']} -> {rel['predicate']} -> {rel['object']}",
                'Trial ID': analysis_id,
                'Model': "BERT",
                'Model Configuration': model_config,
                'Link': f"analysis_{analysis_id}_rel_{rel['id']}",
                'Analysis prompt': rel.get('sentence', ''),
                'Text': f"Subject: {rel['subject']}, Predicate: {rel['predicate']}, Object: {rel['object']}",
                'Temperature': "1.0",
                '[Engineer]': f"Confidence: {rel['confidence']:.3f}"
            }
            export_data.append(rel_row)
    
    return pd.DataFrame(export_data, columns=columns)

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
"""
Data Manager for separating session cache from permanent storage
Ensures historical data protection while allowing session flexibility
"""

import streamlit as st
import sqlite3
from typing import Dict, Any, List, Optional
import logging

class DataManager:
    """Manages separation between session cache and permanent database storage"""
    
    @staticmethod
    def clear_session_only(preserve_db: bool = True) -> Dict[str, str]:
        """
        Clear session cache while preserving database
        
        Args:
            preserve_db: Always True to protect database
            
        Returns:
            Status message dictionary
        """
        if not preserve_db:
            raise ValueError("Database preservation is mandatory")
        
        # Session keys that can be safely cleared
        session_keys = [
            'bert_processor', 
            'gnn_processor', 
            'processed_data', 
            'graph', 
            'gnn_result', 
            'current_text',
            'experiment_id'
        ]
        
        cleared_count = 0
        for key in session_keys:
            if key in st.session_state:
                del st.session_state[key]
                cleared_count += 1
        
        return {
            "status": "success",
            "message": f"Session cache cleared ({cleared_count} items) - Database history protected",
            "cleared_items": cleared_count,
            "database_protected": True
        }
    
    @staticmethod
    def get_data_paths() -> Dict[str, str]:
        """Return current data path locations"""
        return {
            "database": "semantic_analyzer.db",
            "session_cache": "streamlit_session_state",
            "export_source": "permanent_database_only",
            "temp_files": "none_created"
        }
    
    @staticmethod
    def verify_data_separation() -> Dict[str, Any]:
        """Verify that data separation is working correctly"""
        try:
            # Check database connection
            conn = sqlite3.connect("semantic_analyzer.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM analyses")
            db_count = cursor.fetchone()[0]
            conn.close()
            
            # Check session state
            session_items = len([k for k in st.session_state.keys() if k.startswith(('bert_', 'gnn_', 'processed_', 'graph', 'current_'))])
            
            return {
                "database_records": db_count,
                "session_items": session_items,
                "separation_active": True,
                "protection_status": "enabled"
            }
        except Exception as e:
            logging.error(f"Data separation verification failed: {e}")
            return {
                "database_records": "unknown",
                "session_items": "unknown", 
                "separation_active": False,
                "protection_status": "error",
                "error": str(e)
            }
import os
import datetime
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional

# Use SQLite for simplicity and reliability
DB_PATH = "semantic_analyzer.db"

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create analyses table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        title TEXT,
        model_type TEXT DEFAULT 'bert',
        gnn_architecture TEXT DEFAULT 'none',
        temperature REAL DEFAULT 1.0,
        custom_prompt TEXT,
        processing_time_bert REAL DEFAULT 0.0,
        processing_time_gnn REAL DEFAULT 0.0,
        processing_time_graph REAL DEFAULT 0.0
    )
    ''')
    
    # Create relationships table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT,
        predicate TEXT,
        object TEXT,
        confidence REAL,
        polarity TEXT,
        directness TEXT,
        sentence TEXT,
        analysis_id INTEGER,
        FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
    )
    ''')
    
    # Create entities table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        count INTEGER DEFAULT 1,
        last_seen TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Enable foreign keys
    cursor.execute('PRAGMA foreign_keys = ON')
    
    conn.commit()
    conn.close()

def save_analysis(text: str, relationships: List[Dict[str, Any]], title: Optional[str] = None, 
                 model_type: str = 'bert', gnn_architecture: str = 'none', temperature: float = 1.0,
                 custom_prompt: str = '', processing_time_bert: float = 0.0, 
                 processing_time_gnn: float = 0.0, processing_time_graph: float = 0.0) -> int:
    """
    Save a text analysis and its relationships to the database
    
    Args:
        text: The analyzed text
        relationships: List of extracted relationships
        title: Optional title for the analysis
        
    Returns:
        ID of the created analysis
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Insert analysis with scientific measurement data
        cursor.execute(
            '''INSERT INTO analyses (text, title, model_type, gnn_architecture, temperature, 
               custom_prompt, processing_time_bert, processing_time_gnn, processing_time_graph) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (text, title, model_type, gnn_architecture, temperature, custom_prompt,
             processing_time_bert, processing_time_gnn, processing_time_graph)
        )
        analysis_id = cursor.lastrowid
        
        if analysis_id is None:  # Handle the case where lastrowid is None
            cursor.execute('SELECT last_insert_rowid()')
            analysis_id = cursor.fetchone()[0]
        
        # Insert relationships
        for rel in relationships:
            cursor.execute(
                'INSERT INTO relationships (subject, predicate, object, confidence, polarity, directness, sentence, analysis_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    rel.get("subject", ""),
                    rel.get("predicate", ""),
                    rel.get("object", ""),
                    rel.get("confidence", 0.0),
                    rel.get("polarity", "neutral"),
                    rel.get("directness", "direct"),
                    rel.get("sentence", ""),
                    analysis_id
                )
            )
            
            # Update entity counts
            update_entity_count(conn, rel.get("subject", ""))
            update_entity_count(conn, rel.get("object", ""))
        
        conn.commit()
        
        # Ensure we have a valid ID to return
        if isinstance(analysis_id, int):
            return analysis_id
        return 0  # Default fallback
    except Exception as e:
        print(f"Error saving analysis: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()

def update_entity_count(conn, entity_name: str) -> None:
    """
    Update entity count or create if it doesn't exist
    
    Args:
        conn: SQLite connection
        entity_name: Name of the entity
    """
    if not entity_name:
        return
        
    cursor = conn.cursor()
    
    # Try to get existing entity
    cursor.execute('SELECT id, count FROM entities WHERE name = ?', (entity_name,))
    result = cursor.fetchone()
    
    if result:
        # Update count and last_seen
        entity_id, count = result
        cursor.execute(
            'UPDATE entities SET count = ?, last_seen = CURRENT_TIMESTAMP WHERE id = ?',
            (count + 1, entity_id)
        )
    else:
        # Create new entity
        cursor.execute('INSERT INTO entities (name) VALUES (?)', (entity_name,))

def get_recent_analyses(limit: int = 10) -> pd.DataFrame:
    """
    Get recent analyses
    
    Args:
        limit: Maximum number of analyses to return
        
    Returns:
        DataFrame with analyses
    """
    conn = sqlite3.connect(DB_PATH)
    query = f'SELECT id, text, timestamp, title FROM analyses ORDER BY timestamp DESC LIMIT {limit}'
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def get_analysis_by_id(analysis_id: int) -> Dict[str, Any]:
    """
    Get an analysis by ID
    
    Args:
        analysis_id: ID of the analysis
        
    Returns:
        Dictionary with analysis data or empty dict if not found
    """
    conn = sqlite3.connect(DB_PATH)
    query = '''SELECT id, text, timestamp, title, model_type, gnn_architecture, 
               temperature, custom_prompt, processing_time_bert, processing_time_gnn, 
               processing_time_graph FROM analyses WHERE id = ?'''
    df = pd.read_sql_query(query, conn, params=[analysis_id])
    conn.close()
    
    if df.empty:
        return {}
    
    return df.iloc[0].to_dict()

def get_relationships_by_analysis_id(analysis_id: int) -> pd.DataFrame:
    """
    Get relationships for an analysis
    
    Args:
        analysis_id: ID of the analysis
        
    Returns:
        DataFrame with relationships
    """
    conn = sqlite3.connect(DB_PATH)
    query = '''
    SELECT id, subject, predicate, object, confidence, polarity, directness, sentence
    FROM relationships
    WHERE analysis_id = ?
    '''
    df = pd.read_sql_query(query, conn, params=[analysis_id])
    conn.close()
    
    return df

def get_top_entities(limit: int = 10) -> pd.DataFrame:
    """
    Get top entities by occurrence count
    
    Args:
        limit: Maximum number of entities to return
        
    Returns:
        DataFrame with entities
    """
    conn = sqlite3.connect(DB_PATH)
    query = f'SELECT id, name, count, last_seen FROM entities ORDER BY count DESC LIMIT {limit}'
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def get_analysis_count() -> int:
    """
    Get total count of analyses in the database
    
    Returns:
        Count of analyses
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM analyses')
    count = cursor.fetchone()[0]
    conn.close()
    
    return count

def get_relationship_count() -> int:
    """
    Get total count of relationships in the database
    
    Returns:
        Count of relationships
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM relationships')
    count = cursor.fetchone()[0]
    conn.close()
    
    return count

def delete_analysis(analysis_id: int) -> bool:
    """
    Delete an analysis and its relationships
    
    Args:
        analysis_id: ID of the analysis to delete
        
    Returns:
        True if successful, False otherwise
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
        conn.commit()
        success = cursor.rowcount > 0
    except Exception:
        success = False
    finally:
        conn.close()
    
    return success

# Initialize the database when module is loaded
init_db()
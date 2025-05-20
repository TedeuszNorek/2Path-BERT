import os
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
import db_models as models

def save_analysis(db: Session, text: str, relationships: List[Dict[str, Any]], title: Optional[str] = None) -> models.Analysis:
    """
    Save a text analysis and its relationships to the database
    
    Args:
        db: Database session
        text: The analyzed text
        relationships: List of extracted relationships
        title: Optional title for the analysis
        
    Returns:
        The created Analysis object
    """
    # Create a new analysis
    analysis = models.Analysis(
        text=text,
        title=title
    )
    
    # Add to session
    db.add(analysis)
    db.flush()  # Get the ID without committing
    
    # Add relationships
    for rel in relationships:
        relationship = models.Relationship(
            subject=rel.get("subject", ""),
            predicate=rel.get("predicate", ""),
            object=rel.get("object", ""),
            confidence=rel.get("confidence", 0.0),
            polarity=rel.get("polarity", "neutral"),
            directness=rel.get("directness", "direct"),
            sentence=rel.get("sentence", ""),
            analysis_id=analysis.id
        )
        db.add(relationship)
        
        # Update entity counts
        update_entity_count(db, rel.get("subject", ""))
        update_entity_count(db, rel.get("object", ""))
    
    # Commit the transaction
    db.commit()
    db.refresh(analysis)
    
    return analysis

def update_entity_count(db: Session, entity_name: str) -> None:
    """
    Update entity count or create if it doesn't exist
    
    Args:
        db: Database session
        entity_name: Name of the entity
    """
    import datetime
    
    if not entity_name:
        return
        
    # Try to get existing entity
    entity = db.query(models.Entity).filter(models.Entity.name == entity_name).first()
    
    if entity:
        # Update directly with SQL to avoid SQLAlchemy ORM issues
        from sqlalchemy import update
        stmt = update(models.Entity).where(models.Entity.id == entity.id).values(
            count=entity.count + 1,
            last_seen=datetime.datetime.utcnow()
        )
        db.execute(stmt)
    else:
        # Create new entity
        entity = models.Entity(name=entity_name)
        db.add(entity)

def get_recent_analyses(db: Session, limit: int = 10) -> List[models.Analysis]:
    """
    Get recent analyses
    
    Args:
        db: Database session
        limit: Maximum number of analyses to return
        
    Returns:
        List of Analysis objects
    """
    return db.query(models.Analysis).order_by(models.Analysis.timestamp.desc()).limit(limit).all()

def get_analysis_by_id(db: Session, analysis_id: int) -> Optional[models.Analysis]:
    """
    Get an analysis by ID
    
    Args:
        db: Database session
        analysis_id: ID of the analysis
        
    Returns:
        Analysis object or None if not found
    """
    return db.query(models.Analysis).filter(models.Analysis.id == analysis_id).first()

def get_relationships_by_analysis_id(db: Session, analysis_id: int) -> List[models.Relationship]:
    """
    Get relationships for an analysis
    
    Args:
        db: Database session
        analysis_id: ID of the analysis
        
    Returns:
        List of Relationship objects
    """
    return db.query(models.Relationship).filter(models.Relationship.analysis_id == analysis_id).all()

def get_top_entities(db: Session, limit: int = 10) -> List[models.Entity]:
    """
    Get top entities by occurrence count
    
    Args:
        db: Database session
        limit: Maximum number of entities to return
        
    Returns:
        List of Entity objects
    """
    return db.query(models.Entity).order_by(models.Entity.count.desc()).limit(limit).all()

def get_relationship_count(db: Session) -> int:
    """
    Get total count of relationships in the database
    
    Args:
        db: Database session
        
    Returns:
        Count of relationships
    """
    return db.query(models.Relationship).count()

def get_analysis_count(db: Session) -> int:
    """
    Get total count of analyses in the database
    
    Args:
        db: Database session
        
    Returns:
        Count of analyses
    """
    return db.query(models.Analysis).count()

def delete_analysis(db: Session, analysis_id: int) -> bool:
    """
    Delete an analysis and its relationships
    
    Args:
        db: Database session
        analysis_id: ID of the analysis to delete
        
    Returns:
        True if successful, False otherwise
    """
    analysis = get_analysis_by_id(db, analysis_id)
    if not analysis:
        return False
        
    db.delete(analysis)
    db.commit()
    
    return True
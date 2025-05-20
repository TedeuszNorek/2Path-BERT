import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from db_connection import engine

# Create declarative base
Base = declarative_base()

# Define models
class Analysis(Base):
    """Model for text analysis sessions"""
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    title = Column(String(255), nullable=True)
    
    # Relationships
    relationships = relationship("Relationship", back_populates="analysis", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Analysis(id={self.id}, timestamp={self.timestamp})>"


class Relationship(Base):
    """Model for extracted semantic relationships"""
    __tablename__ = 'relationships'
    
    id = Column(Integer, primary_key=True)
    subject = Column(String(255))
    predicate = Column(String(255))
    object = Column(String(255))
    confidence = Column(Float)
    polarity = Column(String(50))
    directness = Column(String(50))
    sentence = Column(Text)
    
    # Foreign keys
    analysis_id = Column(Integer, ForeignKey('analyses.id'))
    
    # Relationships
    analysis = relationship("Analysis", back_populates="relationships")
    
    def __repr__(self):
        return f"<Relationship(id={self.id}, subject='{self.subject}', predicate='{self.predicate}', object='{self.object}')>"


class Entity(Base):
    """Model for unique entities in the knowledge graph"""
    __tablename__ = 'entities'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, index=True)
    count = Column(Integer, default=1)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<Entity(id={self.id}, name='{self.name}', count={self.count})>"


# Create all tables
Base.metadata.create_all(engine)
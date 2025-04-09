"""
Database module for Brahma LLM Platform

This module provides database connection and models using SQLAlchemy for
storing user data, API keys, model metadata, usage statistics, and preferences.
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

# Get database URL from environment or use SQLite as default
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./brahma.db")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class DBUser(Base):
    """User model in the database."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=True)
    bio = Column(Text, nullable=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    api_keys = relationship("DBApiKey", back_populates="user", cascade="all, delete-orphan")
    usage_records = relationship("DBUsageRecord", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("DBUserPreferences", back_populates="user", uselist=False, cascade="all, delete-orphan")

class DBApiKey(Base):
    """API Key model in the database."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    key = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("DBUser", back_populates="api_keys")

class DBModel(Base):
    """Model metadata in the database."""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    path = Column(String)
    size = Column(Integer)  # Size in bytes
    parameters = Column(JSON)  # Model parameters
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_public = Column(Boolean, default=True)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    creator = relationship("DBUser")
    usage_records = relationship("DBUsageRecord", back_populates="model", cascade="all, delete-orphan")

class DBDataset(Base):
    """Dataset metadata in the database."""
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    path = Column(String)
    size = Column(Integer)  # Size in bytes
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    creator = relationship("DBUser")

class DBTrainingJob(Base):
    """Training job metadata in the database."""
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    status = Column(String)  # e.g. "queued", "running", "completed", "failed"
    config = Column(JSON)  # Training configuration
    metrics = Column(JSON, nullable=True)  # Training metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("DBUser")
    model = relationship("DBModel")
    dataset = relationship("DBDataset")

class DBUsageRecord(Base):
    """Usage record in the database."""
    __tablename__ = "usage_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_id = Column(Integer, ForeignKey("models.id"))
    endpoint = Column(String)  # e.g. "generate", "chat"
    tokens_in = Column(Integer, default=0)  # Input tokens
    tokens_out = Column(Integer, default=0)  # Output tokens
    processing_time = Column(Float)  # Processing time in seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("DBUser", back_populates="usage_records")
    model = relationship("DBModel", back_populates="usage_records")

class DBUserPreferences(Base):
    """User preferences in the database."""
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    default_model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    default_temperature = Column(Float, default=0.7)
    default_max_tokens = Column(Integer, default=100)
    stream_responses = Column(Boolean, default=True)
    theme = Column(String, default="system")  # e.g. "light", "dark", "system"
    custom_preferences = Column(JSON, default=lambda: {})
    
    # Relationships
    user = relationship("DBUser", back_populates="preferences")
    default_model = relationship("DBModel")

# Database functions
def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize the database with tables."""
    Base.metadata.create_all(bind=engine)

def create_user(db: Session, username: str, email: str, hashed_password: str, 
               full_name: Optional[str] = None, is_admin: bool = False) -> DBUser:
    """Create a new user in the database."""
    db_user = DBUser(
        username=username,
        email=email,
        hashed_password=hashed_password,
        full_name=full_name,
        is_admin=is_admin
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create default preferences
    create_user_preferences(db, db_user.id)
    
    return db_user

def create_user_preferences(db: Session, user_id: int) -> DBUserPreferences:
    """Create default preferences for a user."""
    db_preferences = DBUserPreferences(
        user_id=user_id,
        default_temperature=0.7,
        default_max_tokens=100,
        stream_responses=True,
        theme="system"
    )
    db.add(db_preferences)
    db.commit()
    db.refresh(db_preferences)
    return db_preferences

def get_user_by_username(db: Session, username: str) -> Optional[DBUser]:
    """Get a user by username."""
    return db.query(DBUser).filter(DBUser.username == username).first()

def get_user_by_email(db: Session, email: str) -> Optional[DBUser]:
    """Get a user by email."""
    return db.query(DBUser).filter(DBUser.email == email).first()

def update_user_last_login(db: Session, user_id: int) -> DBUser:
    """Update a user's last login time."""
    db_user = db.query(DBUser).filter(DBUser.id == user_id).first()
    if db_user:
        db_user.last_login = datetime.utcnow()
        db.commit()
        db.refresh(db_user)
    return db_user

def create_api_key(db: Session, user_id: int, name: str, key: str) -> DBApiKey:
    """Create a new API key for a user."""
    db_api_key = DBApiKey(
        user_id=user_id,
        name=name,
        key=key
    )
    db.add(db_api_key)
    db.commit()
    db.refresh(db_api_key)
    return db_api_key

def get_api_keys_for_user(db: Session, user_id: int) -> List[DBApiKey]:
    """Get all API keys for a user."""
    return db.query(DBApiKey).filter(DBApiKey.user_id == user_id).all()

def get_api_key_by_key(db: Session, key: str) -> Optional[DBApiKey]:
    """Get an API key by its key value."""
    return db.query(DBApiKey).filter(DBApiKey.key == key).first()

def delete_api_key(db: Session, key_id: int, user_id: int) -> bool:
    """Delete an API key."""
    db_api_key = db.query(DBApiKey).filter(
        DBApiKey.id == key_id, 
        DBApiKey.user_id == user_id
    ).first()
    
    if db_api_key:
        db.delete(db_api_key)
        db.commit()
        return True
    return False

def record_usage(db: Session, user_id: int, model_id: int, endpoint: str, 
                tokens_in: int, tokens_out: int, processing_time: float) -> DBUsageRecord:
    """Record usage for billing and analytics."""
    db_usage = DBUsageRecord(
        user_id=user_id,
        model_id=model_id,
        endpoint=endpoint,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        processing_time=processing_time
    )
    db.add(db_usage)
    db.commit()
    db.refresh(db_usage)
    return db_usage

def get_user_usage_stats(db: Session, user_id: int, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
    """Get usage statistics for a user."""
    query = db.query(DBUsageRecord).filter(DBUsageRecord.user_id == user_id)
    
    if start_date:
        query = query.filter(DBUsageRecord.created_at >= start_date)
    if end_date:
        query = query.filter(DBUsageRecord.created_at <= end_date)
    
    records = query.all()
    
    total_tokens_in = sum(record.tokens_in for record in records)
    total_tokens_out = sum(record.tokens_out for record in records)
    total_processing_time = sum(record.processing_time for record in records)
    
    # Usage by model
    model_usage = {}
    for record in records:
        model_id = record.model_id
        if model_id not in model_usage:
            model_usage[model_id] = {
                "tokens_in": 0,
                "tokens_out": 0,
                "processing_time": 0,
                "calls": 0
            }
        
        model_usage[model_id]["tokens_in"] += record.tokens_in
        model_usage[model_id]["tokens_out"] += record.tokens_out
        model_usage[model_id]["processing_time"] += record.processing_time
        model_usage[model_id]["calls"] += 1
    
    # Usage by day
    daily_usage = {}
    for record in records:
        day = record.created_at.strftime('%Y-%m-%d')
        if day not in daily_usage:
            daily_usage[day] = {
                "tokens_in": 0,
                "tokens_out": 0,
                "calls": 0
            }
        
        daily_usage[day]["tokens_in"] += record.tokens_in
        daily_usage[day]["tokens_out"] += record.tokens_out
        daily_usage[day]["calls"] += 1
    
    return {
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_processing_time": total_processing_time,
        "total_calls": len(records),
        "model_usage": model_usage,
        "daily_usage": daily_usage
    }

def register_model(db: Session, name: str, path: str, size: int, 
                  parameters: Dict[str, Any], creator_id: Optional[int] = None) -> DBModel:
    """Register a model in the database."""
    db_model = DBModel(
        name=name,
        path=path,
        size=size,
        parameters=parameters,
        creator_id=creator_id
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def get_models(db: Session, skip: int = 0, limit: int = 100, 
              include_private: bool = False, user_id: Optional[int] = None) -> List[DBModel]:
    """Get models from the database."""
    query = db.query(DBModel)
    
    if not include_private:
        query = query.filter(DBModel.is_public == True)
    
    if user_id is not None:
        query = query.filter((DBModel.is_public == True) | (DBModel.creator_id == user_id))
    
    return query.offset(skip).limit(limit).all()

def register_dataset(db: Session, name: str, path: str, size: int, 
                    description: Optional[str] = None, creator_id: Optional[int] = None) -> DBDataset:
    """Register a dataset in the database."""
    db_dataset = DBDataset(
        name=name,
        path=path,
        size=size,
        description=description,
        creator_id=creator_id
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

def get_datasets(db: Session, skip: int = 0, limit: int = 100, 
                creator_id: Optional[int] = None) -> List[DBDataset]:
    """Get datasets from the database."""
    query = db.query(DBDataset)
    
    if creator_id is not None:
        query = query.filter(DBDataset.creator_id == creator_id)
    
    return query.offset(skip).limit(limit).all()

def register_training_job(db: Session, training_id: str, user_id: int, 
                         dataset_id: int, config: Dict[str, Any], 
                         model_id: Optional[int] = None) -> DBTrainingJob:
    """Register a training job in the database."""
    db_training = DBTrainingJob(
        training_id=training_id,
        user_id=user_id,
        model_id=model_id,
        dataset_id=dataset_id,
        status="queued",
        config=config
    )
    db.add(db_training)
    db.commit()
    db.refresh(db_training)
    return db_training

def update_training_job_status(db: Session, training_id: str, status: str, 
                              metrics: Optional[Dict[str, Any]] = None) -> Optional[DBTrainingJob]:
    """Update a training job's status."""
    db_training = db.query(DBTrainingJob).filter(DBTrainingJob.training_id == training_id).first()
    
    if db_training:
        db_training.status = status
        
        if status == "running" and db_training.started_at is None:
            db_training.started_at = datetime.utcnow()
        
        if status in ["completed", "failed"] and db_training.completed_at is None:
            db_training.completed_at = datetime.utcnow()
        
        if metrics is not None:
            db_training.metrics = metrics
        
        db.commit()
        db.refresh(db_training)
    
    return db_training

def get_training_jobs(db: Session, user_id: Optional[int] = None, 
                     status: Optional[str] = None, 
                     skip: int = 0, limit: int = 100) -> List[DBTrainingJob]:
    """Get training jobs from the database."""
    query = db.query(DBTrainingJob)
    
    if user_id is not None:
        query = query.filter(DBTrainingJob.user_id == user_id)
    
    if status is not None:
        query = query.filter(DBTrainingJob.status == status)
    
    return query.order_by(DBTrainingJob.created_at.desc()).offset(skip).limit(limit).all()

def get_user_preferences(db: Session, user_id: int) -> Optional[DBUserPreferences]:
    """Get a user's preferences."""
    return db.query(DBUserPreferences).filter(DBUserPreferences.user_id == user_id).first()

def update_user_preferences(db: Session, user_id: int, preferences: Dict[str, Any]) -> DBUserPreferences:
    """Update a user's preferences."""
    db_preferences = db.query(DBUserPreferences).filter(DBUserPreferences.user_id == user_id).first()
    
    if not db_preferences:
        # Create preferences if they don't exist
        db_preferences = create_user_preferences(db, user_id)
    
    # Update standard fields
    if "default_model_id" in preferences:
        db_preferences.default_model_id = preferences.get("default_model_id")
    
    if "default_temperature" in preferences:
        db_preferences.default_temperature = preferences.get("default_temperature")
    
    if "default_max_tokens" in preferences:
        db_preferences.default_max_tokens = preferences.get("default_max_tokens")
    
    if "stream_responses" in preferences:
        db_preferences.stream_responses = preferences.get("stream_responses")
    
    if "theme" in preferences:
        db_preferences.theme = preferences.get("theme")
    
    # Update any custom preferences
    custom_prefs = db_preferences.custom_preferences or {}
    
    for key, value in preferences.items():
        if key not in ["default_model_id", "default_temperature", "default_max_tokens", 
                      "stream_responses", "theme"]:
            custom_prefs[key] = value
    
    db_preferences.custom_preferences = custom_prefs
    
    db.commit()
    db.refresh(db_preferences)
    return db_preferences

# Initialize the database if this module is run directly
if __name__ == "__main__":
    init_db()
    print("Database initialized.")

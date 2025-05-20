import os
import time
import logging
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, OperationalError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL', '')

# Create database engine with connection pooling and conservative settings
engine = create_engine(
    DATABASE_URL if DATABASE_URL else 'postgresql://user:password@localhost/db',
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    echo=False,         # Set to True for SQL query debugging
    connect_args={
        "connect_timeout": 15,  # Connection timeout in seconds
        "application_name": "semantic_analyzer"  # Identify application in database logs
    }
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_session():
    """Get a database session safely"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def execute_with_retry(operation, max_retries=3, retry_delay=1):
    """
    Execute a database operation with retry logic
    
    Args:
        operation: A function that performs a database operation
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Result of the operation
    
    Raises:
        Exception: If all retry attempts fail
    """
    retries = 0
    last_error = None
    
    while retries < max_retries:
        try:
            return operation()
        except OperationalError as e:
            last_error = e
            retries += 1
            if retries < max_retries:
                logger.warning(f"Database operation failed, retrying ({retries}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Database operation failed after {max_retries} attempts: {str(e)}")
                raise
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error: {str(e)}")
            raise
            
    # This should not be reached, but just in case
    if last_error:
        raise last_error
    raise Exception("Unknown error in database operation")

def get_engine():
    """Get the SQLAlchemy engine"""
    return engine
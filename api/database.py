import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
# Import your centralized logger utility
from .logger import setup_logger

# Initialize a dedicated logger for database operations
logger = setup_logger("api.database")

DATABASE_URL = "sqlite:///./algoforge.db"

logger.info(f"Initializing SQLAlchemy database engine linking to connection string: {DATABASE_URL}")

try:
    # connect_args={"check_same_thread": False} is required ONLY for SQLite
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )
    
    # Create the session maker template
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Establish the declarative base instance mapping
    Base = declarative_base()
    logger.info("SQLAlchemy database engine and core session structures bound successfully.")

except Exception as engine_err:
    # If the file path is restricted or engine configuration flags are broken, flag a critical alert
    logger.critical(f"Fatal breakdown configuring database engine: {str(engine_err)}", exc_info=True)
    raise


# SQLAlchemy Database Model
class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)


# Dependency to get db session per request
def get_db():
    """Generates a contextual transactional session pool for processing incoming API operations."""
    logger.debug("Opening fresh local database session frame context.")
    db = SessionLocal()
    try:
        yield db
    except Exception as db_runtime_err:
        # Capture operational failures (corrupted schemas, connection dropping) mid-request
        logger.error(f"Uncaught transactional runtime error detected inside request thread: {str(db_runtime_err)}", exc_info=True)
        raise
    finally:
        logger.debug("Closing local database connection frame to prevent resource socket leakage.")
        db.close()

# from sqlalchemy import create_engine, Column, Integer, String, Boolean
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# DATABASE_URL = "sqlite:///./algoforge.db"

# # connect_args={"check_same_thread": False} is required ONLY for SQLite
# engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # SQLAlchemy Database Model
# class UserDB(Base):
#     __tablename__ = "users"

#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String, unique=True, index=True, nullable=False)
#     hashed_password = Column(String, nullable=False)
#     is_active = Column(Boolean, default=True)

# # Dependency to get db session per request
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
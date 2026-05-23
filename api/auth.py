# Import date and time utilities
from datetime import datetime, timedelta
from typing import Optional
# jwt is used to encode and decode JWT tokens
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import os

from pydantic import BaseModel
from sqlalchemy.orm import Session
from .database import get_db, UserDB


# Loads variables from .env file
from dotenv import load_dotenv


# Load all environment variables from .env file
load_dotenv()

# Read secret values from environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")


# ---------------- SECURITY SETUP ---------------- #

# Configure password hashing using bcrypt
password_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# OAuth2 scheme configuration
# tokenUrl="login" means frontend should send login requests to /login
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# Pydantic schemas for request validation
class UserCreate(BaseModel):
    username: str
    password: str

# ---------------- HELPER FUNCTIONS ---------------- #

def verify_password(plain_password, hashed_password):
    """
    Verify whether the plain password entered by user
    matches the hashed password stored in database.
    """

    return password_context.verify(
        plain_password,
        hashed_password
    )


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
):
    """
    Create a JWT access token.

    Parameters:
    - data: payload data to encode
    - expires_delta: custom token expiry time

    Returns:
    - Encoded JWT token
    """

    # Create a copy of original payload
    to_encode = data.copy()

    # Set token expiration time
    # If no custom expiry is given, default to 15 minutes
    expires = datetime.utcnow() + (
        expires_delta or timedelta(minutes=15)
    )

    # Add expiration field to payload
    to_encode.update({"exp": expires})

    # Encode and return JWT token
    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


# ---------------- AUTHENTICATION ---------------- #

async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    """
    Get currently authenticated user from JWT token.

    Steps:
    1. Extract token from request
    2. Decode JWT token
    3. Extract username
    4. Fetch user from database
    5. Return user if valid
    """

    # Common exception for invalid authentication
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",

        # Required header for OAuth2 authentication
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        # Extract username from token payload
        # "sub" usually stores subject/user identifier
        username: str = payload.get("sub")

        # If username missing, token is invalid
        if username is None:
            raise credentials_exception

    # Catch invalid token errors
    except JWTError:
        raise credentials_exception

    # Query database instead of dict
    user = db.query(UserDB).filter(UserDB.username == username).first()

    # If user doesn't exist
    if user is None:
        raise credentials_exception

    # Return authenticated user
    return user



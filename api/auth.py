# Import date and time utilities
from datetime import datetime, timedelta

# Optional is used for optional function parameters
from typing import Optional

# JWTError handles JWT-related exceptions
# jwt is used to encode and decode JWT tokens
from jose import JWTError, jwt

# CryptContext is used for password hashing and verification
from passlib.context import CryptContext

# FastAPI utilities for dependency injection and error handling
from fastapi import Depends, HTTPException, status

# OAuth2 authentication helpers
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Used to access environment variables
import os

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


# ---------------- MOCK DATABASE ---------------- #

# Fake in-memory user database
# In production, this should be replaced with a real database
FAKE_USERS_DB = {
    "admin": {
        "username": "admin",

        # Store hashed password instead of plain text password
        # WARNING:
        # Hashing during startup is not ideal for production
        # Better approach:
        # Generate once and store permanently in DB
        "hashed_password": password_context.hash("algoforge2026"),

        # Whether account is active or disabled
        "disabled": False,
    }
}


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
    token: str = Depends(oauth2_scheme)
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

    # Fetch user from fake database
    user = FAKE_USERS_DB.get(username)

    # If user doesn't exist
    if user is None:
        raise credentials_exception

    # Return authenticated user
    return user



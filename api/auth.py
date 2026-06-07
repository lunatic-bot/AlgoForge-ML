# Import date and time utilities
from datetime import datetime, timedelta
from typing import Optional
# jwt is used to encode and decode JWT tokens
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import os

from pydantic import BaseModel
from sqlalchemy.orm import Session
from .database import get_db, UserDB

# Import your centralized logger utility
from .logger import setup_logger

# Initialize a dedicated logger for security and authentication routines
logger = setup_logger("api.auth")

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read secret values from environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")  # Safe fallback if env drops it
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")


# Configure password hashing using bcrypt
password_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# OAuth2 scheme configuration
# tokenUrl="login" means frontend should send login requests to /login
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# Pydantic schemas for request validation (Kept clean without internal logs)
class UserCreate(BaseModel):
    username: str
    password: str


def verify_password(plain_password, hashed_password):
    """
    Verify whether the plain password entered by user
    matches the hashed password stored in database.
    """
    try:
        # Never log raw or hashed passwords! Only log the occurrence of verification
        logger.debug("Executing bcrypt password hash verification check.")
        return password_context.verify(plain_password, hashed_password)
    except Exception as crypto_err:
        logger.error(f"Password hashing engine encountered a structural verification crash: {str(crypto_err)}", exc_info=True)
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token.

    Parameters:
    - data: payload data to encode
    - expires_delta: custom token expiry time

    Returns:
    - Encoded JWT token
    """
    subject = data.get("sub", "unknown")
    logger.info(f"Initiating JWT access token signing passport for subject user: '{subject}'")

    try:
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
        encoded_jwt = jwt.encode(
            to_encode,
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        logger.info(f"JWT access token generated and signed successfully for user: '{subject}'. Expiry timestamp set: {expires}")
        return encoded_jwt

    except TypeError as type_err:
        logger.critical(f"Token generation failed. Invalid data structure passed to payload: {str(type_err)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal token signing profile configuration mismatch.")
    except Exception as token_err:
        logger.critical(f"Cryptographic subsystem signature failure during JWT encoding: {str(token_err)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server security framework failed to register session.")




async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    """
    Get currently authenticated user from JWT token.

    Steps:
    - Extract token from request
    - Decode JWT token
    - Extract username
    - Fetch user from database
    - Return user if valid
    """
    logger.debug("Intercepted incoming route request. Attempting OAuth2 token extraction parsing.")

    # Common exception for invalid authentication
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        # Extract username from token payload ("sub")
        username: str = payload.get("sub")

        if username is None:
            logger.warning("JWT Decode Intercept: Token payload is structurally valid but missing mandatory 'sub' field.")
            raise credentials_exception
            
        logger.debug(f"JWT decrypted. Identity claimed signature belongs to: '{username}'. Verifying status in database.")

    except JWTError as jwt_decode_error:
        # Handles signature expiration, mismatched algorithms, or tampered keys
        logger.warning(f"Authentication rejected: JWT validation decode checkpoint failed. Reason: {str(jwt_decode_error)}")
        raise credentials_exception

    # Query database instead of dict to check identity status
    try:
        user = db.query(UserDB).filter(UserDB.username == username).first()
    except Exception as db_query_err:
        logger.critical(f"Database tracking interface offline or broken during credential lookup for user '{username}': {str(db_query_err)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal security verification engine database timeout.")

    # If user doesn't exist in system records
    if user is None:
        logger.warning(f"Authentication rejected: Token claims valid signature for user '{username}', but no such user exists in database records.")
        raise credentials_exception

    # Check if user account is deactivated
    if hasattr(user, 'is_active') and not user.is_active:
        logger.warning(f"Authentication rejected: User '{username}' is successfully authenticated but their active status profile is disabled.")
        raise HTTPException(status_code=403, detail="Inactive account profile. Access denied.")

    logger.debug(f"Authentication successful. Routing request context for user: '{username}'")
    return user




# # Import date and time utilities
# from datetime import datetime, timedelta
# from typing import Optional
# # jwt is used to encode and decode JWT tokens
# from jose import JWTError, jwt
# from passlib.context import CryptContext
# from fastapi import Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# import os

# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from .database import get_db, UserDB


# # Loads variables from .env file
# from dotenv import load_dotenv


# # Load all environment variables from .env file
# load_dotenv()

# # Read secret values from environment variables
# SECRET_KEY = os.getenv("SECRET_KEY")
# ALGORITHM = os.getenv("ALGORITHM")
# ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")


# # ---------------- SECURITY SETUP ---------------- #

# # Configure password hashing using bcrypt
# password_context = CryptContext(
#     schemes=["bcrypt"],
#     deprecated="auto"
# )

# # OAuth2 scheme configuration
# # tokenUrl="login" means frontend should send login requests to /login
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# # Pydantic schemas for request validation
# class UserCreate(BaseModel):
#     username: str
#     password: str

# # ---------------- HELPER FUNCTIONS ---------------- #

# def verify_password(plain_password, hashed_password):
#     """
#     Verify whether the plain password entered by user
#     matches the hashed password stored in database.
#     """

#     return password_context.verify(
#         plain_password,
#         hashed_password
#     )


# def create_access_token(
#     data: dict,
#     expires_delta: Optional[timedelta] = None
# ):
#     """
#     Create a JWT access token.

#     Parameters:
#     - data: payload data to encode
#     - expires_delta: custom token expiry time

#     Returns:
#     - Encoded JWT token
#     """

#     # Create a copy of original payload
#     to_encode = data.copy()

#     # Set token expiration time
#     # If no custom expiry is given, default to 15 minutes
#     expires = datetime.utcnow() + (
#         expires_delta or timedelta(minutes=15)
#     )

#     # Add expiration field to payload
#     to_encode.update({"exp": expires})

#     # Encode and return JWT token
#     return jwt.encode(
#         to_encode,
#         SECRET_KEY,
#         algorithm=ALGORITHM
#     )


# # ---------------- AUTHENTICATION ---------------- #

# async def get_current_user(
#     token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
# ):
#     """
#     Get currently authenticated user from JWT token.

#     Steps:
#     1. Extract token from request
#     2. Decode JWT token
#     3. Extract username
#     4. Fetch user from database
#     5. Return user if valid
#     """

#     # Common exception for invalid authentication
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",

#         # Required header for OAuth2 authentication
#         headers={"WWW-Authenticate": "Bearer"},
#     )

#     try:
#         # Decode JWT token
#         payload = jwt.decode(
#             token,
#             SECRET_KEY,
#             algorithms=[ALGORITHM]
#         )

#         # Extract username from token payload
#         # "sub" usually stores subject/user identifier
#         username: str = payload.get("sub")

#         # If username missing, token is invalid
#         if username is None:
#             raise credentials_exception

#     # Catch invalid token errors
#     except JWTError:
#         raise credentials_exception

#     # Query database instead of dict
#     user = db.query(UserDB).filter(UserDB.username == username).first()

#     # If user doesn't exist
#     if user is None:
#         raise credentials_exception

#     # Return authenticated user
#     return user



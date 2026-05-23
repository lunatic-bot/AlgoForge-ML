from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from datetime import datetime, timedelta
from fastapi.security import  OAuth2PasswordRequestForm
from .auth import UserCreate, create_access_token, verify_password, ACCESS_TOKEN_EXPIRE_MINUTES, password_context
from sqlalchemy.orm import Session
from .database import engine, Base, UserDB, get_db

# Initialize the main FastAPI application
app = FastAPI(
    title="AlgoForge API", 
    description="End-to-end Machine Learning Demonstrator",
    version="1.0.0"
)

# --- MIDDLEWARE ---
# This prevents CORS errors when Streamlit (running on port 8501) 
# tries to send requests to FastAPI (running on port 8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Plug the endpoints from routes.py into the main app
app.include_router(router)

# Create the SQLite tables on startup automatically
Base.metadata.create_all(bind=engine)

# Optional root endpoint just to check if the server is alive
@app.get("/")
def read_root():
    return {"message": "Welcome to the AlgoForge API Engine. Go to /docs to explore the endpoints."}


# 1. Registration Endpoint
@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if username exists
    db_user = db.query(UserDB).filter(UserDB.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Hash password and save
    hashed_pw = password_context.hash(user.password)
    new_user = UserDB(username=user.username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully"}

# 2. Updated Login Endpoint checking SQLite
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=int(ACCESS_TOKEN_EXPIRE_MINUTES))
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
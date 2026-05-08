from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from datetime import datetime, timedelta
from fastapi.security import  OAuth2PasswordRequestForm
from .auth import create_access_token, verify_password, FAKE_USERS_DB, ACCESS_TOKEN_EXPIRE_MINUTES


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

# --- INCLUDE ROUTER ---
# Plug the endpoints from routes.py into the main app
app.include_router(router)

# Optional root endpoint just to check if the server is alive
@app.get("/")
def read_root():
    return {"message": "Welcome to the AlgoForge API Engine. Go to /docs to explore the endpoints."}

@app.post("login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)

    return {"access_token": access_token, "token_type": "bearer"}
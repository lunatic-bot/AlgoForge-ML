from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

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
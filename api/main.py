# The FastAPI app instance
from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="AlgoForge-ML API",
    description="Machine Learning API for training and predictions",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "Welcome to AlgoForge-ML API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
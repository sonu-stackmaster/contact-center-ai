from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import router
from .utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Contact Center AI API",
    description="AI-powered customer support analytics and assistance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Contact Center AI API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Contact Center AI API shutting down...")

@app.get("/")
async def root():
    return {
        "message": "Welcome to Contact Center AI API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
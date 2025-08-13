"""
Posture Measurement API - Main Application Entry Point

This is the main FastAPI application that provides posture analysis services
from front and side view images using MediaPipe pose detection.
"""

import uvicorn
import nest_asyncio
from fastapi import FastAPI

from api.routes import router
from app_config.settings import APIConfig


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Posture Measurement API",
        version=APIConfig.VERSION,
        description="AI-powered posture analysis service using MediaPipe pose detection",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Include API routes
    app.include_router(router)

    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """Startup event."""
    print("Starting Posture Measurement API...")
    print("API is ready to serve requests!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down Posture Measurement API...")


def main():
    """Run the API server."""
    # Apply nest_asyncio for compatibility
    nest_asyncio.apply()

    # Run the server
    uvicorn.run(
        app,
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        log_level="info",
        reload=False,  # Set to True for development
    )


if __name__ == "__main__":
    main()

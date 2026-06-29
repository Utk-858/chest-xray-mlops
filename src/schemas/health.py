from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """
    Response schema for the service health check endpoint.
    """
    status: str = Field(..., description="Liveness and readiness status of the serving service")

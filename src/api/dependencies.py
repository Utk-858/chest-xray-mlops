from fastapi import Request

from src.model_manager.predictor import ModelPredictor

def get_predictor(request: Request) -> ModelPredictor:
    """
    Dependency provider to retrieve the instantiated ModelPredictor from request state.
    Allows clean dependency injection inside endpoint routes.
    """
    return request.app.state.predictor

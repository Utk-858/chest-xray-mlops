from pydantic import BaseModel, Field

class PredictionResult(BaseModel):
    """
    Schema representing the structured result of an ML chest X-ray inference request.
    """
    predicted_class: str = Field(..., description="The predicted pathology class label (e.g. Normal, Opacity)")
    confidence: float = Field(..., description="The model confidence probability associated with the prediction")
    probabilities: dict[str, float] = Field(..., description="Class label mappings to their respective float probability scores")

class BatchPredictionResult(BaseModel):
    """
    Schema representing the results of a batch inference serving request.
    Maps filenames to either their valid PredictionResult, or a string detailing the failure reason.
    """
    results: dict[str, PredictionResult | str] = Field(
        ...,
        description="Map of filename keys to successful PredictionResult payloads or error description strings"
    )

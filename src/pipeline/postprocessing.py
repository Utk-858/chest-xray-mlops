import torch

from src.schemas.predict import PredictionResult

# Binary class mapping for chest X-ray classification
CLASS_LABELS = ["Normal", "Opacity"]

def postprocess_outputs(
    logits: torch.Tensor,
    confidence_threshold: float = 0.5
) -> PredictionResult:
    """
    Applies Softmax to convert raw logits to probabilities and parses them into a PredictionResult.
    """
    # Convert output logits to probability distribution: shape [B, Classes] -> shape [B, 2]
    probabilities = torch.softmax(logits, dim=1)[0].tolist()

    max_prob = max(probabilities)
    max_idx = probabilities.index(max_prob)
    predicted_class = CLASS_LABELS[max_idx]

    # Map probability scores to corresponding label names
    mapped_probs = {CLASS_LABELS[i]: probabilities[i] for i in range(len(CLASS_LABELS))}

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=max_prob,
        probabilities=mapped_probs
    )

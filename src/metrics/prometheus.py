from prometheus_client import Counter, Histogram

# Initialize Prometheus Metrics with labels for model tracking
PREDICTION_REQUESTS = Counter(
    "chest_xray_prediction_requests_total",
    "Total prediction requests received",
    ["model_name", "model_version"]
)

PREDICTION_SUCCESS = Counter(
    "chest_xray_prediction_success_total",
    "Successful prediction count",
    ["model_name", "model_version"]
)

PREDICTION_FAILURES = Counter(
    "chest_xray_prediction_failures_total",
    "Failed prediction count",
    ["model_name", "model_version"]
)

REQUEST_LATENCY = Histogram(
    "chest_xray_request_latency_seconds",
    "Request latency in seconds",
    ["model_name", "model_version"]
)

MODEL_INFERENCE_LATENCY = Histogram(
    "chest_xray_model_inference_seconds",
    "Model inference latency in seconds",
    ["model_name", "model_version"]
)

PREPROCESSING_LATENCY = Histogram(
    "chest_xray_preprocessing_seconds",
    "Preprocessing latency in seconds",
    ["model_name", "model_version"]
)

POSTPROCESSING_LATENCY = Histogram(
    "chest_xray_postprocessing_seconds",
    "Postprocessing latency in seconds",
    ["model_name", "model_version"]
)

# Reusable metrics helper functions
def record_prediction_request(model_name: str, model_version: str) -> None:
    PREDICTION_REQUESTS.labels(model_name=model_name, model_version=model_version).inc()

def record_prediction_success(model_name: str, model_version: str) -> None:
    PREDICTION_SUCCESS.labels(model_name=model_name, model_version=model_version).inc()

def record_prediction_failure(model_name: str, model_version: str) -> None:
    PREDICTION_FAILURES.labels(model_name=model_name, model_version=model_version).inc()

def observe_request_latency(model_name: str, model_version: str, latency: float) -> None:
    REQUEST_LATENCY.labels(model_name=model_name, model_version=model_version).observe(latency)

def observe_model_inference_latency(model_name: str, model_version: str, latency: float) -> None:
    MODEL_INFERENCE_LATENCY.labels(model_name=model_name, model_version=model_version).observe(latency)

def observe_preprocessing_latency(model_name: str, model_version: str, latency: float) -> None:
    PREPROCESSING_LATENCY.labels(model_name=model_name, model_version=model_version).observe(latency)

def observe_postprocessing_latency(model_name: str, model_version: str, latency: float) -> None:
    POSTPROCESSING_LATENCY.labels(model_name=model_name, model_version=model_version).observe(latency)

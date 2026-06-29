# Chest X-Ray MLOps Serving Platform - Development Documentation

This document tracks the incremental design choices, implementation steps, and architectural rationales as we build this production-ready ML serving service.

---

## Phase 1: Serving Infrastructure

### Step 0: Repository Layout and Directory Reorganization
*   **What was done**: 
    Created a modular repository layout segregating API endpoints, schemas, ML pipelines, model management, configurations, utilities, and testing modules. Added placeholder `__init__.py` modules to define package namespaces.
*   **Why we did it**:
    *   **Separation of Concerns**: Isolated web server constraints (FastAPI) from ML framework transformations (PyTorch) to ensure that if serving tooling changes in the future, core processing code remains unchanged.
    *   **Namespace Clarity**: Renamed the internal code package from `models/` to `model_manager/` to prevent directory name collision with the root-level `models/` directory where large binary `.pth` files are stored.
    *   **Pipeline Extensibility**: Renamed processing directories to `pipeline/` to fit future ML tasks (e.g., image quality checks, noise estimation, explainability, Grad-CAM).
    *   **Deferred Tooling (YAGNI)**: Intentionally removed Docker configurations and GitHub actions to prevent premature infrastructure debt, ensuring tooling is only added once a specific development or deployment problem arises.

---

### Step 1: Production-Ready Configuration System
*   **What was done**:
    *   Created [config/config.yaml](file:///Users/utkarshbansal/chest-xray-mlops/config/config.yaml) separating configurations into logical blocks (`app`, `models`, `active_model`, `inference`, `logging`).
    *   Implemented [src/core/config.py](file:///Users/utkarshbansal/chest-xray-mlops/src/core/config.py) using `pydantic-settings` to load configurations from YAML files and parse environment overrides dynamically.
    *   Implemented an `@lru_cache` retrieval function `get_settings()` to inject configuration settings project-wide efficiently.
    *   Added a `@model_validator` to verify model registry configuration at startup.
    *   Created a test suite in [tests/unit/test_config.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/unit/test_config.py) to assert environment overrides and field bounds constraints.
*   **Why we did it**:
    *   **Fail-Fast Startup Principle**: Validation logic runs during initialization rather than request execution. If a developer sets an unsupported active model key or wrong device name, the service raises a `ValidationError` and halts immediately on boot.
    *   **Configuration Flexibility**: The nested `models` dictionary maps model keys to specific attributes (e.g., `input_size`, `path`), making it trivial to configure and swap multiple models dynamically in the future without code refactoring.
    *   **Immutable Types**: Used native `tuple[int, int]` instead of lists for `input_size` variables to guarantee dimension configurations remain immutable.
    *   **Dynamic Overrides**: Configured nested delimiter mapping (`__`) to easily pass production overrides via environment variables (e.g., setting `INFERENCE__DEVICE=cuda`).

---

### Step 2: Production Logging System
*   **What was done**:
    *   Created [config/logging.yaml](file:///Users/utkarshbansal/chest-xray-mlops/config/logging.yaml) defining console-only loggers, standard handlers, stream outputs, and structured formatting metrics (timestamps, levels, file names, lines, and messages).
    *   Implemented [src/core/logging.py](file:///Users/utkarshbansal/chest-xray-mlops/src/core/logging.py) exposing initialization hooks (`setup_logging()`) and logger lookup functions (`get_logger(name)`).
    *   Created a logging test suite in [tests/unit/test_logging.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/unit/test_logging.py) mocking `dictConfig` and checking environment fallback structures.
*   **Why we did it**:
    *   **Robust Diagnostic Context**: Configured formatters to print the filename and line number (`%(filename)s:%(lineno)d`) where log commands execute, simplifying debugging in multithreaded serving applications.
    *   **Dynamic Configuration**: Interlocked `setup_logging` execution with `get_settings()` configuration log level properties (`settings.logging.level`), ensuring logging verbosity can be altered on environment boots.
    *   **Fault-Tolerant Fallback**: Included standard library fallback routines (`basicConfig()`) if custom configuration loaders fail, avoiding serving crashes due to logging parsing errors.

---

### Step 3: FastAPI Application Bootstrap
*   **What was done**:
    *   Created [src/main.py](file:///Users/utkarshbansal/chest-xray-mlops/src/main.py) to instantiate the FastAPI application using metadata resolved from Pydantic settings.
    *   Implemented application `lifespan` context manager hooks to bootstrap the logging system at server startup and log status changes on server shutdown.
    *   Created [src/api/router.py](file:///Users/utkarshbansal/chest-xray-mlops/src/api/router.py) defining the base API endpoint routing map.
    *   Added integration tests in [tests/integration/test_main.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/integration/test_main.py) to verify documentation path access (`/docs`, `/openapi.json`).
*   **Why we did it**:
    *   **Lifespan Event Standardization**: Utilized FastAPI's modern async `lifespan` context manager to control the application startup/shutdown loops, replacing deprecated `startup` and `shutdown` events.
    *   **Modular API Design**: Separated bootstrap code (`main.py`) from api endpoint mapping (`router.py`) to isolate core platform orchestration settings from router endpoint modifications.
    *   **Fail-Safe Serving Hooks**: Pre-configured automatic Swagger UI (`/docs`) and ReDoc (`/redoc`) documentation mounts based on active configuration properties to enable fast service discoverability in production.

---

### Step 4: Health Endpoint
*   **What was done**:
    *   Defined the Pydantic schema in [src/schemas/health.py](file:///Users/utkarshbansal/chest-xray-mlops/src/schemas/health.py) to represent the `/health` service response structure.
    *   Implemented `GET /health` in [src/api/v1/endpoints/health.py](file:///Users/utkarshbansal/chest-xray-mlops/src/api/v1/endpoints/health.py) to return a basic liveness status.
    *   Registered the endpoint under a versioned routing directory in [src/api/v1/router.py](file:///Users/utkarshbansal/chest-xray-mlops/src/api/v1/router.py) and integrated it with the base [api_router](file:///Users/utkarshbansal/chest-xray-mlops/src/api/router.py#L6).
    *   Created tests in [tests/integration/test_health.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/integration/test_health.py) to verify payload content and check logger operations.
*   **Why we did it**:
    *   **Pydantic Contract Validation**: Wrapping the health check output in a strict `HealthResponse` model ensures consistent, validated serialization that aligns with Swagger UI specifications.
    *   **Modular Versioning Layout**: Isolated `v1` routers from the base api router to simplify registration and management as the project scales.
    *   **Diagnostic Audit Logs**: Integrated health check invocations with the centralized logger (`logger.info`) to output standard diagnostic metrics, helping systems administrators debug load balancer heartbeats.

---

### Step 5: Model Manager
*   **What was done**:
    *   Implemented [src/model_manager/registry.py](file:///Users/utkarshbansal/chest-xray-mlops/src/model_manager/registry.py) to resolve model configuration parameters and filesystem locations.
    *   Created [src/model_manager/loader.py](file:///Users/utkarshbansal/chest-xray-mlops/src/model_manager/loader.py) containing PyTorch device resolution (`resolve_device`) and weight deserialization (`load_pytorch_model`).
    *   Created [src/model_manager/predictor.py](file:///Users/utkarshbansal/chest-xray-mlops/src/model_manager/predictor.py) as a wrapper class (`ModelPredictor`) executing model forwards under `torch.no_grad()`.
    *   Registered the active `ModelPredictor` in FastAPI's startup hook inside [src/main.py](file:///Users/utkarshbansal/chest-xray-mlops/src/main.py) and exposed a dependency resolver in [src/api/dependencies.py](file:///Users/utkarshbansal/chest-xray-mlops/src/api/dependencies.py).
    *   Added comprehensive unit tests in [tests/unit/test_model_manager.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/unit/test_model_manager.py).
*   **Why we did it**:
    *   **Encapsulation of Inference Lifecycle**: Separated model registry lookup, architecture loading, and runtime execution logic into isolated scripts to make model serving framework-agnostic.
    *   **In-Memory Lifecycle Caching**: Model weights are loaded once at startup and stored in `app.state` to avoid disk reading latency during request evaluations.
    *   **Device Autodetect Capability**: Engineered hardware detection mappings (`auto`) to dynamically target GPU structures (`cuda`/`mps`) or fail back to `cpu`.
    *   **Testing Resilience**: Configured weight loading routines to fail gracefully to random weights if `.pth` state dicts are absent. This allows local environments and automated testing pipelines to boot and verify code structures without massive artifact downloads.

---

### Step 6: Inference Service
*   **What was done**:
    *   Defined the Pydantic prediction response schema in [src/schemas/predict.py](file:///Users/utkarshbansal/chest-xray-mlops/src/schemas/predict.py).
    *   Implemented image transformations and normalizations in [src/pipeline/preprocessing.py](file:///Users/utkarshbansal/chest-xray-mlops/src/pipeline/preprocessing.py).
    *   Implemented output softmax parsing and mapped logits to labels in [src/pipeline/postprocessing.py](file:///Users/utkarshbansal/chest-xray-mlops/src/pipeline/postprocessing.py).
    *   Created the main `InferenceService` orchestrator in [src/services/inference.py](file:///Users/utkarshbansal/chest-xray-mlops/src/services/inference.py) to manage the execution order of these tasks.
    *   Created unit tests in [tests/unit/test_inference_service.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/unit/test_inference_service.py).
*   **Why we did it**:
    *   **Unified ML Pipelines**: Wrapped preprocessing transforms, model forward executions, and softmax logit mappings inside a single coordinator class (`InferenceService`), keeping API route files clean and framework-unaware.
    *   **Image Mode Standardizations**: Configured preprocessing functions to automatically convert single-channel grayscale chest X-rays to 3-channel RGB arrays, preventing channel dimension mismatch errors inside ResNet model weights.
    *   **Strict Output Contracts**: Ensured postprocessing converts model outputs to a validated `PredictionResult` structure, guaranteeing data serialization compliance.

---

### Step 7: Prediction Endpoint
*   **What was done**:
    *   Implemented `POST /predict` in [src/api/v1/endpoints/predict.py](file:///Users/utkarshbansal/chest-xray-mlops/src/api/v1/endpoints/predict.py) using FastAPI `UploadFile`.
    *   Registered the endpoint under tags `["predict"]` inside [src/api/v1/router.py](file:///Users/utkarshbansal/chest-xray-mlops/src/api/v1/router.py).
    *   Implemented image byte array validation by attempting PIL open triggers, returning `HTTP 400 Bad Request` for invalid files.
    *   Exposed integration test validations in [tests/integration/test_predict.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/integration/test_predict.py).
*   **Why we did it**:
    *   **Decoupled Controller Flow**: Maintained a thin API layer that delegates all ML logic to `InferenceService`, leaving the HTTP endpoint unaware of preprocessing, device logic, or postprocessing operations.
    *   **Proactive Input Validations**: Added image parsing gates (`Image.verify()`) to filter corrupt images or invalid text payloads at the controller layer.
    *   **Central Logging Traceability**: Logged critical stages of the request cycle (file verification warnings, internal exceptions, prediction outcomes) to assist production monitoring logs.

---

### Step 8: Pipeline Timing Instrumentation (Production Incident #1)
*   **What was done**:
    *   Created [src/utils/timing.py](file:///Users/utkarshbansal/chest-xray-mlops/src/utils/timing.py) implementing a reusable `time_block()` context manager measuring execution duration in milliseconds.
    *   Instrumented [src/services/inference.py](file:///Users/utkarshbansal/chest-xray-mlops/src/services/inference.py) wrapping preprocessing, model forwards, and postprocessing with timing blocks.
    *   Logged latency breakdown statistics (preprocessing, inference, postprocessing, total request time) to standard log output.
    *   Added timing tests in [tests/unit/test_timing.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/unit/test_timing.py) and added log format validations inside [tests/unit/test_inference_service.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/unit/test_inference_service.py).
*   **Why we did it**:
    *   **Production Traceability**: Added timing context checkpoints to easily isolate whether latency spikes originate from preprocessing transformations, PyTorch forward iterations, or softmax formatting structures.
    *   **Observability Foundation**: Formatted logs to print clear, grep-friendly latency metrics to stdout. This makes it trivial for log forwarders (e.g. FluentBit, Logstash) to parse these lines, or convert them later to Prometheus histograms without refactoring pipeline loops.

---

### Step 9: Context-Bound Prediction Logging (Production Incident #2)
*   **What was done**:
    *   Exposed `request_id_var` context variable and `RequestIdFilter` logging filter in [src/core/logging.py](file:///Users/utkarshbansal/chest-xray-mlops/src/core/logging.py).
    *   Configured the log formatter format to print request IDs automatically and registered the filter on the console stream handler inside [config/logging.yaml](file:///Users/utkarshbansal/chest-xray-mlops/config/logging.yaml).
    *   Implemented `add_request_id_header` FastAPI HTTP middleware in [src/main.py](file:///Users/utkarshbansal/chest-xray-mlops/src/main.py) to set context variables and append custom `X-Request-ID` headers to responses.
    *   Enriched latency and config metrics logs inside [src/services/inference.py](file:///Users/utkarshbansal/chest-xray-mlops/src/services/inference.py).
    *   Updated unit and integration tests in [tests/unit/test_logging.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/unit/test_logging.py), [tests/unit/test_inference_service.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/unit/test_inference_service.py), and [tests/integration/test_predict.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/integration/test_predict.py).
*   **Why we did it**:
    *   **Async-Safe Context Propagation**: Leveraged Python's `contextvars` to bound unique request identifiers thread-safely without having to refactor business interface signatures or method calls.
    *   **Zero-Overhead Logging Filters**: Integrated custom logging filters so that every log statement automatically prints the trace identifier, decoupling tracing context from specific print statements.
    *   **Telemetry Readiness**: Formatted performance metrics into a standardized string format (Model, Version, Device, Latency Breakdown) to ensure future compatibility with parsing engines or JSON converters.

---

### Step 10: Batch Inference & Docker Containerization (Production Incident #3)
*   **What was done**:
    *   Defined the Pydantic schema `BatchPredictionResult` in [src/schemas/predict.py](file:///Users/utkarshbansal/chest-xray-mlops/src/schemas/predict.py) to represent batch outcomes.
    *   Extended [src/services/inference.py](file:///Users/utkarshbansal/chest-xray-mlops/src/services/inference.py) implementing `predict_batch(images)`: preprocesses valid items, splits them into configured batch sizes, stacks tensors using `torch.stack`, runs batched forwards, and aggregates partial corruptions/exceptions.
    *   Implemented `POST /predict/batch` endpoint inside [src/api/v1/endpoints/predict.py](file:///Users/utkarshbansal/chest-xray-mlops/src/api/v1/endpoints/predict.py).
    *   Created the production [Dockerfile](file:///Users/utkarshbansal/chest-xray-mlops/Dockerfile) (built from `python:3.12-slim` using a non-root system user) and [.dockerignore](file:///Users/utkarshbansal/chest-xray-mlops/.dockerignore) (blocking local virtualenvs and weights binaries).
    *   Added integration tests in [tests/integration/test_batch_predict.py](file:///Users/utkarshbansal/chest-xray-mlops/tests/integration/test_batch_predict.py) to verify batch execution loops and partial corruptions outputting formats.
*   **Why we did it**:
    *   **Graceful Partial Fault Tolerance**: Isolated preprocessing and logit postprocessing checkpoints individually. This guarantees that one corrupted or invalid image upload does not crash or invalidate the entire batch result collection.
    *   **Extensible Batch Performance**: Leveraged `torch.stack` to construct unified 4D tensors for model forward passes. This enables hardware-optimized vectorized computation, laying the foundation for future GPU/MPS throughput enhancements.
    *   **Secure Containerization Boundaries**: Packaged the serving app using standard system user permissions (`appuser` / `appgroup`) and locked down copy layouts to prevent accidental inclusions of large weight binaries (`.pth` files) inside Docker layers.

# Chest X-Ray MLOps Serving Platform

This service provides production-ready chest X-ray classification API endpoints utilizing PyTorch models (e.g. ResNet50).

---

## Metrics and Observability

The serving platform is instrumented with Prometheus metrics for near real-time observability and service level objectives (SLO) tracking.

### Accessing Metrics

To retrieve current application telemetry, send a `GET` request to:
```http
GET /metrics
```

### Metrics Reference

The following metrics are exported:

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `chest_xray_prediction_requests_total` | Counter | Total prediction requests received | `model_name`, `model_version` |
| `chest_xray_prediction_success_total` | Counter | Successful prediction count | `model_name`, `model_version` |
| `chest_xray_prediction_failures_total` | Counter | Failed prediction count | `model_name`, `model_version` |
| `chest_xray_request_latency_seconds` | Histogram | Request latency in seconds | `model_name`, `model_version` |
| `chest_xray_model_inference_seconds` | Histogram | Model forward pass latency in seconds | `model_name`, `model_version` |
| `chest_xray_preprocessing_seconds` | Histogram | Input image preprocessing transformation latency in seconds | `model_name`, `model_version` |
| `chest_xray_postprocessing_seconds` | Histogram | Logic output softmax conversion latency in seconds | `model_name`, `model_version` |

### Configuration

Metrics tracking can be enabled or disabled via [config/config.yaml](file:///Users/utkarshbansal/chest-xray-mlops/config/config.yaml):
```yaml
metrics:
  enabled: true
```

Or overridden dynamically in environment variables:
```bash
METRICS__ENABLED=false
```

### Monitoring and Telemetry Stack

The monitoring infrastructure is integrated using Docker Compose.

#### Starting the Stack
To boot the FastAPI application, Prometheus, and Grafana containers simultaneously, run:
```bash
docker compose up -d
```

#### Service Access Endpoints

- **FastAPI API**: [http://localhost:8000](http://localhost:8000) (Swagger docs available at [/docs](http://localhost:8000/docs))
- **Prometheus UI**: [http://localhost:9090](http://localhost:9090) (Targets page at [/targets](http://localhost:9090/targets))
- **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000) (Login using credentials `admin` / `admin`)

#### Provisioning and Dashboards
- **Prometheus**: Automatically configured to scrape target `chest-xray-api:8000` at a `5s` interval.
- **Grafana Datasource**: Automatically provisioned to link to the Prometheus container on boot.
- **Grafana Dashboard**: Automatically configured as the home dashboard showing panels for:
  - Total prediction requests, success count, and failure count.
  - API request latency (average and 95th percentile).
  - ML Pipeline latency breakdown (preprocessing, inference, postprocessing 95th percentiles).


---

## Production Hardening TODO

- Enable strict model loading in production mode.
- Application should fail startup if:
  - model weights are missing
  - checkpoint architecture mismatches
  - model checksum validation fails
- Development mode may allow fallback behavior for debugging.
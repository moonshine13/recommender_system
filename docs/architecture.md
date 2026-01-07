## System Architecture

The recommender system is designed with a modular layered architecture supporting both CLI and FastAPI interfaces.

---

### Data Flow
**CLI**

User executes CLI command

Arguments parsed and validated

Application layer calls domain logic

Results printed to stdout

**API**

Client sends HTTP request

Endpoint validates input

Application layer orchestrates domain logic

JSON response returned

---

### Model Lifecycle

**Training**

Preprocess data, normalize ratings and timestamps

Leave-last-out split for train/test

Train model (TimeSVD++ for model-based)

Serialize model to file

**Inference**

Load pretrained model

Generate predictions per request

Optionally exclude already-rated items

---

### Deployment

CLI Docker image: lightweight, volume-mounted data, one-shot execution
API Docker image: FastAPI + Uvicorn, exposes port 8000, volume-mounted data

---

### Design Principles

Modularity

Explicit dependencies

Testability and reproducibility

Framework isolation
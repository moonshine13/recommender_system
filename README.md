# Recommender System

A Python-based recommendation system supporting **user-based**, **time-aware user-based**, **model-based (TimeSVD++)**, and **top-N popularity-based** recommendations.  
The system can be used via **CLI** or exposed as a **FastAPI service**.

## Documentation
  Data preparation.
  [Details](docs/data_preparation.md)

  Architecture.
  [Details](docs/architecture.md)

  Recommendation strategies
- **User-Based Collaborative Filtering**  

  Memory-based method using cosine similarity between users.
  [Details](docs/strategies/user_based.md)

- **Time-Aware User-Based Filtering**  

  Extends user-based CF by applying temporal decay to older interactions.  
  [Details](docs/strategies/user_based_time_aware.md)

- **Model-Based (TimeSVD++)**  

  Latent factor model incorporating implicit feedback and temporal dynamics.  
  [Details](docs/strategies/model_based.md)

- **Top-N Products**  

  Popularity-based baseline using recent average ratings.  
  [Details](docs/strategies/top_n_products.md)

---

## Install

### Clone the repo
```bash
git clone https://github.com/moonshine13/recommender_system.git
cd recommender_system
```

### Create virtual environment
```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements-dev.txt
# or using pyproject.toml
pip install .
```

---

## CLI Usage

### Top-N products
```bash
python -m cli.top_n_products_cli --path data/ratings.csv --days 365 --min_ratings 10 --n 5
```

### User-based recommendations
```bash
python -m cli.user_rec_cli --path data/ratings.csv --user_id 671 --n 5 --k 5 --rec_type user_based_with_time
```

### Model-based recommendations
```bash
python -m cli.model_rec_cli --user_id 671 --exclude_rated True --n 5
```

---

## Build and Run CLI in Docker

### Build image
```bash
docker build -f Dockerfile.cli -t recommendation-cli .
```

### Run CLI
```bash
docker run --rm recommendation-cli cli.model_rec_cli --user_id 671 --exclude_rated True --n 5
docker run --rm -v C:/repos/recommender_system/data:/data recommendation-cli  cli.user_rec_cli --path /data/ratings.csv --user_id 671 --n 5 --k 5 --rec_type user_based_with_time
docker run --rm -v C:/repos/recommender_system/data:/data recommendation-cli  cli.top_n_products_cli --path /data/ratings.csv --days 365 --min_ratings 10 --n 5
```

---

## Build and Run API in Docker

### Build image
```bash
docker build -t recommendation-api .
```

### Run API
```bash
docker run -it --rm -p 8000:8000 -v C:/repos/recommender_system/data:/data recommendation-api
```

Swagger UI: [http://127.0.0.1:8000/docs]

## Testing

```bash
pytest tests/
```

## Development Notes

Configuration is centralized in pyproject.toml

Code style enforced with Black and isort

Linting with pylint

Security scanning with bandit

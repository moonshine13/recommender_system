Recommender System – Quick Start

A Python recommendation system supporting user-based, time-aware user-based, model-based recommendations, and top-N products. Works via CLI or FastAPI.

Install
# Clone the repo
git clone https://github.com/moonshine13/recommender_system.git
cd recommender_system

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements-dev.txt


CLI Usage
Top-N products
python -m cli.top_n_products_cli --path data/ratings.csv --days 365 --min_ratings 10 --n 5

User-based recommendations
python -m cli.user_rec_cli --path data/ratings.csv --user_id 671 --n 5 --k 5 --rec_type user_based_with_time

Model-based recommendations
python -m cli.model_rec_cli --user_id 671 --exclude_rated True --n 5


Build and Run CLI in Docker
docker build -f Dockerfile.cli -t recommendation-cli .
docker run --rm recommendation-cli cli.model_rec_cli --user_id 671
docker run --rm -v C:/repos/recommender_system/data:/data recommendation-cli  cli.user_rec_cli --path /data/ratings.csv --user_id 671
docker run --rm -v C:/repos/recommender_system/data:/data recommendation-cli  cli.top_n_products_cli --path /data/ratings.csv


Build and Run API in Docker
docker build -t recommendation-api .
docker run -it --rm -p 8000:8000 -v C:/repos/recommender_system/data:/data recommendation-api

Swagger UI: http://127.0.0.1:8000/docs


Testing
pytest tests/

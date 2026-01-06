## API
docker build -t recommendation-api .
docker run -it --rm -p 8000:8000 -v C:/repos/recommender_system/data:/data recommendation-api

## CLI
docker build -f Dockerfile.cli -t recommendation-cli .
docker run --rm recommendation-cli cli.model_rec_cli --user_id 671
docker run --rm -v C:/repos/recommender_system/data:/data recommendation-cli  cli.user_rec_cli --path /data/ratings.csv --user_id 671
docker run --rm -v C:/repos/recommender_system/data:/data recommendation-cli  cli.top_n_products_cli --path /data/ratings.csv

## CMD
python -m cli.top_n_products_cli --path data/ratings.csv
python -m cli.user_rec_cli --path data/ratings.csv --user_id 671
python -m cli.model_rec_cli --user_id 671

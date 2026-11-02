from __future__ import annotations

import threading
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")
DATA_DIR = ROOT_DIR / "data" / "ml-latest-small"
FRONTEND_DIR = ROOT_DIR / "frontend"
MINIMUM_RATINGS = 5
MAX_RECOMMENDATIONS = 10
MODEL_SEED = 42
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")
logger = logging.getLogger(__name__)

_catalog: pd.DataFrame | None = None
_ratings: pd.DataFrame | None = None
_recommender: dict[str, Any] | None = None
_model_lock = threading.Lock()
_poster_cache: dict[int, str | None] = {}
_poster_cache_lock = threading.Lock()


def _load_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    global _catalog, _ratings
    if _catalog is not None and _ratings is not None:
        return _catalog, _ratings

    movies_path = DATA_DIR / "movies.csv"
    ratings_path = DATA_DIR / "ratings.csv"
    if not movies_path.is_file() or not ratings_path.is_file():
        raise FileNotFoundError(
            f"MovieLens CSVs were not found in {DATA_DIR}. "
            "Extract ml-latest-small.zip into the project's data directory."
        )

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies["title"] = movies["title"].astype("string").str.strip()
    movies = movies.dropna(subset=["movieId", "title"])
    movies = movies.loc[(movies["movieId"] > 0) & (movies["movieId"] % 1 == 0)].copy()
    movies["movieId"] = movies["movieId"].astype("int64")
    movies["genres"] = movies["genres"].fillna("(no genres listed)").astype("string").str.strip()
    movies.loc[movies["genres"].eq(""), "genres"] = "(no genres listed)"
    movies = movies.drop_duplicates("movieId", keep="first").copy()

    links_path = DATA_DIR / "links.csv"
    if links_path.is_file():
        links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"])
        links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce")
        links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
        links = links.dropna(subset=["movieId", "tmdbId"])
        links = links.loc[
            (links["movieId"] > 0)
            & (links["movieId"] % 1 == 0)
            & (links["tmdbId"] > 0)
            & (links["tmdbId"] % 1 == 0)
        ].copy()
        links["movieId"] = links["movieId"].astype("int64")
        links["tmdbId"] = links["tmdbId"].astype("int64")
        movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    else:
        logger.warning("MovieLens links.csv is missing; TMDB poster matching is unavailable.")
        movies["tmdbId"] = pd.NA

    for column in ["userId", "movieId", "rating"]:
        ratings[column] = pd.to_numeric(ratings[column], errors="coerce")
    valid_ids = (
        ratings["userId"].notna()
        & ratings["movieId"].notna()
        & (ratings["userId"] > 0)
        & (ratings["movieId"] > 0)
        & (ratings["userId"] % 1 == 0)
        & (ratings["movieId"] % 1 == 0)
    )
    ratings = ratings.loc[valid_ids & ratings["rating"].between(0.5, 5.0)].copy()
    ratings["userId"] = ratings["userId"].astype("int64")
    ratings["movieId"] = ratings["movieId"].astype("int64")
    ratings["timestamp_datetime"] = pd.to_datetime(
        ratings["timestamp"], unit="s", errors="coerce", utc=True
    )
    ratings = (
        ratings.sort_values("timestamp_datetime", na_position="first", kind="stable")
        .drop_duplicates(["userId", "movieId"], keep="last")
        .copy()
    )
    ratings = ratings.loc[ratings["movieId"].isin(movies["movieId"])].copy()

    _catalog = movies.reset_index(drop=True)
    _ratings = ratings.reset_index(drop=True)
    return _catalog, _ratings


def _load_recommender() -> dict[str, Any]:
    global _recommender
    if _recommender is not None:
        return _recommender

    with _model_lock:
        if _recommender is not None:
            return _recommender

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required to make recommendations. "
                "Install the project's requirements in your Python 3.11 environment."
            ) from exc

        movies, ratings = _load_catalog()
        np.random.seed(MODEL_SEED)
        tf.keras.utils.set_random_seed(MODEL_SEED)

        user_ids = ratings["userId"].unique()
        movie_ids = ratings["movieId"].unique()
        user_to_index = {int(user_id): index for index, user_id in enumerate(user_ids)}
        movie_to_index = {int(movie_id): index for index, movie_id in enumerate(movie_ids)}

        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(movies["genres"].str.split("|"))
        genre_df = pd.DataFrame(genre_matrix, index=movies["movieId"])

        train, _ = train_test_split(ratings, test_size=0.2, random_state=MODEL_SEED)
        train = train.copy()
        train["user_index"] = train["userId"].map(user_to_index)
        train["movie_index"] = train["movieId"].map(movie_to_index)
        train_genres = genre_df.loc[train["movieId"]].values.astype("float32")
        model_inputs = {
            "user": train["user_index"].to_numpy(dtype=np.int32),
            "movie": train["movie_index"].to_numpy(dtype=np.int32),
            "genres": train_genres,
        }
        targets = train["rating"].to_numpy(dtype=np.float32)

        class HybridMovieRecommender(tf.keras.Model):
            def __init__(self) -> None:
                super().__init__()
                self.user_embedding = tf.keras.layers.Embedding(len(user_to_index), 50)
                self.movie_embedding = tf.keras.layers.Embedding(len(movie_to_index), 50)
                self.user_bias = tf.keras.layers.Embedding(len(user_to_index), 1)
                self.movie_bias = tf.keras.layers.Embedding(len(movie_to_index), 1)
                self.genre_dense = tf.keras.layers.Dense(50, activation="relu")

            def call(self, inputs: dict[str, Any]) -> Any:
                user_indices = inputs["user"]
                movie_indices = inputs["movie"]
                genres = inputs["genres"]
                user_vector = self.user_embedding(user_indices)
                movie_vector = self.movie_embedding(movie_indices)
                user_bias = self.user_bias(user_indices)
                movie_bias = self.movie_bias(movie_indices)
                genre_vector = self.genre_dense(genres)
                interaction = tf.reduce_sum(
                    user_vector * (movie_vector + genre_vector), axis=1
                )
                return (
                    interaction
                    + tf.squeeze(user_bias, axis=1)
                    + tf.squeeze(movie_bias, axis=1)
                )

        model = HybridMovieRecommender()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
        model.fit(
            model_inputs,
            targets,
            epochs=30,
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0,
        )

        _recommender = {
            "model": model,
            "movies": movies,
            "movie_to_index": movie_to_index,
            "genre_df": genre_df,
        }
        return _recommender


def _movie_payload(movie: pd.Series) -> dict[str, Any]:
    return {
        "movieId": int(movie["movieId"]),
        "title": str(movie["title"]),
        "genres": str(movie["genres"]),
        "tmdbId": int(movie["tmdbId"]) if pd.notna(movie.get("tmdbId")) else None,
    }


def _tmdb_poster_path(tmdb_id: int) -> str | None:
    token = os.environ.get("TMDB_ACCESS_TOKEN", "").strip()
    if not token:
        return None

    with _poster_cache_lock:
        if tmdb_id in _poster_cache:
            return _poster_cache[tmdb_id]

    request_obj = Request(
        f"{TMDB_API_BASE}/movie/{tmdb_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request_obj, timeout=4) as response:
            details = json.load(response)
    except HTTPError as exc:
        logger.warning("TMDB poster lookup failed for TMDB movie %s (HTTP %s).", tmdb_id, exc.code)
        return None
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("TMDB poster lookup failed for TMDB movie %s: %s", tmdb_id, exc)
        return None

    if not isinstance(details, dict):
        logger.warning("TMDB returned an invalid response for TMDB movie %s.", tmdb_id)
        return None

    poster_path = details.get("poster_path")
    if poster_path is not None and (
        not isinstance(poster_path, str)
        or not poster_path.startswith("/")
        or "://" in poster_path
    ):
        logger.warning("TMDB returned an invalid poster path for TMDB movie %s.", tmdb_id)
        poster_path = None

    with _poster_cache_lock:
        _poster_cache[tmdb_id] = poster_path
    return poster_path


def _with_poster_urls(movies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not os.environ.get("TMDB_ACCESS_TOKEN", "").strip():
        return [{**movie, "posterUrl": None} for movie in movies]

    tmdb_ids = sorted({
        movie["tmdbId"]
        for movie in movies
        if movie.get("tmdbId") is not None
    })
    with ThreadPoolExecutor(max_workers=6) as executor:
        poster_paths = dict(zip(tmdb_ids, executor.map(_tmdb_poster_path, tmdb_ids)))

    return [
        {
            **movie,
            "posterUrl": (
                f"{TMDB_IMAGE_BASE}{poster_paths[movie['tmdbId']]}"
                if movie.get("tmdbId") is not None
                and poster_paths.get(movie["tmdbId"])
                else None
            ),
        }
        for movie in movies
    ]


@app.get("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "minimumRatings": MINIMUM_RATINGS,
            "tmdbConfigured": bool(os.environ.get("TMDB_ACCESS_TOKEN", "").strip()),
        }
    )


@app.get("/api/movies/popular")
def popular_movies():
    movies, ratings = _load_catalog()
    counts = (
        ratings.groupby("movieId")
        .agg(rating_count=("rating", "size"), average_rating=("rating", "mean"))
        .reset_index()
    )
    popular = (
        movies.merge(counts, on="movieId", how="inner")
        .sort_values(["rating_count", "average_rating"], ascending=False)
        .head(10)
    )
    payload = [_movie_payload(movie) for _, movie in popular.iterrows()]
    return jsonify(_with_poster_urls(payload))


@app.get("/api/movies/search")
def search_movies():
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify([])

    movies, ratings = _load_catalog()
    known_movie_ids = set(ratings["movieId"].unique())
    matches = movies.loc[
        movies["movieId"].isin(known_movie_ids)
        & movies["title"].str.contains(query, case=False, na=False, regex=False)
    ].head(12)
    payload = [_movie_payload(movie) for _, movie in matches.iterrows()]
    return jsonify(_with_poster_urls(payload))


@app.post("/api/recommendations")
def recommendations():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("ratings"), list):
        return jsonify({"error": "Send ratings as a list of movieId and rating values."}), 400

    submitted = payload["ratings"]
    if len(submitted) < MINIMUM_RATINGS:
        return jsonify(
            {"error": f"Rate at least {MINIMUM_RATINGS} movies before requesting recommendations."}
        ), 400

    selected: dict[int, float] = {}
    for entry in submitted:
        if not isinstance(entry, dict):
            return jsonify({"error": "Each rating must include movieId and rating."}), 400
        try:
            movie_id = int(entry["movieId"])
            rating = float(entry["rating"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return jsonify({"error": "Each rating must include a valid movieId and rating."}), 400
        if not np.isfinite(rating) or not 0.5 <= rating <= 5.0 or not np.isclose(rating * 2, round(rating * 2)):
            return jsonify({"error": "Ratings must be from 0.5 to 5.0 in half-star increments."}), 400
        if movie_id in selected:
            return jsonify({"error": "A movie can only be rated once."}), 400
        selected[movie_id] = rating

    try:
        recommender = _load_recommender()
    except (FileNotFoundError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 503

    movie_to_index = recommender["movie_to_index"]
    unknown_ids = [movie_id for movie_id in selected if movie_id not in movie_to_index]
    if unknown_ids:
        return jsonify({"error": "One or more selected movies are not supported by the model."}), 400

    tf = __import__("tensorflow")
    model = recommender["model"]
    genre_df = recommender["genre_df"]
    rated_ids = list(selected)
    rated_indices = np.array([movie_to_index[movie_id] for movie_id in rated_ids], dtype=np.int32)
    rated_genres = genre_df.loc[rated_ids].values.astype("float32")
    rated_genre_vectors = model.genre_dense(tf.convert_to_tensor(rated_genres)).numpy()
    movie_embeddings = model.movie_embedding.get_weights()[0]
    movie_biases = model.movie_bias.get_weights()[0].reshape(-1)
    rated_items = movie_embeddings[rated_indices] + rated_genre_vectors

    design = np.column_stack([rated_items, np.ones(len(rated_indices))])
    target_ratings = np.array([selected[movie_id] for movie_id in rated_ids], dtype=np.float32)
    targets = target_ratings - movie_biases[rated_indices]
    penalty = np.eye(design.shape[1], dtype=np.float32) * 5.0
    penalty[-1, -1] = 0.01
    profile = np.linalg.solve(design.T @ design + penalty, design.T @ targets)

    movies = recommender["movies"]
    candidates = movies.loc[
        movies["movieId"].isin(movie_to_index)
        & ~movies["movieId"].isin(rated_ids)
    ].copy()
    candidate_ids = candidates["movieId"].tolist()
    candidate_indices = np.array([movie_to_index[movie_id] for movie_id in candidate_ids])
    candidate_genres = genre_df.loc[candidate_ids].values.astype("float32")
    candidate_genre_vectors = model.genre_dense(tf.convert_to_tensor(candidate_genres)).numpy()
    candidate_vectors = movie_embeddings[candidate_indices] + candidate_genre_vectors
    scores = (
        candidate_vectors @ profile[:-1]
        + profile[-1]
        + movie_biases[candidate_indices]
    )
    candidates["predicted_rating"] = scores
    top_movies = candidates.nlargest(MAX_RECOMMENDATIONS, "predicted_rating")
    payload = [
        {
            **_movie_payload(movie),
            "predictedRating": float(np.clip(movie["predicted_rating"], 0.5, 5.0)),
        }
        for _, movie in top_movies.iterrows()
    ]
    return jsonify(_with_poster_urls(payload))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

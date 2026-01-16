import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import random

movies = pd.read_csv("data/movies.csv")

def convert(text):
    try:
        return [i["name"] for i in ast.literal_eval(text)]
    except:
        return []

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["overview"] = movies["overview"].fillna("")

movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"].apply(lambda x: " ".join(x)) + " " +
    movies["keywords"].apply(lambda x: " ".join(x))
)

movies = movies[["title", "tags", "vote_average", "popularity"]]
movies["tags"] = movies["tags"].str.lower()

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
vectors = tfidf.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
movies["cluster"] = kmeans.fit_predict(vectors)

def hybrid_recommend(movie_name, top_n=5):
    titles = movies["title"].str.lower()
    if movie_name.lower() not in titles.values:
        return recommend_popular(top_n)

    index = titles[titles == movie_name.lower()].index[0]
    cluster_id = movies.iloc[index]["cluster"]

    cluster_indices = movies[movies["cluster"] == cluster_id].index
    cluster_indices = [i for i in cluster_indices if i != index]

    scored = [(i, similarity[index][i]) for i in cluster_indices]
    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, _ in scored[:top_n]]
    return movies.loc[top_indices]["title"].tolist()

def recommend_by_genre(genre, top_n=5):
    genre = genre.lower()
    filtered = movies[movies["tags"].str.contains(genre)]
    if len(filtered) == 0:
        return recommend_popular(top_n)
    pool = filtered.sort_values(by="vote_average", ascending=False).head(80)
    return pool.sample(min(top_n, len(pool)))["title"].tolist()

mood_map = {
    "happy": ["comedy", "family"],
    "sad": ["drama", "romance"],
    "bored": ["thriller", "mystery"],
    "excited": ["action", "adventure"],
    "thinking": ["sci-fi", "psychological"]
}

def recommend_by_mood(mood, top_n=5):
    genres = mood_map.get(mood.lower(), [])
    if not genres:
        return recommend_popular(top_n)
    filtered = movies[movies["tags"].str.contains("|".join(genres))]
    pool = filtered.sort_values(by="vote_average", ascending=False).head(80)
    return pool.sample(min(top_n, len(pool)))["title"].tolist()

def recommend_popular(top_n=5):
    movies["score"] = movies["vote_average"] * 0.7 + movies["popularity"] * 0.3
    pool = movies.sort_values(by="score", ascending=False).head(100)
    return pool.sample(top_n)["title"].tolist()

def detect_mood_from_text(text):
    text = text.lower()
    if any(w in text for w in ["happy", "fun", "comedy", "light"]):
        return "happy"
    if any(w in text for w in ["sad", "emotional", "romantic"]):
        return "sad"
    if any(w in text for w in ["bored", "thrill", "mystery"]):
        return "bored"
    if any(w in text for w in ["action", "fight", "fast"]):
        return "excited"
    if any(w in text for w in ["mind", "think", "deep", "sci"]):
        return "thinking"
    return None

def smart_text_recommend(user_text):
    mood = detect_mood_from_text(user_text)
    if mood:
        return recommend_by_mood(mood)
    return recommend_popular()

def smart_recommend(movie_name=None, mood=None, genre=None):
    if movie_name:
        return hybrid_recommend(movie_name)
    if genre:
        return recommend_by_genre(genre)
    if mood:
        return recommend_by_mood(mood)
    return recommend_popular()

def get_movie_genres(title):
    row = movies[movies["title"].str.lower() == title.lower()]
    if row.empty:
        return set()
    return set(row.iloc[0]["tags"].split())

def precision_at_k(movie_name, k=5):
    recommendations = hybrid_recommend(movie_name, top_n=k)
    if not recommendations:
        return 0.0
    query_genres = get_movie_genres(movie_name)
    if not query_genres:
        return 0.0
    relevant = 0
    for rec in recommendations:
        if query_genres.intersection(get_movie_genres(rec)):
            relevant += 1
    return relevant / k

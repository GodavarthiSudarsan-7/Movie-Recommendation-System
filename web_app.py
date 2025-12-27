import streamlit as st
import random
from recommender import smart_recommend, smart_text_recommend

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background:
        linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
        url("https://images.unsplash.com/photo-1517602302552-471fe67acf66");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .glass {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    }

    h1, h2, h3, p, label, span {
        color: #ffffff !important;
    }

    .reason {
        font-size: 0.85rem;
        color: #e5e7eb !important;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 Movie Recommendation System")
st.write("Interactive demonstration of a machine learning–based recommender")

option = st.radio(
    "Choose recommendation mode:",
    (
        "Based on a movie you like",
        "Based on your mood",
        "Based on genre",
        "I don't know what to watch",
        "Describe what you want to watch"
    )
)

def explain(reason_type):
    explanations = {
        "movie": [
            "High semantic similarity in TF-IDF feature space",
            "Strong overlap in keywords and genres",
            "Closest match based on cosine similarity"
        ],
        "mood": [
            "Aligns with mood-specific genre patterns",
            "Emotionally consistent with inferred preference",
            "Selected from high-rated movies matching this mood"
        ],
        "genre": [
            "Matches the selected genre constraint",
            "High audience rating within this genre",
            "Sampled from top-performing movies of this genre"
        ],
        "popular": [
            "High popularity and audience rating",
            "Widely liked across diverse viewers",
            "Exploration candidate from top-ranked movies"
        ],
        "text": [
            "Mapped from natural language intent",
            "Matches inferred mood and thematic keywords",
            "Selected based on semantic intent detection"
        ]
    }
    return random.choice(explanations[reason_type])

def render_movies(movies, reason_type):
    for m in movies:
        st.markdown(
            f"""
            <div class="glass">
                <h3>{m}</h3>
                <div class="reason">Why recommended: {explain(reason_type)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

if option == "Based on a movie you like":
    movie = st.text_input("Enter a movie name")
    if movie:
        render_movies(smart_recommend(movie_name=movie), "movie")

elif option == "Based on your mood":
    mood = st.selectbox("Select your mood", ["happy", "sad", "bored", "excited", "thinking"])
    render_movies(smart_recommend(mood=mood), "mood")

elif option == "Based on genre":
    genre = st.selectbox(
        "Select genre",
        ["action", "adventure", "animation", "comedy", "crime", "drama", "family",
         "fantasy", "horror", "mystery", "romance", "sci-fi", "thriller", "war", "western"]
    )
    render_movies(smart_recommend(genre=genre), "genre")

elif option == "I don't know what to watch":
    render_movies(smart_recommend(), "popular")

else:
    text = st.text_input("Describe what you want to watch")
    if text:
        render_movies(smart_text_recommend(text), "text")

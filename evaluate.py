from recommender import precision_at_k
import pandas as pd

movies = pd.read_csv("data/movies.csv")

sample_movies = movies["title"].dropna().sample(20, random_state=42)

scores = []
for title in sample_movies:
    scores.append(precision_at_k(title, k=5))

average_precision = sum(scores) / len(scores)

print("Evaluation Results")
print("------------------")
print(f"Average Precision@5: {round(average_precision, 3)}")

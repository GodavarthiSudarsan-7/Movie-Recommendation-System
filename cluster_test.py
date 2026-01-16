from recommender import movies

clusters = {}

for _, row in movies.iterrows():
    clusters.setdefault(row["cluster"], []).append(row["title"])

for c in list(clusters.keys())[:5]:
    print(f"\nCluster {c}")
    for t in clusters[c][:5]:
        print("-", t)

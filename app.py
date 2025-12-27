from recommender import smart_recommend, smart_text_recommend

print("🎬 MOVIE RECOMMENDATION SYSTEM")
print("--------------------------------")
print("1. Recommend based on a movie you like")
print("2. Recommend based on your mood")
print("3. I don't know what to watch")
print("4. Describe what you want to watch")

choice = input("Choose an option (1/2/3/4): ")

if choice == "1":
    movie = input("Enter a movie name: ")
    recommendations = smart_recommend(movie_name=movie)

elif choice == "2":
    mood = input("Enter your mood (happy/sad/bored/excited/thinking): ")
    recommendations = smart_recommend(mood=mood)

elif choice == "4":
    text = input("Describe what you want to watch: ")
    recommendations = smart_text_recommend(text)

else:
    recommendations = smart_recommend()

print("\n🍿 Recommended Movies:")
for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie}")

# https://surpriselib.com/
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# pip install scikit-surprise


#Step 1: Collaborative Filtering
#Using Matrix Factorization (Surprise Library):
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from AIEx_interaction_data import interaction_data
from topic5.AIEX.AIEx_movies_data import movies_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load interaction data
# Load the interaction data
# read User-Movie
file_path_user_movies = "User-Movie.csv"
interaction_data_df = interaction_data(file_path_user_movies)

# read Movies
file_path_movies = "Movie.csv"
movies_data_df = movies_data(file_path_movies)
print("movies_data_df",movies_data_df["genre"])
reader = Reader(rating_scale=(1, 5))

# Check if data was loaded successfully
if interaction_data_df is not None:
    data = Dataset.load_from_df(interaction_data_df[['user_id', 'movie_id', 'rating']], reader)
    #print("data:", data)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    #print("trainset:", trainset)
    #print("testset:", testset)

    # Train collaborative filtering model
    model = SVD()
    model.fit(trainset)

    # Evaluate model
    predictions = model.test(testset)
    print("RMSE:", accuracy.rmse(predictions))

    # Generate recommendations for a user
    user_id = 1
    movie_id = 103
    predicted_rating = model.predict(user_id, movie_id).est
    print(f"Predicted rating for user {user_id} and movie {movie_id}: {predicted_rating}")
else:
    print("Failed to load interaction data.")

# Step 2: Content-Based Filtering
# Using TF-IDF for Movie Descriptions:
# Check if data was loaded successfully
if movies_data_df is not None:
    # Combine movie metadata into a single text field
    movies_data_df['combined_features'] = movies_data_df['genre'] + " " + movies_data_df['director'] + " " + movies_data_df['cast']

    # Compute TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_data_df['combined_features'])

    # Compute similarity scores
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Recommend movies similar to a given movie
    movie_idx = 0  # Example: Recommend movies similar to the first movie
    similar_movies = similarity_matrix[movie_idx].argsort()[-5:][::-1]
    print(f"Movies similar to movie {movie_idx}: {similar_movies}")
else:
    print("Failed to load movies data.")

# Step 3: Reinforcement Learning
import numpy as np
from sklearn.linear_model import LogisticRegression

# Simulate user context and rewards
n_actions = 5  # Number of movies to recommend
n_features = 3  # User context features (e.g., preferences)
context = np.random.rand(1000, n_features)
rewards = np.random.randint(2, size=(1000, n_actions))  # 1: Clicked, 0: Not clicked

# Train logistic regression for each action
models = [LogisticRegression().fit(context, rewards[:, action]) for action in range(n_actions)]

# Recommend movie for a new user context
new_context = np.random.rand(1, n_features)
action_scores = [model.predict_proba(new_context)[:, 1] for model in models]
recommended_action = np.argmax(action_scores)
print(f"Recommended movie: {recommended_action}")
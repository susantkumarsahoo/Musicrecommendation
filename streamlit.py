import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



# Load your dataset
df = pd.read_csv('clustered_df.csv')


numerical_features = [
    "valence", "danceability", "energy", "tempo",
    "acousticness", "liveness", "speechiness", "instrumentalness"
]

def recommend_songs(song_name, df, num_recommendations=5):
    # Get the cluster of the input song
    song_cluster = df[df["name"] == song_name]["Cluster"].values[0]
    same_cluster_songs = df[df["Cluster"] == song_cluster]
    song_index = same_cluster_songs[same_cluster_songs["name"] == song_name].index[0]
    cluster_features = same_cluster_songs[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)
    similar_songs = np.argsort(similarity[song_index])[-(num_recommendations + 1):-1][::-1]
    recommendations = same_cluster_songs.iloc[similar_songs][["name", "year", "artists"]]
    return recommendations


# Streamlit app
st.title("Songs Recommendation System")

# Create a text input box
user_input = st.text_input("Enter a song name:")

# Create a submit button
if st.button("Submit"):
    if user_input:
        recommended_songs = recommend_songs(user_input, df, num_recommendations=5)
        if recommended_songs is not None:
            st.write(f"Songs similar to '{user_input}':")
            st.table(recommended_songs.reset_index())
        else:
            st.error("Sorry, the song was not found or recommendations are unavailable.")
    else:
        st.warning("Please enter a song name before submitting!")











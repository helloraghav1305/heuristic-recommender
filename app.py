import gradio as gr
from fastai.collab import *
from fastai.tabular.all import *
import pandas as pd

def load_data_and_dls():
    path = untar_data(URLs.ML_100k)
    ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None, 
                          names=['user','movie','rating','timestamp'])
    movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
    ratings = ratings.merge(movies)
    ratings.drop(columns='timestamp', inplace=True)

    dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
    return ratings, dls

ratings, dls = load_data_and_dls()

def recommend_heuristic(user_likes, top_n=5):
    df = ratings.copy()
    liked_users = df[df['title'].isin(user_likes) & (df['rating'] >= 4)]
    similar_users = liked_users['user'].unique()
    candidate_movies = df[(df['user'].isin(similar_users)) & (~df['title'].isin(user_likes))]
    top_movies = (candidate_movies.groupby('title')
                  .agg({'rating': ['mean', 'count']})
                  .reset_index())
    top_movies.columns = ['title', 'mean_rating', 'count']
    top_movies = top_movies[top_movies['count'] > 5].sort_values(by='mean_rating', ascending=False)
    return top_movies['title'].head(top_n).to_list()

all_titles = sorted(ratings['title'].unique())

def recommend(user_movies):
    if len(user_movies) < 3:
        return "Select at least 3 movies.", [], "-", []
    
    heuristic_recs = recommend_heuristic(user_movies)

    return (
        "Heuristic (Memory-Based)",
        heuristic_recs
    )


with gr.Blocks(title="Movie Recommender") as demo:
    gr.Markdown("## Movie Recommendation System")
    gr.Markdown("Select 3-5 movies you liked. Compare results from three approaches.")

    movie_selector = gr.Dropdown(choices=all_titles, multiselect=True, label="Select movies you liked:")

    btn = gr.Button("Get Recommendations")

    label_h = gr.Label()
    out_h = gr.Textbox(label="Heuristic Recommendations")

    btn.click(fn=recommend, inputs=movie_selector, 
              outputs=[label_h, out_h])

demo.launch()

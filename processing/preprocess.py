import os
import string
import pickle
import pandas as pd
import ast
import nltk
import requests
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ps = PorterStemmer()
nltk.download('stopwords')

# fallback images
FALLBACK_POSTER = "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"
FALLBACK_PERSON = "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"


def get_genres(obj):
    lista = ast.literal_eval(obj)
    return [i['name'] for i in lista]


def get_cast(obj):
    a = ast.literal_eval(obj)
    return [a[i]['name'] for i in range(min(10, len(a)))]


def get_crew(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []


def read_csv_to_df():
    credit_ = pd.read_csv(r'Files/tmdb_5000_credits.csv')
    movies = pd.read_csv(r'Files/tmdb_5000_movies.csv')

    movies = movies.merge(credit_, on='title')

    movies2 = movies.copy()
    movies2.drop(['homepage', 'tagline'], axis=1, inplace=True)
    movies2 = movies2[['movie_id', 'title', 'budget', 'overview', 'popularity', 'release_date', 'revenue', 'runtime',
                       'spoken_languages', 'status', 'vote_average', 'vote_count']]

    movies = movies[
        ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'production_companies',
         'release_date']]
    movies.dropna(inplace=True)

    movies['genres'] = movies['genres'].apply(get_genres)
    movies['keywords'] = movies['keywords'].apply(get_genres)
    movies['top_cast'] = movies['cast'].apply(get_cast)
    movies['director'] = movies['crew'].apply(get_crew)
    movies['prduction_comp'] = movies['production_companies'].apply(get_genres)

    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcast'] = movies['top_cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcrew'] = movies['director'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tprduction_comp'] = movies['prduction_comp'].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['tcast'] + movies['tcrew']

    new_df = movies[['movie_id', 'title', 'tags', 'genres', 'keywords', 'tcast', 'tcrew', 'tprduction_comp']]

    new_df['genres'] = new_df['genres'].apply(lambda x: " ".join(x))
    new_df['tcast'] = new_df['tcast'].apply(lambda x: " ".join(x))
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: " ".join(x))

    new_df['tcast'] = new_df['tcast'].apply(lambda x: x.lower())
    new_df['genres'] = new_df['genres'].apply(lambda x: x.lower())
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: x.lower())

    new_df['tags'] = new_df['tags'].apply(stemming_stopwords)
    new_df['keywords'] = new_df['keywords'].apply(stemming_stopwords)

    return movies, new_df, movies2


def stemming_stopwords(li):
    ans = [ps.stem(i) for i in li]

    stop_words = set(stopwords.words('english'))
    filtered = [w.lower() for w in ans if w.lower() not in stop_words]

    return " ".join([i for i in filtered if len(i) > 2])


# ✅ OMDb: poster + rating + year
def fetch_posters(movie_name):
    try:
        url = f"http://www.omdbapi.com/?t={movie_name}&apikey=3172db60"
        data = requests.get(url, timeout=5).json()

        poster = data.get('Poster')
        if not poster or poster == "N/A":
            poster = FALLBACK_POSTER
        rating = data.get('imdbRating', "N/A")
        year = data.get('Year', "N/A")

        return poster, rating, year

    except:
        return FALLBACK_POSTER, "N/A", "N/A"


def recommend(new_df, movie, pickle_file_path):
    try:
        # 🔥 auto-generate if missing
        if not os.path.exists(pickle_file_path):
            print(f"Generating {pickle_file_path}...")

            if "genres" in pickle_file_path:
                similarity_tags = vectorise(new_df, 'genres')
            elif "keywords" in pickle_file_path:
                similarity_tags = vectorise(new_df, 'keywords')
            elif "tcast" in pickle_file_path:
                similarity_tags = vectorise(new_df, 'tcast')
            elif "tprduction_comp" in pickle_file_path:
                similarity_tags = vectorise(new_df, 'tprduction_comp')
            else:
                similarity_tags = vectorise(new_df, 'tags')

            with open(pickle_file_path, 'wb') as f:
                pickle.dump(similarity_tags, f)

        else:
            with open(pickle_file_path, 'rb') as f:
                similarity_tags = pickle.load(f)

        movie_idx = new_df[new_df['title'] == movie].index[0]

        movie_list = sorted(
            list(enumerate(similarity_tags[movie_idx])),
            reverse=True,
            key=lambda x: x[1]
        )[1:26]

        rec_movie_list = []
        rec_poster_list = []
        rec_rating_list = []
        rec_year_list = []

        for i in movie_list:
            movie_name = new_df.iloc[i[0]]['title']
            poster, rating, year = fetch_posters(movie_name)

            rec_movie_list.append(movie_name)
            rec_poster_list.append(poster)
            rec_rating_list.append(rating)
            rec_year_list.append(year)

        return rec_movie_list, rec_poster_list, rec_rating_list, rec_year_list

    except Exception as e:
        print("Error:", e)

        fallback_movies = new_df['title'].head(10).values
        fallback_posters = ["https://via.placeholder.com/300x450"] * 10
        fallback_ratings = ["N/A"] * 10
        fallback_years = ["N/A"] * 10

        return fallback_movies, fallback_posters, fallback_ratings, fallback_years


def vectorise(new_df, col_name):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vec_tags = cv.fit_transform(new_df[col_name]).toarray()
    return cosine_similarity(vec_tags)


def fetch_person_details(id_):
    return FALLBACK_PERSON, "No biography available"


# ✅ UPDATED: full plot + cast
def get_details(selected_movie_name):
    try:
        # load local data
        with open(r'Files/movies_dict.pkl', 'rb') as f:
            movies = pd.DataFrame.from_dict(pickle.load(f))

        with open(r'Files/movies2_dict.pkl', 'rb') as f:
            movies2 = pd.DataFrame.from_dict(pickle.load(f))

        a = movies2[movies2['title'] == selected_movie_name]
        b = movies[movies['title'] == selected_movie_name]

        budget = a.iloc[0, 2]
        release_date = a.iloc[:, 5].iloc[0]
        revenue = a.iloc[:, 6].iloc[0]
        runtime = a.iloc[:, 7].iloc[0]
        vote_count = a.iloc[:, 11].iloc[0]

        genres = b.iloc[:, 3].iloc[0]
        director = b.iloc[:, 10].iloc[0]

        # 🔥 OMDb full data
        url = f"http://www.omdbapi.com/?t={selected_movie_name}&plot=full&apikey=OMDBAPIKEY"
        data = requests.get(url, timeout=5).json()

        overview = data.get('Plot', "No description available")
        rating = data.get('imdbRating', "N/A")
        year = data.get('Year', "N/A")

        cast = data.get('Actors', "No data")
        cast_list = cast.split(", ") if cast != "No data" else []

        poster = data.get('Poster') if data.get('Poster') != "N/A" else FALLBACK_POSTER

        return [
            poster, budget, genres, overview, release_date,
            revenue, runtime, [], rating, vote_count,
            0, cast_list, director, [], []
        ]

    except:
        return [
            FALLBACK_POSTER, "N/A", [], "No data available", "N/A",
            "N/A", "N/A", [], "N/A", "N/A",
            0, [], [], [], []
        ]
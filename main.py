import streamlit as st
import streamlit_option_menu
from processing import preprocess
from processing.display import Main

st.set_page_config(layout="wide")

displayed = []

if 'movie_number' not in st.session_state:
    st.session_state['movie_number'] = 0

if 'selected_movie_name' not in st.session_state:
    st.session_state['selected_movie_name'] = ""

if 'user_menu' not in st.session_state:
    st.session_state['user_menu'] = ""


def main():

    def initial_options():
        st.session_state.user_menu = streamlit_option_menu.option_menu(
            menu_title='What are you looking for? 👀',
            options=['Recommend me a similar movie', 'Describe the movie', 'Check all Movies'],
            icons=['film', 'film', 'film'],
            menu_icon='list',
            orientation="horizontal",
        )

        if st.session_state.user_menu == 'Recommend me a similar movie':
            recommend_display()

        elif st.session_state.user_menu == 'Describe the movie':
            display_movie_details()

        elif st.session_state.user_menu == 'Check all Movies':
            paging_movies()

    def recommend_display():
        st.title('Movie Recommender System')

        selected_movie_name = st.selectbox(
            'Select a Movie...', new_df['title'].values
        )

        if st.button('Recommend'):
            st.session_state.selected_movie_name = selected_movie_name

            with st.spinner(f"Recommending movies based on '{selected_movie_name}'..."):

                # 🔥 MAIN (BEST)
                st.header("Recommended Movies ")
                recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_tags.pkl')

                # 🎭 GENRE
                st.header("🎭 Based on Genre")
                recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_genres.pkl')

                # 👥 CAST / DIRECTOR
                st.header("👥 Based on Cast")
                recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_tcast.pkl')

            st.success("Recommendations ready ✅")

    def recommendation_tags(new_df, selected_movie_name, pickle_file_path):

        movies, posters, ratings, years = preprocess.recommend(new_df, selected_movie_name, pickle_file_path)

        rec_movies = []
        rec_posters = []
        rec_ratings = []
        rec_years = []

        for i, j in enumerate(movies):
            if len(rec_movies) == 5:
                break
            if j not in displayed:
                rec_movies.append(j)
                rec_posters.append(posters[i])
                rec_ratings.append(ratings[i])
                rec_years.append(years[i])
                displayed.append(j)

        cols = st.columns(5)

        for i in range(len(rec_movies)):
            with cols[i]:
                poster = rec_posters[i] if rec_posters[i] else "https://via.placeholder.com/300x450"
                rating = rec_ratings[i] if rec_ratings[i] != "N/A" else "No Rating"
                year = rec_years[i] if rec_years[i] != "N/A" else "Unknown"

                st.image(poster)
                st.text(rec_movies[i])
                st.text(f"⭐ {rating} | 📅 {year}")

    def display_movie_details():
        selected_movie_name = st.session_state.selected_movie_name

        info = preprocess.get_details(selected_movie_name)

        with st.container():
            image_col, text_col = st.columns((1, 2))

            with image_col:
                st.image(info[0])

            with text_col:
                st.title(selected_movie_name)

                col1, col2, col3 = st.columns(3)
                col1.write(f"⭐ Rating: {info[8]}")
                col2.write(f"Votes: {info[9]}")
                col3.write(f"Runtime: {info[6]}")

                # 🎬 TRAILER BUTTON (FIXED)
                youtube_url = f"https://www.youtube.com/results?search_query={selected_movie_name}+official+trailer"
                st.link_button("▶ Watch Trailer", youtube_url)

                st.markdown(f"**Overview:**\n\n{info[3]}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text("Release Date")
                    st.text(info[4])
                with col2:
                    st.text("Budget")
                    st.text(info[1])
                with col3:
                    st.text("Revenue")
                    st.text(info[5])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text("Genres")
                    st.write(", ".join(info[2]))
                with col2:
                    st.text("Director")
                    st.write(info[12][0] if info[12] else "N/A")

        # 👨‍🎬 CAST
        st.header('Cast')

        if info[11]:
            st.markdown("**Cast:** " + ", ".join(info[11][:5]))
        else:
            st.write("No cast information available")

    def paging_movies():

        max_pages = int(movies.shape[0] / 10) - 1

        col1, col2, col3 = st.columns([1, 9, 1])

        with col1:
            if st.button("Prev"):
                st.session_state['movie_number'] = max(0, st.session_state['movie_number'] - 10)

        with col2:
            page = st.slider("Page", 0, max_pages, st.session_state['movie_number'] // 10)
            st.session_state['movie_number'] = page * 10

        with col3:
            if st.button("Next"):
                if st.session_state['movie_number'] + 10 < len(movies):
                    st.session_state['movie_number'] += 10

        display_all_movies(st.session_state['movie_number'])

    def display_all_movies(start):

        i = start
        cols = st.columns(5)

        for col in cols:
            if i >= len(movies):
                break

            movie_name = movies.iloc[i]['title']
            poster, rating, year = preprocess.fetch_posters(movie_name)

            with col:
                st.image(poster if poster else "https://via.placeholder.com/300x450")
                st.text(movie_name)
                st.text(f"⭐ {rating} | 📅 {year}")

            i += 1

    with Main() as bot:
        bot.main_()
        new_df, movies, movies2 = bot.getter()
        initial_options()


if __name__ == '__main__':
    main()
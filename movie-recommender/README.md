##### PERSONALIZED MOVIE RECOMMENDATION SYSTEM #####

I. Overview: This is a machine learning-based movie recommendation application which tailors suggestions based on user preferences and behavior. 

II. User need assessment: Real users were surveyed regarding their experiences watching movies and interacting with movie suggestion avenues. Participants were asked questions about user behavior, movie preferences, rating and viewing history, and user needs.

This assessment provided insight into what a user wants from a movie recommendation system. Although a supplementary movie review dataset was used to train the model, this assessment gave structure to the goals this system should achieve. As a result, this system was created with the goals of usability, personalization, and explanation in mind. These efforts will hopefully make this platform accessible and appealing to a larger population.

III. The MovieLens Dataset: The MovieLens 100K Dataset was leveraged within this project to train the model and provide accurate recommendations to users. 
    * Please note that this dataset is not included in this repo.
    * The exact dataset used is the MovieLens Latest Small dataset
    * Please download the "ml-latest-small.zip" file from MovieLens and place this zip file in the "data" folder of this repo. Then, unzip the zip file in the "data" folder to ensure all dependencies are present.

IV. Environment: TensorFlow, an open source machine learning platform, and Juptyer notebooks are utilized within this project. A conda environment was created to include the tensorflow, pandas, numpy, matplotlib, jupyter, and ipykernel packages. Then, the environment was registered with Juptyer. To create the environment, the following commands were run:

    conda create -n movie-recommender python=3.11

    conda activate movie-recommender

    pip install tensorflow pandas numpy matplotlib jupyter ipykernel

    python -m ipykernel install --user --name movie-recommender --display-name "Movie Recommender"

V. Run the web app: The MovieMatch frontend is built with HTML, CSS, and JavaScript. A small Flask API serves the frontend and connects it to the TensorFlow recommender. From the project root, activate a Python 3.11 environment and install the app dependencies:

    conda activate movie-recommender

    python -m pip install -r requirements.txt

The MovieLens Latest Small CSV files should be extracted under `data/ml-latest-small/`. Start the app from the project root:

    python app.py

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The app searches titles, accepts ratings from 0.5 to 5 in half-star increments, and requires at least five ratings before requesting recommendations. On the first recommendation request after the server starts, the API trains the hybrid model in memory using the same model structure and training settings as the notebook. The browser then displays the ten highest-scoring unrated movies as cards. MovieLens does not include poster images, so the cards use locally styled artwork rather than external poster images.

To load TMDB posters, copy `.env.example` to `.env` and set `TMDB_ACCESS_TOKEN` to your TMDB API Read Access Token in that local file. Restart `python app.py` after saving it. The token stays on the server and is not sent to the browser or included in the repository; `.env` is ignored by Git. The app matches MovieLens `movieId` values to TMDB IDs through `links.csv`, requests poster metadata from TMDB, and caches poster paths in memory. If the token is unset, a movie has no TMDB poster, or TMDB is unavailable, the styled poster fallback remains visible.

TMDB attribution is shown in the app footer, including the required notice: “This product uses the TMDB API but is not endorsed or certified by TMDB.” Poster paths are served from TMDB’s image CDN according to its [image guidelines](https://developer.themoviedb.org/docs/image-basics). See TMDB’s [authentication documentation](https://developer.themoviedb.org/docs/authentication-application) for API token details.

VI. UI Development: The web app is based on wire-framing and high-fidelity prototype iterations. The notebook remains available for data exploration, model evaluation, and experimentation.

VII. Usability Testing and Application Iteration: A usability test was conducted to gather user feedback, identify bugs, and understand user interaction. Results were analyzed and the application was updated accordingly.

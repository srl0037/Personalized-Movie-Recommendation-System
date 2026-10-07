##### PERSONALIZED MOVIE RECOMMENDATION SYSTEM #####

I. Overview: This is a machine learning-based movie recommendation application which tailors suggestions based on user preferences and behavior. 

II. User need assessment: Real users were surveyed regarding their experiences watching movies and interacting with movie suggestion avenues. Participants were asked questions about 
    - user behavior
    - movie preferences
    - rating and viewing history
    - user needs

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

V. UI Development: Once the model was trained, a UI was created. This UI was developed through wire-framing and high-fidelity prototype iterations. Once created, the AI model was integrated with the front-end to provide the real-time movie recommendation system.

VI. Usability Testing and Application Iteration: A usability test was conducted to gather user feedback, identify bugs, and understand user interaction. Results were analyzed and the application was updated accordingly. 


# NBA Game Classification

This project focuses on developing an LGBM model to predict the outcomes of NBA games. A Streamlit application has been built to visualize the model's performance and provide predictions for recent games.

You can access the application here: [NBA Game Classification App](https://nba-game-classification-898760610238.europe-west3.run.app/)

## Project Overview

### Data Sources & Storage
- The data is sourced from the official NBA API.
- After preprocessing, the data is stored in a PostgreSQL database hosted on Supabase.
- Originally, a cloud function was intended to update the data daily, but due to NBA API restrictions on cloud provider requests, the data may not always be up-to-date.

### Data Preprocessing
For each team, the following features are calculated:
- **Season Statistics**: Metrics such as points per game, true shooting percentage, and win percentage up to each game.
- **Recent Performance**: Averages over the last 8 games to capture short-term trends.
- **ELO Rating**: A dynamic rating reflecting team strength heading into each game.

### Feature Selection & Model Training
1. **Feature Selection**: Forward feature selection using Shapley values is applied to identify the most relevant predictive features.
2. **Hyperparameter Tuning**: The model is optimized using the best-performing parameters.
3. **Model Training**: The final LGBM model is trained on the selected features and stored in Google Cloud Storage.
4. **Monthly Updates**: A new model is trained every month to ensure the predictions remain accurate and relevant.

## Streamlit Application
The Streamlit app provides an interactive interface to:
- Visualize model performance.
- Display predictions for recent NBA games.

## Limitations
- Due to NBA API restrictions on cloud-based requests, data updates are not automated and may occasionally be outdated.

## Future Improvements
- Explore alternative data update mechanisms.
- Experiment with additional features and model architectures to enhance prediction accuracy.

This project is an ongoing effort to refine NBA game predictions using machine learning techniques. Contributions and feedback are welcome!


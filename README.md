# NYC-Taxi-Fare-Prediction---Capstone-Project
Kaggle competition entry and Capstone project for Udacity Machine Learning Engineer Nanodegree

## Kaggle Competition - New York City Taxi Fare Prediction
Competition:
[https://www.kaggle.com/c/new-york-city-taxi-fare-prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)

## Dataset available for download directly from Kaggle or used within a Kernel on their site
[https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data)
Alternatively, you can access the source of this dataset as a public dataset on Google Big Query:
[https://bigquery.cloud.google.com/table/nyc-tlc:yellow.trips](https://bigquery.cloud.google.com/table/nyc-tlc:yellow.trips?tab=details)

You can access the capstone project rubric here:
[https://review.udacity.com/#!/rubrics/108/view](https://review.udacity.com/#!/rubrics/108/view)

The data files are not included here on GitHub due to size limitations, but they can be downloaded from Kaggle then transformed using my "Data Exploration and File Prep.ipnyb" notebook.

All the bash commands to create, train, deploy and predict with the model are bundled up in "Train, Deploy, Predict with MLengine.ipynb", including the code needed to create baseline predictions.

Performance of this model is judged by RMSE scores returned by making a submission to the Kaggle competition.

Many thanks to [Google](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive/03_tensorflow) and [Albert Van Breemen](https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration) for the code they've made freely available, which I've used as a starting point for this project and the competition.

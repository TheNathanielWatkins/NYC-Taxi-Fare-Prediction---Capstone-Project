# Content: Capstone Project
Udacity - Machine Learning Engineer Nanodegree | Nathaniel Watkins
## Project: NYC-Taxi-Fare-Prediction
Kaggle competition entry and Capstone project for Udacity Machine Learning Engineer Nanodegree

### Final Report
* To see my work and results, open: 
[Capstone Project Report.pdf](https://github.com/TheNathanielWatkins/NYC-Taxi-Fare-Prediction---Capstone-Project/blob/master/Capstone%20Project%20Report.pdf)

### Capstone Project Rubric
* [https://review.udacity.com/#!/rubrics/108/view](https://review.udacity.com/#!/rubrics/108/view)

-----

### Kaggle Competition - New York City Taxi Fare Prediction
* Competition:
  * [https://www.kaggle.com/c/new-york-city-taxi-fare-prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)

* Results:
  * Placed within top third on leaderboard with 3.26 RMSE (less than 2 points away from the top score)

### Dataset available for download directly from Kaggle or used within a Kernel on their site
[https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data)

Alternatively, you can access the source of this dataset as a public dataset on Google Big Query:
[https://bigquery.cloud.google.com/table/nyc-tlc:yellow.trips](https://bigquery.cloud.google.com/table/nyc-tlc:yellow.trips?tab=details)


## Key Files

* The data files are not included here on GitHub due to size limitations, but they can be downloaded from Kaggle then transformed using my "[Data Exploration and File Prep.ipnyb](https://github.com/TheNathanielWatkins/NYC-Taxi-Fare-Prediction---Capstone-Project/blob/master/Train%2C%20Deploy%2C%20Predict%20with%20MLengine.ipynb)" notebook.

* All the bash commands to create, train, deploy and predict with the model are bundled up in "[Train, Deploy, Predict with MLengine.ipynb](https://github.com/TheNathanielWatkins/NYC-Taxi-Fare-Prediction---Capstone-Project/blob/master/Train%2C%20Deploy%2C%20Predict%20with%20MLengine.ipynb)", including the code needed to create baseline predictions.

* Another important file is contained in the taxifare/v2-trainer folder: "[model.py](https://github.com/TheNathanielWatkins/NYC-Taxi-Fare-Prediction---Capstone-Project/blob/master/taxifare/v2-trainer/model.py)" which contains all the code for creating and training the TensorFlow model including the feature columns.

-----

Many thanks to [Google](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive/03_tensorflow) and [Albert Van Breemen](https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration) for the code they've made freely available, which I've used as a starting point for this project and the competition.

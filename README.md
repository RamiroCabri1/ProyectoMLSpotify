Spotify Song Classifier
Introduction
In this project, we focus on a dataset filled with key attributes of various songs. The goal is to establish a classifier using machine learning tools that can predict an individual's musical preferences and determine whether they will enjoy a particular song.

Dataset
A CSV file is provided containing a dataset with 16 columns. Of these, 13 represent specific attributes of the songs. Additionally, there are columns dedicated to the song name and artist. The "target" column acts as a label, indicating user preferences: a value of "1" means the user liked the song, while a "0" indicates otherwise. The attributes characterizing each track include:

Danceability: How suitable a track is for dancing based on tempo, rhythm stability, and beat strength.

Energy: A perceptual measure of intensity and activity.

Loudness: General measure of the track's loudness in decibels (dB).

Speechiness: Detects the presence of spoken words in a track.

Acousticness: Measures how acoustic a track is.

Instrumentalness: Predicts whether a track contains no vocals.

Liveness: Detects the presence of an audience in the recording.

Valence: Describes the musical positiveness conveyed by a track.

Tempo: The overall speed or pace of a track, measured in beats per minute (BPM).

Report
The main task is to develop a program that categorizes songs according to user preferences. This project is executed within a Colab Notebook. Each step of the process is accompanied by text blocks explaining the reasoning behind the decisions made. It is important to detail why certain techniques, methods, or parameters were chosen and their relevance to the project. Additionally, documenting challenges or issues encountered during development provides a clear and comprehensive view of the design and implementation process.

Workflow
Feature Selection: Choose the optimal features for training the models.

Data Splitting: Separate into training and testing data.

Model Training: Implement various machine learning models:

KNN (K-Nearest Neighbors)

SVM (Support Vector Machines)

Decision Tree

Naive Bayes

Another model of your choice

Validation: Perform validation techniques:

Simple Validation

k-fold Cross-Validation

Performance Evaluation: Analyze the performance of each model:

Confusion Matrix

Precision, Recall, and F1-score

Hyperparameter Tuning: Adjust hyperparameters for each model:

Grid Search

Random Search

Model Ensemble: Create an ensemble of models:

Majority Voting

Final Evaluation: Evaluate and analyze the performance:

Confusion Matrix

Precision, Recall, and F1-score

Libraries and Dataset
python

Copiar
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
import glob
import plotly.graph_objects as go
import os
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import VotingClassifier
Technologies Used
Google Colab: For executing the project in a notebook environment.

Pandas: For data manipulation and analysis.

NumPy: For numerical computations.

Plotly: For interactive visualizations.

Matplotlib: For static visualizations.

Seaborn: For statistical data visualization.

Scikit-learn: For implementing machine learning models and evaluation metrics.

Graphviz: For visualizing decision trees.



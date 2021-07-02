"""

    Simple Flask-based API for Serving an sklearn Model.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file instantiates a Flask webserver
    as a means to create a simple API used to deploy models trained within
    the sklearn framework.

"""

# API Dependencies
import pickle
import json
import numpy as np
from model import load_model, make_prediction
from flask import Flask, request, jsonify
# Libraries for data loading, data manipulation and data visulisation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pickle

# Libraries for data preparation and model building
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb


#Ensemble
from sklearn.ensemble import VotingRegressor #Hetero
from sklearn.ensemble import StackingRegressor #Hetero
from sklearn.ensemble import BaggingRegressor #Homo
from sklearn.ensemble import AdaBoostRegressor #Homo
from sklearn.ensemble import ExtraTreesRegressor #Homo
import xgboost as xgb


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



# Setting global constants to ensure notebook results are reproducible
PARAMETER_CONSTANT = 42


# Application definition
app = Flask(__name__)

# Load our model into memory.
# Please update this path to reflect your own trained model.
static_model = load_model(
    path_to_model='assets/trained-models/best_reg_model.pkl')

print ('-'*40)
print ('Model successfully loaded')
print ('-'*40)

""" You may use this section (above the app routing function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""


# Define the API's interface.
# Here the 'model_prediction()' function will be called when a POST request
# is sent to our interface located at:
# http:{Host-machine-ip-address}:5000/api_v0.1
@app.route('/api_v0.1', methods=['POST'])
def model_prediction():
    # We retrieve the data payload of the POST request
    data = request.get_json(force=True)
    # We then preprocess our data, and use our pretrained model to make a
    # prediction.
    output = make_prediction(data, static_model)
    # We finally package this prediction as a JSON object to deliver a valid
    # response with our API.
    return jsonify(output)

# Configure Server Startup properties.
# Note:
# When developing your API, set `debug=True`
# This will allow Flask to automatically restart itself everytime you
# update your API code.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

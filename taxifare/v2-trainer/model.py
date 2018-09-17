#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import shutil
import pandas as pd
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

# List the CSV columns
CSV_COLUMNS = ['key', 'fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 
               'hour', 'day_of_week', 'day_of_month', 'week', 'month', 'year', 'distance_km']

#Choose which column is your label
LABEL_COLUMN = 'fare_amount'

# Set the default values for each CSV column in case there is a missing value
default_values = [['nokey'], [2.5], [-74.0], [40.0], [-74.0], [40.0], [1], [14], [3], [16], [25], [6], [11], [3.4]]
    
# Create an input function that stores your data into a dataset
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults = default_values)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return features, label
    
        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list, and skips the first/header line
        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 100 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn

# Define your feature columns
INPUT_COLUMNS = []

for col in CSV_COLUMNS[2:]:
    INPUT_COLUMNS.append(tf.feature_column.numeric_column(col))

# Create a function that will augment your feature set
def add_more_features(feats):
    pickup_longitude = tf.feature_column.bucketized_column(
        source_column = feats[0],
        boundaries = list(np.linspace(-74.5, -72.8, 170)))
    pickup_latitude = tf.feature_column.bucketized_column(
        source_column = feats[1],
        boundaries = list(np.linspace(40.5, 41.8, 130)))
    dropoff_longitude = tf.feature_column.bucketized_column(
        source_column = feats[2],
        boundaries = list(np.linspace(-74.5, -72.8, 170)))
    dropoff_latitude = tf.feature_column.bucketized_column(
        source_column = feats[3],
        boundaries = list(np.linspace(40.5, 41.8, 130)))
    passenger_count = tf.feature_column.bucketized_column(
        source_column = feats[4],
        boundaries = list(range(2,7)))
    hour = tf.feature_column.bucketized_column(
        source_column = feats[5],
        boundaries = list(range(1,24)))
    day_of_week = tf.feature_column.bucketized_column(
        source_column = feats[6],
        boundaries = list(range(1,7)))
    day_of_month = tf.feature_column.bucketized_column(
        source_column = feats[7],
        boundaries = list(range(2,32)))
    week = tf.feature_column.bucketized_column(
        source_column = feats[8],
        boundaries = list(range(2,53)))
    month = tf.feature_column.bucketized_column(
        source_column = feats[9],
        boundaries = list(range(2,13)))
    year = tf.feature_column.bucketized_column(
        source_column = feats[10],
        boundaries = list(range(9,16)))
    distance = tf.feature_column.bucketized_column(
        source_column = feats[11],
        boundaries = list(range(1, 32, 2)))
    
    crossed_pickup = tf.feature_column.crossed_column([pickup_longitude, pickup_latitude], 29399)
    """
    This hash_bucket_size chosen because there are 22100 buckets:
    Multiplied by a load factor of 1.33 is 29393 and the next prime number is 29399
    """
    crossed_dropoff =  tf.feature_column.crossed_column([dropoff_longitude, dropoff_latitude], 29399)
    
    crossed_time = tf.feature_column.crossed_column([hour, day_of_week], 223)
    
    crossed_day = tf.feature_column.crossed_column([day_of_month, day_of_week], 293)
    
    crossed_month = tf.feature_column.crossed_column([month, year], 111)
    
    embedding_feats = [crossed_pickup, crossed_dropoff, crossed_time, crossed_day, crossed_month]
    dimensions = [13, 13, 4, 4, 3]
    
    bucketized_feats = [passenger_count, hour, day_of_week, day_of_month, week, month, year,
                        pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude]
    
    deep_columns = bucketized_feats

    for feat, dimension in zip(embedding_feats, dimensions):
        deep_columns.append(tf.feature_column.embedding_column(
            categorical_column=feat, dimension=dimension))

    deep_columns.append(distance)

    wide_columns = bucketized_feats + embedding_feats

    wide_columns.append(feats[11])
 
    return wide_columns, deep_columns

wide_columns, deep_columns = add_more_features(INPUT_COLUMNS)

# Create your serving input function so that your trained model will be able to serve predictions
def serving_input_fn():
    feature_placeholders = {
        column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
    }

    features = feature_placeholders
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

## The below may not be needed if including Key in the feature list, but not the feature columns ends up working

KEY = 'key'
def key_model_fn_gen(estimator):
    def _model_fn(features, labels, mode):
        key = features.pop(KEY)
        params = estimator.params
        model_fn_ops = estimator._model_fn(features=features, labels=labels, mode=mode, params=params)
        model_fn_ops.predictions[KEY] = key
        model_fn_ops.output_alternatives[None][1][KEY] = key
        print(model_fn_ops.output_alternatives)
        return model_fn_ops
    return _model_fn

## May need to use the below if key not passing through
# """
# my_key_estimator = tf.contrib.learn.Estimator(
#     model_fn=key_model_fn_gen(
#         tf.contrib.learn.DNNClassifier(model_dir=model_dir...)
#     ),
#     model_dir=model_dir
# )

# ## From: https://stackoverflow.com/questions/44381879/training-and-predicting-with-instance-keys
# """
                     
# ## Or if not working, try this
# """
# def forward_key_to_export(estimator):
#     estimator = tf.contrib.estimator.forward_features(estimator, KEY_COLUMN)
#     # return estimator
# ## This shouldn't be necessary (I've filed CL/187793590 to update extenders.py with this code)
#     config = estimator.config
#     def model_fn2(features, labels, mode):
#       estimatorSpec = estimator._call_model_fn(features, labels, mode, config=config)
#       if estimatorSpec.export_outputs:
#         for ekey in ['predict', 'serving_default']:
#           estimatorSpec.export_outputs[ekey] = \
#             tf.estimator.export.PredictOutput(estimatorSpec.predictions)
#       return estimatorSpec
#     return tf.estimator.Estimator(model_fn=model_fn2, config=config)
#     ##

# ## From: https://towardsdatascience.com/how-to-extend-a-canned-tensorflow-estimator-to-add-more-evaluation-metrics-and-to-pass-through-ddf66cd3047d
# """

## Also, this may help: https://github.com/GoogleCloudPlatform/cloudml-samples/issues/67

# Create an estimator that we are going to train and evaluate
def train_and_evaluate(args):
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir=args['output_dir'],
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=args['hidden_units'])
    
#     tf.estimator.DNNRegressor(
#         model_dir = args['output_dir'],
#         feature_columns = feature_cols,
#         hidden_units = args['hidden_units'])

    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(args['train_data_paths'],
                                batch_size = args['train_batch_size'],
                                mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = args['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(args['eval_data_paths'],
                                batch_size = 10000,
                                mode = tf.estimator.ModeKeys.EVAL),
        steps = None,
        start_delay_secs = args['eval_delay_secs'],
        throttle_secs = args['min_eval_frequency'],
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
# Import Libraries
import sys
import logging
import pandas as pd
    
# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

def standardise_features(df, cols):
    pass

def one_hot_encode_features(df, categorical_cols):
    pass

# def round_features(df, cols, decimals=4):
# def compute_class_weights(df, target_col):
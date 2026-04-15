import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess(file_path='data/raw/asteroid_data.csv'):
    """
    Loads raw NASA asteroid data, cleans redundant columns, 
    and splits into training and testing sets.
    """
    df = pd.read_csv(file_path)
    
    # 1. DROP IDENTIFIERS AND REDUNDANT INFO
    # These columns don't represent physical characteristics
    to_drop = [
        'Neo Reference ID', 'Name', 'Orbit identification', 'Close Approach Date',
        'Epoch Date Close Approach', 'Orbit Determination Date', 'Orbiting Body', 
        'Equinox', 'Orbit ID'
    ]
    
    # 2. DROP REDUNDANT UNITS
    # Keep only Kilometers and km/sec to keep it simple
    redundant_units = [
        'Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)',
        'Est Dia in Miles(min)', 'Est Dia in Miles(max)',
        'Est Dia in Feet(min)', 'Est Dia in Feet(max)',
        'Relative Velocity km per hr', 'Miles per hour',
        'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)', 'Miss Dist.(miles)'
    ]
    
    all_drop = to_drop + redundant_units
    df_clean = df.drop(columns=[c for c in all_drop if c in df.columns])
    
    # 3. HANDLE MISSING VALUES
    # (NASA data is usually clean, but let's be safe)
    df_clean = df_clean.fillna(df_clean.mean())

    # 4. SPLIT INTO FEATURES (X) AND TARGET (y)
    X = df_clean.drop('Hazardous', axis=1)
    y = df_clean['Hazardous']

    # 5. TRAIN-TEST SPLIT
    # Stratify ensuring the distribution of 'Hazardous' is the same in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

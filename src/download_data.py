import pandas as pd
import requests

def download_sample_data():
    """
    Downloads a sample dataset of Near Earth Objects (NEO) from a verified source.
    Source: NASA Open Data via ShubhankarRawat's GitHub.
    """
    url = "https://raw.githubusercontent.com/ShubhankarRawat/NASA-asteroid-Classification/master/nasa.csv"
    destination = "data/raw/asteroid_data.csv"
    
    print(f"Downloading dataset from {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        
        with open(destination, 'wb') as f:
            f.write(response.content)
            
        print(f"Successfully downloaded to: {destination}")
        print("You can now load this in your notebooks using: pd.read_csv('data/raw/asteroid_data.csv')")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Alternative: Download 'neo.csv' from Kaggle: https://www.kaggle.com/datasets/saurabhshahane/neo-guide")

if __name__ == "__main__":
    download_sample_data()

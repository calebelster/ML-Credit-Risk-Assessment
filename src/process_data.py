import pandas as pd

DATA_PATH = "../data/credit_risk_dataset.csv"

def read_data(file_path):
    """Reads a CSV file and returns a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")

def preprocess_data(df):
    """Preprocess the DataFrame by handling missing values and encoding categorical variables."""
    # Example preprocessing steps
    df = df.dropna()  # Drop rows with missing values
    return df

def main():
    # Read the data
    df = read_data(DATA_PATH)
    print("Data read successfully.")

    # Preprocess the data
    df = preprocess_data(df)
    print("Data preprocessed successfully.")

    # Display the first few rows of the processed DataFrame
    print(df.head())

if __name__ == "__main__":
    main()

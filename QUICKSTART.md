# Quickstart Guide --- ML Credit Risk Analysis Project

This guide walks you through running the full machine-learning pipeline
and launching the Streamlit app.

## 1. Requirements

Make sure you have the following installed:

-   **Python 3.10+**

-   **uv** (fast Python package & environment manager)\
    Install with:

    ``` bash
    pip install uv
    ```

-   **Streamlit**

## 2. Install Dependencies

From the project root:

``` bash
uv sync
```

This creates a virtual environment and installs all project
dependencies.

Activate the environment:

``` bash
uv venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

## 3. Train All Models

Still in the project root, run:

``` bash
uv run main.py
```

`main.py` will:

-   Load and preprocess the dataset
-   Train all ML models
-   Save the trained models + metrics to the appropriate folders

## 4. Start the Streamlit App

Navigate into the Streamlit app directory:

``` bash
cd app
```

Run the app:

``` bash
streamlit run Home.py
```

This launches the interactive dashboard where you can:

-   Explore model performance
-   Run predictions
-   Visualize data and feature importance

## 5. That's It!

Once the app is running, open the URL printed in your
terminal---usually:

    http://localhost:8501

You're ready to explore credit-risk predictions.

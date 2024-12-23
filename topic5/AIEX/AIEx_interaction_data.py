import pandas as pd


def interaction_data(file_path):
    """
    Reads user-movie interaction data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing interaction data.

    Returns:
        pd.DataFrame: DataFrame containing the interaction data.
    """
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Verify required columns are present
        required_columns = {'user_id', 'movie_id', 'rating', 'timestamp'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"CSV file must contain the following columns: {required_columns}")

        return data
    except Exception as e:
        print(f"Error reading interaction data: {e}")
        return None

import pandas as pd


def movies_data(file_path):
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
        required_columns = {'movie_id', 'title', 'genre', 'director','cast'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"CSV file must contain the following columns: {required_columns}")

        return data
    except Exception as e:
        print(f"Error reading movies data: {e}")
        return None

from pathlib import Path
import pandas as pd


def load_dataset(file_path : str ) -> pd.DataFrame:
    """
    Loads a dataset from a CSV or Excel file.
    Parameters
    ----------
    file_path : str

    Returns
    -------
    pd.DataFrame
    """
    file_extension = Path(file_path).suffixes[-1].lower()
    #
    SUPPORTED_EXCEL_EXTENSIONS = [".xls", ".xlsx", ".xlsm", ".xlsb"]

    try:
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension in SUPPORTED_EXCEL_EXTENSIONS:
            df = pd.read_excel(file_path)
        else:
            print("Unsupported file format. Please provide a CSV or Excel file.")
            return pd.DataFrame()  # Return an empty DataFrame
        return df
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
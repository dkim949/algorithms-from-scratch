import numpy as np
import pandas as pd


class SparseMatrix:
    def __init__(self, data, row_indices, col_indices, shape):
        self.data = data
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.shape = shape


class DataPreprocessing:
    def __init__(self):
        self.categorical_maps = {}

    def scale_features(self, data, columns):
        pass

    def _create_sparse_matrix(self, data, column, unique_values, current_col):
        """
        Create a sparse matrix representation(COO) for a single column.

        Args:
        data (pd.DataFrame): Input data
        column (str): Column name to encode
        unique_values (np.array): Unique values in the column
        current_col (int): Starting column index for this encoding

        Returns:
        tuple: (data, row_indices, col_indices, new_current_col)
        """
        all_data = []
        all_row_indices = []
        all_col_indices = []

        for i, value in enumerate(unique_values):
            indices = np.where(data[column] == value)[0]
            all_data.extend([1] * len(indices))
            all_row_indices.extend(indices)
            all_col_indices.extend([current_col + i] * len(indices))

        new_current_col = current_col + len(unique_values)
        return all_data, all_row_indices, all_col_indices, new_current_col

    def onehot_encode(self, data, columns, sparse=False):
        """
        Perform one-hot encoding on categorical variables.

        Args:
        data (pd.DataFrame): Input data
        columns (list): List of column names to encode
        sparse (bool): If True, return a sparse matrix, else return a DataFrame

        Returns:
        pd.DataFrame or SparseMatrix: Data with one-hot encoded categories
        """
        if not sparse:
            encoded = data.copy()

        all_data = []
        all_row_indices = []
        all_col_indices = []
        current_col = 0

        for column in columns:
            unique_values = data[column].unique()
            self.categorical_maps[column] = {
                val: i for i, val in enumerate(unique_values)
            }

            if sparse:
                col_data, col_row_indices, col_col_indices, current_col = (
                    self._create_sparse_matrix(data, column, unique_values, current_col)
                )
                all_data.extend(col_data)
                all_row_indices.extend(col_row_indices)
                all_col_indices.extend(col_col_indices)
            else:
                for value in unique_values:
                    encoded[f"{column}_{value}"] = (data[column] == value).astype(int)
                encoded.drop(column, axis=1, inplace=True)

        if sparse:
            return SparseMatrix(
                np.array(all_data),
                np.array(all_row_indices),
                np.array(all_col_indices),
                (len(data), current_col),
            )
        else:
            return encoded

    def normalize(self, data, columns):
        pass

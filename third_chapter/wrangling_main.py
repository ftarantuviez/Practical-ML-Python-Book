import numpy as np
import pandas as pd


class WranglingData:
    def __init__(self, df):
        self.df = df

    def understand(self):
        """
        This method returns you the basic to understand the dataframe and their features
        """
        print("Number of rows::", df.shape[0])
        print("Number of columns::", df.shape[1])
        print("Column Names::", df.columns.values.tolist())
        print("Column Data Types::\n", df.dtypes)

        print("==" * 60)

        print("Columns with missing values::",
              df.columns[df.isnull().any()].tolist())
        print(
            "Number of rows with missing values::",
            len(pd.isnull(df).any(
                1).to_numpy().nonzero()[0].tolist()),
        )
        print(
            "Sample Indices with missing data::", pd.isnull(
                df).any().to_numpy().nonzero()[0].tolist()[0:5],
        )

        print("==" * 60)

        print("General Stats::")
        print(df.info())
        print("Summary Stats::")
        print(df.describe())

    def cleanup_column_names(self, df, rename_dict={}, do_inplace=True):
        """
        This function renames columns of a pandas dataframe
        It converts column names to snake case if rename_dict is not passed.
        Args:
            rename_dict (dict): keys represent old column names and values point to newer ones.
            do_inplace (bool): flag to update existing dataframe or return a new one.

        Returns:
            pandas dataframe if do_inplace is set to False, None otherwise
        """
        if not rename_dict:
            return df.rename(columns={col: col.lower().replace(
                " ", "_") for col in df.columns.values.tolist()}, inplace=do_inplace)
        else:
            return df.rename(columns=rename_dict, inplace=do_inplace)


if __name__ == "__main__":
    df = pd.read_csv("third_chapter/dataset.csv")
    data = WranglingData(df)

    data.understand()

    df = data.cleanup_column_names(df)

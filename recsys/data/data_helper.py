import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataHelper:
    """Data preprocessing. Process raw data when `${base_dir} / processed` doesn't exist."""

    def __init__(self, base_dir: str = "./dressipi_recsys2022/"):
        """Initialize DataHelper.

        Defalut path:
            - raw data dir: `${base_dir} / raw`
            - processed file dir: `${base_dir} / processed`

        Parameters
        ----------
        base_dir : str, data directory.

        """
        self.base_dir = base_dir
        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.processed_dir = os.path.join(self.base_dir, "processed")

        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            print("Processing raw data...")
            self._process_raw_data()

    def _load_raw_data(self) -> dict:
        """Load original raw data.

        Returns
        -------
        dict
            Data dictionary
            keys: 'train_sessions', 'train_purchases', 'item_features',
                  'candidate_items', 'lb_sessions', 'final_sessions'

        """
        df_sessions = pd.read_csv(os.path.join(self.raw_dir, "train_sessions.csv"))
        df_purchases = pd.read_csv(os.path.join(self.raw_dir, "train_purchases.csv"))
        df_item_feat = pd.read_csv(os.path.join(self.raw_dir, "item_features.csv"))
        df_cand_items = pd.read_csv(os.path.join(self.raw_dir, "candidate_items.csv"))
        df_lb = pd.read_csv(os.path.join(self.raw_dir, "test_leaderboard_sessions.csv"))
        df_final = pd.read_csv(os.path.join(self.raw_dir, "test_final_sessions.csv"))

        return {
            "train_sessions": df_sessions,
            "train_purchases": df_purchases,
            "item_features": df_item_feat,
            "candidate_items": df_cand_items,
            "lb_sessions": df_lb,
            "final_sessions": df_final,
        }

    def _encode_id(self, data: dict, map_dir: str = "index_id_map") -> dict:
        """Encode item id and category feature id as consecutive integers.

        Parameters
        ----------
        data:
            Data dictionary
            keys: 'train_sessions', 'train_purchases', 'item_features',
                  'candidate_items', 'lb_sessions', 'final_sessions'

        Returns
        -------
        dict
            Data dictionary
            keys: 'train_sessions', 'train_purchases', 'item_features',
                  'candidate_items', 'lb_sessions', 'final_sessions'

        """
        map_dir = os.path.join(self.base_dir, map_dir)
        if not os.path.isdir(map_dir):
            os.mkdir(map_dir)

        df_sessions = data["train_sessions"]
        df_purchases = data["train_purchases"]
        df_item_feat = data["item_features"]
        df_cand_items = data["candidate_items"]
        df_lb = data["lb_sessions"]
        df_final = data["final_sessions"]

        item2code_path = os.path.join(map_dir, "item2code.pkl")
        code2item_path = os.path.join(map_dir, "code2item.pkl")

        cat2code_path = os.path.join(map_dir, "cat2code.pkl")
        code2cat_path = os.path.join(map_dir, "code2cat.pkl")

        val2code_path = os.path.join(map_dir, "val2code.pkl")
        code2val_path = os.path.join(map_dir, "code2val.pkl")
        # item_id
        if not (os.path.exists(item2code_path) and os.path.exists(code2item_path)):
            df_item_feat["encoded_item_id"] = LabelEncoder().fit_transform(
                df_item_feat["item_id"]
            )
            item2code = (
                df_item_feat.loc[:, ["item_id", "encoded_item_id"]]
                .set_index("item_id")
                .to_dict()["encoded_item_id"]
            )
            code2item = (
                df_item_feat.loc[:, ["item_id", "encoded_item_id"]]
                .set_index("encoded_item_id")
                .to_dict()["item_id"]
            )
            pickle.dump(item2code, open(item2code_path, "wb"))
            pickle.dump(code2item, open(code2item_path, "wb"))

        else:
            item2code = pickle.load(open(item2code_path, "rb"))
            # code2item = pickle.load(open(code2item_path, "rb"))

        # feature_category_id
        if not (os.path.exists(cat2code_path) and os.path.exists(code2cat_path)):
            df_item_feat["encoded_feature_category_id"] = LabelEncoder().fit_transform(
                df_item_feat["feature_category_id"]
            )
            cat2code = (
                df_item_feat.loc[
                    :, ["feature_category_id", "encoded_feature_category_id"]
                ]
                .set_index("feature_category_id")
                .to_dict()["encoded_feature_category_id"]
            )
            code2cat = (
                df_item_feat.loc[
                    :, ["feature_category_id", "encoded_feature_category_id"]
                ]
                .set_index("encoded_feature_category_id")
                .to_dict()["feature_category_id"]
            )
            pickle.dump(cat2code, open(cat2code_path, "wb"))
            pickle.dump(code2cat, open(code2cat_path, "wb"))

        else:
            cat2code = pickle.load(open(cat2code_path, "rb"))
            # code2cat = pickle.load(open(code2cat_path, "rb"))

        # feature_value_id
        if not (os.path.exists(val2code_path) and os.path.exists(code2val_path)):

            val2code = dict(
                zip(
                    df_item_feat["feature_value_id"].unique(),
                    np.arange(df_item_feat["feature_value_id"].nunique()),
                )
            )
            code2val = dict(
                zip(
                    np.arange(df_item_feat["feature_value_id"].nunique()),
                    df_item_feat["feature_value_id"].unique(),
                )
            )
            pickle.dump(val2code, open(val2code_path, "wb"))
            pickle.dump(code2val, open(code2val_path, "wb"))

        else:
            val2code = pickle.load(open(val2code_path, "rb"))
            # code2_val = pickle.load(open(code2val_path, "rb"))

        # transform item_id
        df_sessions["item_id"] = df_sessions["item_id"].map(item2code)
        df_purchases["item_id"] = df_purchases["item_id"].map(item2code)
        df_item_feat["item_id"] = df_item_feat["item_id"].map(item2code)
        df_cand_items["item_id"] = df_cand_items["item_id"].map(item2code)
        df_lb["item_id"] = df_lb["item_id"].map(item2code)
        df_final["item_id"] = df_final["item_id"].map(item2code)
        # transform feature_category_id
        df_item_feat["feature_category_id"] = df_item_feat["feature_category_id"].map(
            cat2code
        )
        # transform feature_value_id
        df_item_feat["feature_value_id"] = df_item_feat["feature_value_id"].map(
            val2code
        )

        return {
            "train_sessions": df_sessions,
            "train_purchases": df_purchases,
            "item_features": df_item_feat,
            "candidate_items": df_cand_items,
            "lb_sessions": df_lb,
            "final_sessions": df_final,
        }

    def _gen_base_features(self, data: dict) -> dict:
        """Extract datetime features from data dictionary.

        Parameters
        ----------
        data:
            Data dictionary
            keys: 'train_sessions', 'train_purchases', 'item_features',
                  'candidate_items', 'lb_sessions', 'final_sessions'

        Returns
        -------
        dict
            Data dictionary
            keys: 'train_sessions', 'train_purchases', 'item_features',
                  'candidate_items', 'lb_sessions', 'final_sessions'
        """

        df_sessions = data["train_sessions"]
        df_purchases = data["train_purchases"]
        df_lb = data["lb_sessions"]
        df_final = data["final_sessions"]

        df_sessions["date"] = pd.to_datetime(df_sessions["date"])
        # df_sessions['year'] = df_sessions['date'].apply(lambda x: x.year)
        # df_sessions['month'] = df_sessions['date'].apply(lambda x: x.month)
        # df_sessions['day'] = df_sessions['date'].apply(lambda x: x.day)
        # df_sessions['weekday'] = df_sessions['date'].apply(lambda x: x.weekday())

        df_purchases["date"] = pd.to_datetime(df_purchases["date"])
        df_lb["date"] = pd.to_datetime(df_lb["date"])
        df_final["date"] = pd.to_datetime(df_final["date"])

        data["train_sessions"] = df_sessions
        data["train_purchases"] = df_purchases
        data["lb_sessions"] = df_lb
        data["final_sessions"] = df_final

        return data

    def _gen_label(self, data: dict):
        """Set purchased item id as `target_item_id` for each session.

        Parameters
        ----------
        data:
            Data dictionary
            keys: 'train_sessions', 'train_purchases', 'item_features',
                  'candidate_items', 'lb_sessions', 'final_sessions'

        Returns
        -------
        dict
            Data dictionary
            keys: 'train_sessions', 'train_purchases', 'item_features',
                  'candidate_items', 'lb_sessions', 'final_sessions'
        """
        df_sessions = data["train_sessions"]
        df_purchases = data["train_purchases"]

        df_sessions["target_item_id"] = df_sessions.session_id.map(
            df_purchases[["session_id", "item_id"]]
            .set_index("session_id")
            .to_dict()["item_id"]
        )
        data["train_sessions"] = df_sessions

        return data

    def _trans_dtype(self, data: dict, verbose: bool = True) -> dict:
        """Change data types.

        Parameters
        ----------
        data:
            Data dictionary
            keys: 'train_sessions', 'train_purchases', 'item_features',
                  'candidate_items', 'lb_sessions', 'final_sessions'

        Returns
        -------
        dict
            Data dictionary
            keys: 'train_sessions', 'train_purchases', 'item_features',
                  'candidate_items', 'lb_sessions', 'final_sessions'

        """
        numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
        for k, df in data.items():
            start_mem = df.memory_usage().sum() / 1024**2
            for col in df.columns:
                col_type = df[col].dtypes
                if col_type in numerics:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if str(col_type)[:3] == "int":
                        if (
                            c_min > np.iinfo(np.int8).min
                            and c_max < np.iinfo(np.int8).max
                        ):
                            df[col] = df[col].astype(np.int8)
                        elif (
                            c_min > np.iinfo(np.int16).min
                            and c_max < np.iinfo(np.int16).max
                        ):
                            df[col] = df[col].astype(np.int16)
                        elif (
                            c_min > np.iinfo(np.int32).min
                            and c_max < np.iinfo(np.int32).max
                        ):
                            df[col] = df[col].astype(np.int32)
                        elif (
                            c_min > np.iinfo(np.int64).min
                            and c_max < np.iinfo(np.int64).max
                        ):
                            df[col] = df[col].astype(np.int64)
                    else:
                        c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                        if (
                            c_min > np.finfo(np.float16).min
                            and c_max < np.finfo(np.float16).max
                            and c_prec == np.finfo(np.float16).precision
                        ):
                            df[col] = df[col].astype(np.float16)
                        elif (
                            c_min > np.finfo(np.float32).min
                            and c_max < np.finfo(np.float32).max
                            and c_prec == np.finfo(np.float32).precision
                        ):
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)
            end_mem = df.memory_usage().sum() / 1024**2
            if verbose:
                print(f"{k}:")
                print(
                    "Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                        end_mem, 100 * (start_mem - end_mem) / start_mem
                    )
                )

        return data

    def _save_as_parquet(self, data: dict):
        """Save data dictionary as parquet.

        Files:
            'train_sessions.parquet'
            'train_purchases.parquet'
            'item_features.parquet'
            'candidate_items.parquet'

        Save Path: `${base_dir} / processed`

        Parameters
        ----------
        data:
            Data dictionary
            keys: 'train_sessions', 'train_purchases', 'item_features',
                  'candidate_items', 'lb_sessions', 'final_sessions'

        """
        path = self.processed_dir
        for key, df in data.items():
            df.to_parquet(os.path.join(path, key + ".parquet"))
            print(f"File {key + '.parquet'} saved.")

    def _process_raw_data(self):
        """Process raw data."""

        data = self._load_raw_data()
        data = self._encode_id(data, "index_id_map")
        data = self._gen_base_features(data)
        data = self._gen_label(data)
        data = self._trans_dtype(data)
        self._save_as_parquet(data)

    def load_data(self) -> dict:
        """Return processed data.

        Returns
        -------
        dict
            Data dictionary
            keys: 'train_sessions', 'train_purchases', 'item_features',
                  'candidate_items', 'lb_sessions', 'final_sessions'

        """
        if not os.path.exists(self.processed_dir):
            raise OSError(f"{self.processed_dir} does not exist.")

        df_sessions = pd.read_parquet(
            os.path.join(self.processed_dir, "train_sessions.parquet")
        )
        df_purchases = pd.read_parquet(
            os.path.join(self.processed_dir, "train_purchases.parquet")
        )
        df_item_feat = pd.read_parquet(
            os.path.join(self.processed_dir, "item_features.parquet")
        )
        df_cand_items = pd.read_parquet(
            os.path.join(self.processed_dir, "candidate_items.parquet")
        )
        df_lb = pd.read_parquet(os.path.join(self.processed_dir, "lb_sessions.parquet"))
        df_final = pd.read_parquet(
            os.path.join(self.processed_dir, "final_sessions.parquet")
        )

        return {
            "train_sessions": df_sessions,
            "train_purchases": df_purchases,
            "item_features": df_item_feat,
            "candidate_items": df_cand_items,
            "lb_sessions": df_lb,
            "final_sessions": df_final,
        }

    def split_data(
        self,
        trans_data: pd.DataFrame,
        train_start_date: str,
        train_end_date: str,
        valid_start_date: str,
        valid_end_date: str,
    ) -> Tuple[pd.DataFrame]:
        """Split transaction data into train set and validation set.

        Parameters
        ----------
        trans_data : pd.DataFrame
            Transaction dataframe.

        train_start_date : str
            Start date of train set, min(train_set.date) >= train_start_date.

        train_end_date : str
            End date of train set, max(train_set.date) < train_end_date.

        valid_start_date : str
            Start date of valid set, min(valid_set.date) >= valid_start_date.

        valid_end_date : str
            End date of valid set, max(valid_set.date) < valid_end_date.

        Returns
        -------
        Tuple[pd.DataFrame]
            [train set, valid set]

        """
        train_set = trans_data.loc[
            (trans_data["date"] >= train_start_date)
            & (trans_data["date"] < train_end_date)
        ]
        valid_set = trans_data.loc[
            (trans_data["date"] >= valid_start_date)
            & (trans_data["date"] < valid_end_date)
        ]

        valid_set = (
            valid_set.groupby(["session_id"])["item_id"].apply(list).reset_index()
        )

        return train_set, valid_set


if __name__ == "__main__":

    data_dir = "./dressipi_recsys2022/"
    DataHelper(base_dir=data_dir)

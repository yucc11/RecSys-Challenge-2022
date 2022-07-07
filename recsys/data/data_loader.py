import datetime as dt
import os
import pickle
from pathlib import Path

import pandas as pd

data_dir = Path("./recsys/data/dressipi_recsys2022/")


class SessionDataLoader:
    """Process session records, and generate positive samples (items to be removed in session candidates).

    Example:
    dcit of posotive samples
    {
        list of clicked item ids (one session): set of purchased item ids
    }
    """

    def __init__(
        self,
        df_base: pd.DataFrame,
        day_start: str = "2021-04-20",
        day_end: str = "2021-05-31",
        drop_duplicate_clicks: bool = True,
        agg_seq_type: str = "list",
        agg_window: int = -1,
    ):
        """Initialize dictionary for positive samples.
        `day_start` `day_end` is the time range for collecting all positive samples.

        Parameters
        ----------
        df_base: pd.DataFrame
            dataframe of session records

        day_start: str = "2021-05-01"
            start date of session records

        day_end: str = "2021-05-31"
            end date of session records

        drop_duplicate_clicks: bool=True,
            drop duplicate clicks in session.

        agg_seq_type: str='list'
            'list', 'set'
            take sessions as same sample by
               - list: consider the click order
               - set: consider clicked items

        agg_window: int
            consider last n items in each session
            default -1, use full session

        """
        self._dt_start = dt.datetime.strptime(day_start, "%Y-%m-%d")
        self._dt_end = dt.datetime.strptime(day_end, "%Y-%m-%d")
        self._drop_duplicate_clicks = drop_duplicate_clicks
        self._agg_seq_type = agg_seq_type
        self._agg_window = agg_window
        self._days = (self._dt_end - self._dt_start).days + 1
        self._dict_path = (
            data_dir
            / "buy_set"
            / f"CliskSeq2BuySet_Base{self._days}Days_Drop{self._drop_duplicate_clicks}_AggBy{self._agg_seq_type}_window{self._agg_window}.pkl"
        )
        # load dict
        if os.path.exists(self._dict_path):
            self.click_seq2buy_set = pickle.load(open(self._dict_path, "rb"))
            print(f"Load {self._dict_path}.")
        else:
            print("Generate purchase set for each session.")
            self.click_seq2buy_set = self._gen_pos_set(df_base)

    def _gen_pos_set(
        self,
        df_base,
    ):
        """Generate positive samples (purchased item ids for the same clicked sequence).

        Parameters
        ----------
        df_base: pd.DataFrame
            dataframe of session records.

        Returns
        -------
        click_seq2buy_set: dict
            { 'item_id_list': set of purchased iid }
        """
        # process session records
        df_base = df_base.query(
            "(date >= @self._dt_start) and (date <= @self._dt_end)"
        ).sort_values(["session_id", "date"])
        if self._drop_duplicate_clicks:
            df_base = df_base.drop_duplicates(
                subset=["session_id", "item_id"], keep="last"
            )

        # last n item in each session
        if self._agg_window != -1:
            df_base = df_base.groupby(["session_id"]).tail(self._agg_window)

        if self._agg_seq_type == "list":
            df_seq = (
                df_base.groupby(["session_id", "target_item_id"])["item_id"]
                .apply(list)
                .astype(str)
                .reset_index(name="item_id_list")
            )
        elif self._agg_seq_type == "set":
            df_seq = (
                df_base.groupby(["session_id", "target_item_id"])["item_id"]
                .apply(set)
                .astype(str)
                .reset_index(name="item_id_list")
            )

        # positive set for click sequences
        click_seq2buy_set = (
            df_seq.groupby(["item_id_list"])["target_item_id"]
            .apply(set)
            .reset_index()
            .set_index("item_id_list")
            .to_dict()["target_item_id"]
        )

        # save dict
        pickle.dump(click_seq2buy_set, open(self._dict_path, "wb"))
        print(f"File saved at {self._dict_path}.")
        return click_seq2buy_set

    def get_data(
        self,
        df_sess: pd.DataFrame,
        day_start: str = None,
        day_end: str = None,
        infer_type: str = None,
    ):
        """Return processed session records and positive samples for each session.
        The positive sample includes the purchased items of the same clicked sequence and every item in the clicked sequence.

        Parameters
        ----------
        df_sess: pd.DataFrame
            dataframe of session records (full session)

        day_start: str = "2021-05-25"
            start date of session records

        day_end: str = "2021-05-31"
            end date of session records

        infer_type: str = None
            'lb', 'final', 'valid'
            Default = None, which represents the training process.

        Returns
        -------
        df_sess: pd.DataFrame
            processed session records

        sid2pos_set: dict
            { session id: set of purchased iid and clicked iid in session }
        """
        # process session records
        if (day_start is not None) and (day_end is not None):
            dt_start = dt.datetime.strptime(day_start, "%Y-%m-%d")
            dt_end = dt.datetime.strptime(day_end, "%Y-%m-%d")
            df_sess = df_sess.query("(date >= @dt_start) and (date <= @dt_end)")
        df_sess = df_sess.sort_values(["session_id", "date"])
        if self._drop_duplicate_clicks:
            df_sess = df_sess.drop_duplicates(
                subset=["session_id", "item_id"], keep="last"
            )

        if infer_type is None:  # train
            # items in full session
            df_full_seq = (
                df_sess.groupby(["session_id", "target_item_id"])["item_id"]
                .apply(set)
                .reset_index(name="item_id_set")
            )
            # last n item in each session
            if self._agg_window != -1:
                df_sess = df_sess.groupby(["session_id"]).tail(self._agg_window)

            # take purchased item set as positive samples
            if self._agg_seq_type == "list":
                df_seq = (
                    df_sess.groupby(["session_id", "target_item_id"])["item_id"]
                    .apply(list)
                    .astype(str)
                    .reset_index(name="item_id_list")
                )
            # TODO: elif self._agg_seq_type == 'set':

            df_seq["positive_tar_item_set"] = df_seq["item_id_list"].apply(
                lambda x: self.click_seq2buy_set[x]
                if x in self.click_seq2buy_set
                else set([])
            )
            # add clicked items in session to pos set
            df_seq["item_id_set"] = df_full_seq["item_id_set"]
            df_seq["positive_tar_item_set"] = df_seq.apply(
                lambda x: (x.positive_tar_item_set) | (x.item_id_set), axis=1
            )
            sid2pos_set = df_seq.set_index("session_id")[
                "positive_tar_item_set"
            ].to_dict()

        else:  # lb, final, valid
            # last n item in each session
            if self._agg_window != -1:
                df_sess = df_sess.groupby(["session_id"]).tail(self._agg_window)

            # sid2pos_set = {'session_id': 'clicked iid set in session'} .
            sid2pos_set = (
                df_sess.groupby(["session_id"])["item_id"]
                .apply(set)
                .reset_index(name="item_id_set")
                .set_index("session_id")["item_id_set"]
                .to_dict()
            )

        return df_sess, sid2pos_set

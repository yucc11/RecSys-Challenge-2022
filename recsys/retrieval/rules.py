import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from recsys.utils.utils import reduce_mem_usage


data_dir = Path("./recsys/data/dressipi_recsys2022/")


class ItemSimilarity:
    """Item based rule. Similarity between clicked item and target item."""

    def __init__(
        self,
        base_item_list: list,
        rec_item_list: list,
        wv_fname: str = "Word2Vec_v06_wv",
        topn: int = 300,
    ):
        """Generate top n candidates for each item.

        Parameters
        ----------
        base_item_list: list
            List of unique item ids in dataset (23691 items).
            Generate top n candidates for each item in `base_item_list`.

        rec_item_list: list
            List of recommended item ids (4990 items)

        wv_fname: str = "Word2Vec_v06_wv",
            File name of Word2Vec model

        topn: int = 300
            Top n most similar items

        """
        self._base_item_list = base_item_list
        self._rec_item_list = rec_item_list
        self._wv_fname = wv_fname
        self._wv_path = f"./recsys/modeling/word2vec/{wv_fname}.wordvectors"
        self._cand_path = (
            data_dir
            / "top_n_candidates"
            / f"ItemSimilarity_Top{topn}_{wv_fname}.parquet"
        )
        self._topn = topn

        if os.path.exists(self._cand_path):
            self.cands = pd.read_parquet(self._cand_path)
            print(f"Load {self._cand_path}.")
        else:
            print("Generate candidate file...")
            self.cands = self._gen_cands(self._topn)

    def _gen_cands(self, topn):
        """Generate top n similar items for each item in `base_item_list`.

        Parameters
        ----------
        topn: int = 300
            Top n most similar items

        Returns
        -------
        cands: pd.DataFrame
            columns = ["item_id", "cand_iid", "score"]
        """
        base_item_list = self._base_item_list
        rec_item_list = self._rec_item_list
        t0 = time.time()
        # load word vectors
        self._wv = KeyedVectors.load(self._wv_path, mmap="r")
        # self._words = list(self._wv.index_to_key) # vocab

        # top similar items and score
        sim = []
        for iid in base_item_list:
            sim += self._wv.most_similar([iid], topn=topn + 500)
        cands = pd.DataFrame(sim).rename(columns={0: "cand_iid", 1: "score"})
        cands["item_id"] = np.repeat(base_item_list, topn + 500).tolist()

        # select cand items in rec_item_list
        cands = cands.query("cand_iid in @rec_item_list")
        # select topn
        cands = (
            cands.sort_values(["item_id", "score"], ascending=False)
            .groupby(["item_id"])
            .head(topn)
        )
        # reduce memory & save file
        cands = reduce_mem_usage(cands)
        cands.to_parquet(self._cand_path)

        t1 = time.time()
        print(f"File saved at {self._cand_path}.")
        print(
            f"Total time for generating top 300 candidates for each item: {t1 - t0} sec."
        )
        return cands[["item_id", "cand_iid", "score"]]

    def retrieve(
        self,
        df_sess: pd.DataFrame,
        infer_type: str = None,
        sid2pos_set: Dict = None,
        week_num: int = 1,
        topk: int = 300,
        count_hit_num: bool = False,
        rm_items: bool = True,
        save_retrieve: bool = False,
    ) -> pd.DataFrame:
        """Retrieve candidates. Get most similar items correspond to each item in sessions.

        Parameters
        ----------
        df_sess: pd.DataFrame
            Dataframe to retrieve candidates on.
            Get df_sess from SessionDataLoader().get_data()

        infer_type: str=None, 'lb', 'final', 'valid'
            Default None, retrieve on training data
            - 'lb' - retrieve on testing leaderboard data
            - 'final' - retrieve on testing final data

        sid2pos_set: Dict
            Positive sameple set (items to be removed for each session).
            Get sid2pos_set from SessionDataLoader().get_data()

        week_num: int = 1
            week number for df_sess

        topk: int=300
            Top k most similar items

        count_hit_num: bool = False
            Evaluate rule performance

        rm_items: bool = True
            Default True, remove items that cannot be candidates.
            Set rm_items=False to evaluate rule performance.

        save_retrieve: bool = False
            Save df_sess after retrieve

        Returns
        -------
        res: pd.DataFrame
            columns = ["session_id", "item_id", "cand_iid", "score", "method"]
        """
        # inference or training
        if infer_type == "lb" or infer_type == "final":
            prefix = (
                f"{infer_type}_week{week_num}_ItemSimilarity_Top{topk}_{self._wv_fname}"
            )
        else:  # train, valid
            prefix = f"week{week_num}_ItemSimilarity_Top{topk}_{self._wv_fname}_HR"

        path = ""
        for fname in os.listdir(data_dir / "retrieve"):
            if fname.startswith(prefix):
                path = data_dir / "retrieve" / fname

        if os.path.exists(path):
            res = pd.read_parquet(path)
            print(f"Load {path}.")

        else:
            print("Retrieve 'ItemSimilarity' candidates...")
            t0 = time.time()

            res = df_sess.merge(self.cands, on=["item_id"], how="left")
            res["method"] = "ItemSimilarity"
            res = res.loc[res["score"].notnull()]
            res["cand_iid"] = res["cand_iid"].apply(int)

            # evaluate performance of rule
            hr = ""
            if count_hit_num and "target_item_id" in res.columns:
                hit_num = res[res["target_item_id"] == res["cand_iid"]][
                    "session_id"
                ].nunique()
                session_num = df_sess["session_id"].nunique()
                hr = round(hit_num / session_num, 4)
                print(f"Hit Rate: {hit_num} / {session_num} = {hr}")

            # remove positive samples of each session from negative sample candidates
            if rm_items:
                rm = []
                for idx, row in res.iterrows():
                    if row.cand_iid in sid2pos_set[row.session_id]:
                        rm += [idx]
                res = res.drop(rm, axis=0)

            res = reduce_mem_usage(res)
            # save retrieved
            if save_retrieve:
                retrieve_path = data_dir / "retrieve" / f"{prefix}{hr}.parquet"
                res.to_parquet(retrieve_path)
                print(f"File saved at {retrieve_path}.")

            t1 = time.time()
            print(
                f"Total time for generating top {topk} candidates for each item: {t1 - t0} sec."
            )

        return res  # [["session_id", "item_id", "cand_iid", "score", "method"]]

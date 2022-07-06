import datetime as dt
import gc

import pandas as pd
from gensim.models import FastText, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

import wandb
from recsys.data.data_helper import DataHelper
from recsys.utils.utils import load_config


class callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print("Loss after epoch {}: {}".format(self.epoch, loss_now))
        # log result
        wandb.log({"train_loss": loss_now})
        self.epoch += 1


def load_click_seq(cfg):
    """Prepare sentence like dataframe for gensim models."""

    # preprocess & load data
    data_dir = cfg["data_dir"]
    dh = DataHelper(base_dir=data_dir)
    data = dh.load_data()

    df_lb = data["lb_sessions"]
    df_final = data["final_sessions"]

    dt_start = dt.datetime.strptime(cfg["dt_start"], "%Y%m%d")
    dt_end = dt.datetime.strptime(cfg["dt_end"], "%Y%m%d")
    df_sessions, _ = dh.split_data(
        trans_data=data["train_sessions"],
        train_start_date=dt_start,
        train_end_date=dt_end,
        valid_start_date=dt_start,
        valid_end_date=dt_start,
    )

    # create "sentences"
    df_seq = (
        df_sessions.groupby(["session_id", "target_item_id"])["item_id"]
        .apply(list)
        .reset_index(name="item_list")
    )
    if cfg["add_tar_item"]:
        df_seq["item_list"] = df_seq.apply(
            lambda x: x.item_list + [x.target_item_id], axis=1
        )

    del data, df_sessions, df_lb, df_final
    gc.collect()

    # filter out one-element lists (useless for this kind of embedding)
    df_seq["list_size"] = df_seq["item_list"].apply(len)
    df_seq = df_seq.loc[df_seq["list_size"] > cfg["cutoff"]]

    return df_seq[["session_id", "item_list"]]


def train(df_seq, cfg):
    """ """
    wandb.init(
        project="RecSys",
        job_type="train",
        config=cfg,
        group=cfg["model_name"],
        name=cfg["version_name"],
    )
    print("Training process starts...")

    # model
    if cfg["model_name"] == "FastText":
        model = FastText(
            vector_size=cfg["embedding_dim"],
            window=cfg["window"],
            min_count=cfg["cutoff"],
        )
        model.build_vocab(df_seq.item_list.values.tolist())
        model.train(
            df_seq.item_list.values.tolist(),
            total_examples=len(df_seq),
            epochs=cfg["epochs"],
            compute_loss=True,
            callbacks=[callback()],
        )

    elif cfg["model_name"] == "Word2Vec":
        model = Word2Vec(
            vector_size=cfg["embedding_dim"],
            window=cfg["window"],
            min_count=cfg["cutoff"],
        )
        model.build_vocab(df_seq.item_list.values.tolist())
        model.train(
            df_seq.item_list.values.tolist(),
            total_examples=len(df_seq),
            epochs=cfg["epochs"],
            compute_loss=True,
            callbacks=[callback()],
        )

    # save model & trained embeddings
    model.save(f"{cfg['model_path']}/{cfg['model_name']}_{cfg['version_name']}.model")
    model.wv.save(
        f"{cfg['model_path']}/{cfg['model_name']}_{cfg['version_name']}_wv.wordvectors"
    )

    wandb.finish()
    print("End.")


if __name__ == "__main__":
    # args
    CFG_PATH = "./recsys/config/word2vec.yml"

    # load configs
    cfg = load_config(CFG_PATH)
    # load data
    df_seq = load_click_seq(cfg)
    # train model
    train(df_seq, cfg)

import pandas as pd
import numpy as np
import datetime


class Dataloader:
    # Overly specific dataloader class

    def __init__(self):
        self.sf = pd.read_csv("data/processed/sf.csv")
        self.mf = pd.read_csv("data/processed/mf.csv")
        self.condo = pd.read_csv("data/processed/condo.csv")
        self.ds_dict = {
            "sf" : self.sf,
            "mf" : self.mf,
            "condo" : self.condo
        }

    def subset_to_interest(self,
        sf_maxprice=750000,
        mf_maxprice=1000000,
        hoa_maxprice=400
        ):

        self.sf = self.sf.query(f"PRICE < {sf_maxprice}")

        self.condo = self.condo.query(f"PRICE < {sf_maxprice}")
        self.condo = self.condo.query(f"`HOA/MONTH` < {hoa_maxprice}")

        self.mf = self.mf.query(f"PRICE < {mf_maxprice}")



def create_data():
    dt_pull_date = datetime.datetime(year=2021, month=9, day=8)
    pull_date = "2021-09-08" 
    avail_df = pd.read_csv("data/raw/available_redfin_2021-09-08-18-54-22.csv")
    avail_df["list_date"] = np.where(
        avail_df["SOLD DATE"].isna(),
        pd.Timestamp(pull_date) - pd.to_timedelta(avail_df["DAYS ON MARKET"], unit="days"),
        pd.to_datetime(avail_df["SOLD DATE"]) - pd.to_timedelta(avail_df["DAYS ON MARKET"], unit="days")
    )   
    avail_df_skim = avail_df[~avail_df["list_date"].isna()]

    sf_df = avail_df_skim.query("`PROPERTY TYPE` == 'Single Family Residential'")
    mf_df = avail_df_skim.query("`PROPERTY TYPE` == 'Multi-Family (2-4 Unit)'")
    condo_df = avail_df_skim.query("`PROPERTY TYPE` == 'Townhouse' | `PROPERTY TYPE` == 'Condo/Co-op'")

    for df, name in zip(
        [sf_df, mf_df, condo_df],
        ["sf","mf","condo"]
    ):
        df.to_csv(f"data/processed/{name}.csv")


if __name__ == "__main__":

    create_data()
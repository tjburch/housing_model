import pandas as pd
import numpy as np
import datetime

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
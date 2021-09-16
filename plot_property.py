########################################################################################
# Load modules
########################################################################################
import pandas as pd
import arviz as az
import argparse
from data_pipeline import Dataloader
import cloudpickle
import matplotlib.pyplot as plt
plt.style.use("ggplot")

########################################################################################
# Parse Arguments
########################################################################################
parser = argparse.ArgumentParser(description="Plot house diagnostics")
parser.add_argument("-b", "--beds", type=int, required=True, help="Number of Beds")
parser.add_argument("-p", "--baths", type=float, required=True, help="Number of Baths")
parser.add_argument("-s", "--sqft", type=float, required=True, help="Square footage")
parser.add_argument(
    "-t",
    "--type",
    type=str,
    required=True,
    help="Property type (sf, mf, condo accepted)",
)
parser.add_argument("-l", "--listprice", type=float, required=True, help="List Price")
parser.add_argument("--hoa", type=float, required=False, help="HOA fees")
parser.add_argument("-f", "--figs", default=False, action="store_true", help="Save figures")
args = parser.parse_args()

########################################################################################
# Load associated inferencedata and model
########################################################################################
if not args.hoa:
    model = cloudpickle.load(open(f"models/bbs_linear_{args.type}_model.pkl", "rb"))
    idata = az.from_netcdf(f"models/bbs_linear_{args.type}_idata.nc")
else:
    model = cloudpickle.load(open(f"models/condo_bbsh_linear_{args.type}_model.pkl", "rb"))
    idata = az.from_netcdf(f"models/condo_bbsh_linear_{args.type}_idata.nc")


########################################################################################
# Create Data for Given Property and predict
########################################################################################
check_data = pd.DataFrame(
    {"SQUARE FEET": args.sqft, "BEDS": args.beds, "BATHS": args.baths}, index=[1]
)
if args.hoa:
    check_data["HOA"] = args.hoa

predict_idata = model.predict(idata, data=check_data, inplace=False)
predict_value = predict_idata.posterior["PRICE_mean"].mean().round(2).values
price_posterior_values = predict_idata.posterior["PRICE_mean"].values.flatten()

########################################################################################
# Check if passing or failing the model
########################################################################################
print(f"List Price: {args.listprice}")
print(f"Predicted Price: {predict_value}")
print(f"Price above prediction: {args.listprice - predict_value:.2f}")
cdf_eval = 100 * (
    (price_posterior_values < args.listprice).sum() / len(price_posterior_values)
)
print(f"This is in the {cdf_eval:.2f} price percentile")

if predict_value - args.listprice > 0:
    print("Listing is ACCEPTED by the model as a good value")
else:
    print("Listing is REJECTED by the model as an overpay")

########################################################################################
# Make figure for diagnostics
########################################################################################
fig, ax = plt.subplot_mosaic(
    """
    AAA
    BCD
    """,
    figsize=(12,12)
)

########################################################################################
# Make Posterior Probability Distribution for Price
########################################################################################
plt.sca(ax["A"])
az.plot_kde(predict_idata.posterior["PRICE_mean"], label="Model Expectation")
plt.axvline(args.listprice, linestyle="--", color="black", label="List Price")
plt.xlabel("Price")
plt.ylim(bottom=0)
plt.legend(frameon=False, fontsize=13)


if args.figs:
    plt.savefig(f"projections/{args.type}_{args.beds}br_{str(args.baths).replace('.','')}bt_{round(args.sqft)}sqft.png")
else:
    plt.show()
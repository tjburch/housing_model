########################################################################################
# Load modules
########################################################################################
from data_pipeline import Dataloader
from bambi import Model
import arviz as az
import os
import matplotlib.pyplot as plt
import cloudpickle

plt.style.use("ggplot")

########################################################################################
# Define Global Paths
########################################################################################
MODEL_DIR = "models/"
DIAGNOSTIC_DIR = "diagnostics/"


def main():
    # Note - have to wrap in main for multiprocess sampling on my mac

    ####################################################################################
    # Load data
    ####################################################################################
    datasets = Dataloader()
    datasets.subset_to_interest()

    ####################################################################################
    # Create dict of model name : fomula pairs
    ####################################################################################
    formula_dictionary = {
        "bbs_linear": "PRICE ~ BEDS + BATHS + scale(`SQUARE FEET`)",
        "bbs_interaction": "PRICE ~ BEDS*BATHS + scale(`SQUARE FEET`)",
    }

    ####################################################################################
    # Iterate over dictionary and fit model for each
    ####################################################################################
    model_dictionary = {}
    for ptype, df in datasets.ds_dict.items():
        for title, formula in formula_dictionary.items():

            ############################################################################
            # Create model, fit, run ppc
            ############################################################################
            model = Model(formula, df, dropna=True)
            idata = model.fit(draws=3000, chains=2)
            ppc = model.predict(idata, kind="pps", draws=500)

            ############################################################################
            # Save model to file and dictionary
            ############################################################################
            cloudpickle.dump(model, open(f"models/{title}_{ptype}_model.pkl", "wb"))
            idata.to_netcdf(filename=f"models/{title}_{ptype}_idata.nc")
            model_dictionary[title] = idata

            ############################################################################
            # Run Model-level Checking Tests
            ############################################################################
            # Create Folder if needed
            pathway = DIAGNOSTIC_DIR + "/" + title + "_" + ptype
            if not os.path.exists(pathway):
                os.makedirs(pathway)

            # Trace
            az.plot_trace(idata)
            plt.savefig(pathway + "/trace")
            plt.close()

            # PPCs
            az.plot_ppc(idata)
            plt.savefig(pathway + "/ppc")
            plt.close()

        ############################################################################
        # Run model comparisons
        ############################################################################
        az.plot_compare(az.compare(model_dictionary))
        plt.savefig(f"diagnostics/{ptype}_model_comparison")


if __name__ in "__main__":
    main()

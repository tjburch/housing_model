from data_pipeline import Dataloader
from bambi import Model
import arviz as az

# Load data
datasets = Dataloader()
datasets.subset_to_interest()

# Generate a big set of models
formula_dictionary = {
    "bbs_linear" : "PRICE ~ BEDS + BATHS + scale(`SQUARE FEET`)",
    "bbs_interaction" : "PRICE ~ BEDS*BATHS + scale(`SQUARE FEET`)",
}

for title, formula in formula_dictionary.items():
    
    for ptype, df in datasets.ds_dict.items():

        # Create model, fit 
        model = Model(formula, df, dropna=True)
        idata = model.fit(draws=3000, chains=1)
        ppc = model.predict(idata, kind="pps", draws=500)
        # Save
        idata.to_netcdf(filename=f"models/{title}_{ptype}.nc")
#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os

# %%
# user settings
# when calculating the number of fields per cell, in many cases we get only 1 or 2 fields (as the agshare is in some cells very small)
# this would mean that the stochastic uncertainty is very big. We therefore can impose that a cell should have min_n_of_fields fields, at least, and for all cells
# with a lower calculated number of fields min_n_of_fields is instead used.
# to allow all number of fields, just set min_n_of_fields to 0
min_n_of_fields = 5
# note that this will be overwritten, if min_n_of_fields is defined in the user specifications
#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]


raw_data_path=data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
delineation_and_parameter_path = (
    data_main_path+"delineation_and_parameters/"
)
#%%
# input paths
LUCAS_preprocessed_path = intermediary_data_path + "Preprocessed_Inputs/LUCAS/LUCAS_preprocessed.csv"
parameter_path = delineation_and_parameter_path+"DGPCM_user_parameters.xlsx"
nuts_input_path = intermediary_data_path + "Preprocessed_Inputs/NUTS/NUTS_all_regions_all_years.csv"
inferred_UAA_path = intermediary_data_path + "Zonal_Stats/"
excluded_NUTS_regions_path = delineation_and_parameter_path + "excluded_NUTS_regions.xlsx"

# output paths
n_of_fields_output_path = intermediary_data_path + "Zonal_Stats/"
# %%
if __name__ == "__main__":
    # import parameters
    selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
    countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
    country_codes_relevant = np.array(countries["country_code"])
    LUCAS_fieldsize_conversion = pd.read_excel(
        parameter_path, sheet_name="LUCAS_fieldsize"
    )
    excluded_NUTS_regions = pd.read_excel(excluded_NUTS_regions_path)
    excluded_NUTS_regions = np.array(excluded_NUTS_regions["excluded NUTS1 regions"])
    levl_info = pd.read_excel(parameter_path, sheet_name="lowest_agg_level")
    try:
        min_n_of_fields_info = pd.read_excel(
            parameter_path, sheet_name="cell_min_n_of_fields"
        )
        min_n_of_fields = min_n_of_fields_info["cell_min_n_of_fields"].iloc[0]
    except:
        pass
    # import data
    #%%
    print("calculate field size...")
    print("assumed minimum number of fields per cell: "+ str(min_n_of_fields))
    #%%
    # load data on NUTS regions in a year
    NUTS_data = pd.read_csv(nuts_input_path)
    NUTS_data=NUTS_data[(NUTS_data["CNTR_CODE"].isin(country_codes_relevant))& (NUTS_data["year"].isin(selected_years))]
    LUCAS_preprocessed = pd.read_csv(LUCAS_preprocessed_path)
    #%%
    LUCAS_parcel = LUCAS_preprocessed[
        ["nuts0", "nuts1", "nuts2", "nuts3", "parcel_area_ha"]
    ]
    LUCAS_parcel.dropna(inplace=True)
    #%%
    LUCAS_parcel
    #%%
    converted_parcel_size_array = np.ndarray(len(LUCAS_parcel))
    for i, size in enumerate(np.array(LUCAS_fieldsize_conversion["LUCAS"])):
        converted_parcel_size_array[
            np.where(np.array(LUCAS_parcel["parcel_area_ha"]) == size)[0]
        ] = LUCAS_fieldsize_conversion["field_size_in_ha"].iloc[i]
    LUCAS_parcel["converted_parcel_size"] = converted_parcel_size_array

    #%%
    # use the median field size for the nuts3 region. if this is nan (because there is no data), use the median of the higher NUTS level for which data is available

    median_field_size_dict = {
        "country": [],
        "nuts3": [],
        "year": [],
        "median_field_size": [],
    }
    for country in country_codes_relevant:
        for year in selected_years:
            nuts3_regs = np.array(
                NUTS_data[
                    (NUTS_data["CNTR_CODE"] == country)
                    & (NUTS_data["year"] == year)
                    & (NUTS_data["LEVL_CODE"] == 3)
                ]["NUTS_ID"]
            )
            for reg in nuts3_regs:
                median_field_size_dict["country"].append(country)
                median_field_size_dict["nuts3"].append(reg)
                median_field_size_dict["year"].append(year)
                median_selected_reg = LUCAS_parcel[LUCAS_parcel["nuts3"] == reg][
                    "converted_parcel_size"
                ].median()
                if np.isnan(median_selected_reg):
                    median_selected_reg = LUCAS_parcel[
                        LUCAS_parcel["nuts2"] == reg[:4]
                    ]["converted_parcel_size"].median()
                    if np.isnan(median_selected_reg):
                        median_selected_reg = LUCAS_parcel[
                            LUCAS_parcel["nuts1"] == reg[:3]
                        ]["converted_parcel_size"].median()
                        if np.isnan(median_selected_reg):
                            median_selected_reg = LUCAS_parcel[
                                LUCAS_parcel["nuts0"] == country
                            ]["converted_parcel_size"].median()
                median_field_size_dict["median_field_size"].append(median_selected_reg)

    median_field_size_df = pd.DataFrame(median_field_size_dict)
    median_field_size_df["median_n_of_fields_per_km2"] = 100 / np.array(
        median_field_size_df["median_field_size"]
    )
    median_field_size_df["nuts2"] = np.array(median_field_size_df["nuts3"]).astype("U4")
    median_field_size_df["nuts1"] = np.array(median_field_size_df["nuts3"]).astype("U3")
    median_field_size_df = median_field_size_df[
        [
            "country",
            "nuts1",
            "nuts2",
            "nuts3",
            "year",
            "median_field_size",
            "median_n_of_fields_per_km2",
        ]
    ]
    
    #%%
    """take info on a cell's agshare into account"""
    
    median_field_size_dict = {
        "country": [],
        "NUTS_ID": [],
        "year": [],
        "median_field_size": [],
    }
    for country in country_codes_relevant:
        levl_info_selected = levl_info[levl_info["country_code"] == country][
            "lowest_agg_level"
        ].iloc[0]
        nuts1_regs = np.unique(
            np.array(
                NUTS_data[
                    (NUTS_data["CNTR_CODE"] == country) & (NUTS_data["LEVL_CODE"] == 1)
                ]["NUTS_ID"]
            )
        )
        # discard those nuts regions which are excluded (e.g., France overseas)
        nuts1_regs = nuts1_regs[
            np.where(np.isin(nuts1_regs, excluded_NUTS_regions, invert=True))[0]
        ]
        all_inferred_UAA_data_country = pd.DataFrame()
        for reg in nuts1_regs:
            inferred_UAA_data = pd.read_csv(
                inferred_UAA_path + country + "/inferred_UAA/1kmgrid_" + reg + ".csv"
            )
            all_inferred_UAA_data_country = pd.concat(
                (all_inferred_UAA_data_country, inferred_UAA_data)
            )

        relevant_nuts_regs = np.unique(
            np.array(all_inferred_UAA_data_country[f"nuts{levl_info_selected}"])
        )
        median_n_of_field_array = np.zeros(len(all_inferred_UAA_data_country))
        for subreg in relevant_nuts_regs:
            median_subreg = median_field_size_df[
                (median_field_size_df[f"nuts{levl_info_selected}"] == subreg)
            ]["median_n_of_fields_per_km2"].mean()
            median_n_of_field_array[
                np.where(
                    all_inferred_UAA_data_country[f"nuts{levl_info_selected}"] == subreg
                )[0]
            ] = median_subreg
        all_inferred_UAA_data_country["median_n_of_fields"] = median_n_of_field_array

        n_of_fiels_array = np.multiply(
            (np.array(all_inferred_UAA_data_country["inferred_UAA"]) / 1000000),
            np.array(all_inferred_UAA_data_country["median_n_of_fields"]),
        )
        all_inferred_UAA_data_country["n_of_fields_calculated"] = n_of_fiels_array
        # round values up, and if min_n_of_fields > 0 impose this
        all_inferred_UAA_data_country["n_of_fields_assumed"] = np.max(
            (
                np.ceil(all_inferred_UAA_data_country["n_of_fields_calculated"]),
                np.repeat(min_n_of_fields, len(all_inferred_UAA_data_country)),
            ),
            axis=0,
        )
        all_inferred_UAA_data_country = all_inferred_UAA_data_country[
            [
                "CELLCODE",
                f"nuts{levl_info_selected}",
                "year",
                "median_n_of_fields",
                "n_of_fields_calculated",
                "n_of_fields_assumed",
            ]
        ]
        #%%
        all_inferred_UAA_data_country
        #%%
        # export data for country
        print(f"export data for {country}")
        Path(n_of_fields_output_path + country + "/n_of_fields/").mkdir(
            parents=True, exist_ok=True
        )
        all_inferred_UAA_data_country.to_csv(
            n_of_fields_output_path
            + country
            + "/n_of_fields/n_of_fields_allcountry_"+str(selected_years.min())+str(selected_years.max())+".csv"
        )
# %%

# %%

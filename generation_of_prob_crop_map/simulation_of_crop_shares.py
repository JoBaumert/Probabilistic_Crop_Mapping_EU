# %%from cProfile import label
import argparse
import gc
import jax.numpy as jnp  # it works almost exactly like numpy
from jax import grad, jit, vmap, hessian
from jax import random
from jax import jacfwd, jacrev


import jax
import jax.numpy as jnp
from jaxopt import projection
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative
from jaxopt.projection import projection_box
from jaxopt import BoxOSQP
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy import nanargmin, nanargmax
import os
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from sklearn.metrics import r2_score
import sys
sys.path.append(
    str(Path(Path(os.path.abspath(__file__)).parents[0]))+"/modules/"
)
import modules.functions_for_prediction as ffp

# %%
# %env XLA_PYTHON_CLIENT_PREALLOCATE=false
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

postsampling_reps = 50  # overwritten if defined in user specifications
deviation_tolerance = 0.01  # overwritten if defined in user specifications
n_of_param_samples = 10 # overwritten if defined in user specifications
true_n_of_fields = False  # overwritten if defined as true via parser
ignore_consistency = True

# %%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]

raw_data_path=data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
delineation_and_parameter_path = (
    data_main_path+"delineation_and_parameters/"
)
estimated_parameter_path=data_main_path+"Results/Model_Parameter_Estimates/"
#input paths
parameter_path=delineation_and_parameter_path+"DGPCM_user_parameters.xlsx"
nuts_path=intermediary_data_path+"Preprocessed_Inputs/NUTS/NUTS_all_regions_all_years.csv"
excluded_NUTS_regions_path=delineation_and_parameter_path+ "excluded_NUTS_regions.xlsx"

UAA_input_path = (
    intermediary_data_path+ "Regional_Aggregates/coherent_UAA_"
)
cropdata_input_path = (
    intermediary_data_path+ "Regional_Aggregates/cropdata_"
)

parameter_path = delineation_and_parameter_path + "DGPCM_user_parameters.xlsx"

crop_level_path=intermediary_data_path+"Regional_Aggregates/crop_levels_selected_countries_"
posterior_probabilities_path=data_main_path+"Results/Posterior_crop_probability_estimates/"
n_of_fields_path=intermediary_data_path+"Zonal_Stats/"

#output paths
temporary_file_path=data_main_path+"/Temporary/"
results_postsampling_path=data_main_path+"Results/Simulated_consistent_crop_shares/"

# %%
# import parameters
selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])

parser = argparse.ArgumentParser()
parser.add_argument("-cc", "--ccode", type=str, required=False)
parser.add_argument("-y", "--year", type=int, required=False)
parser.add_argument("--from_xlsx", type=str, required=False)
parser.add_argument("--true_n_of_fields", type=str, required=False)

args = parser.parse_args()
if args.from_xlsx == "True":
    country_year_list = [np.repeat(country_codes_relevant,len(selected_years)),np.tile(selected_years,len(country_codes_relevant))]

else:
    country_year_list = [[args.ccode], [args.year]]

if args.true_n_of_fields == "True":
    true_n_of_fields = True
country_year_list = [np.repeat(country_codes_relevant,len(selected_years)),np.tile(selected_years,len(country_codes_relevant))]

# %%
""" SET UP FUNCTIONS NEEDED FOR THE EVALUATION"""

def generate_random_results(
    posterior_probabilities_selected, n_of_fields_array, postsampling_reps
):
    p_matrix = np.array(
        posterior_probabilities_selected["posterior_probability"]
    ).reshape((I, C))

    """for some cells the sums of all crop probabilities are marginally larger than 1 (like 1.0001). 
    Normalize them to ensure that all probabilities within a cell add up to 1 (or marginally less), 
    which is necessary for the random sampling
    """
    p_matrix_corrected = np.array(
        [p_vector / (p_vector.sum() + 0.0001) for p_vector in p_matrix]
    )
    p_matrix_corrected = np.where(
        p_matrix_corrected == 0, 0.000001, p_matrix_corrected
    )

    random_results = np.array(
        [
            np.random.multinomial(
                n_of_fields_array[i], p_matrix_corrected[i], postsampling_reps
            )
            / n_of_fields_array[i]
            for i in range(p_matrix_corrected.shape[0])
        ]
    )
    return random_results

def calculate_level_UAA(level):
    level_UAA = relevant_UAA[["NUTS_ID", "UAA_in_ha"]].copy()
    level_UAA["level_NUTS_ID"] = np.array(level_UAA["NUTS_ID"]).astype(
        f"U{2+level}"
    )
    level_UAA = level_UAA.groupby("level_NUTS_ID").sum().reset_index()
    return np.array(level_UAA["UAA_in_ha"])

def create_region_construction_matrix_alternative(level):
    regions = np.sort(
        np.array(
            nuts_regions_relevant[nuts_regions_relevant["LEVL_CODE"] == level][
                "NUTS_ID"
            ]
        )
    )
    # create a matrix of the dimension (# of regions x newC*I) which indicates with 1 in the respective row which region a cell belongs to
    region_construction_matrix = np.zeros((len(regions), I))
    weight_df_regions = np.array(
        cell_weight_info_selected["lowest_relevant_NUTS_level"]
    ).astype(f"U{level+2}")
    for r, region in enumerate(regions):
        region_construction_matrix[r][
            np.where(weight_df_regions == region)[0]
        ] = 1
    return region_construction_matrix

def calculate_deviation_from_known_aggregates(x, level):
    if level == 0:
        nuts_regions_construction_matrix_alternative = (
            nuts0_regions_construction_matrix_alternative
        )
        nuts_cropdata_matrix = nuts0_cropdata_matrix
        UAA_nuts_inverse = UAA_nuts0_inverse
    if level == 1:
        nuts_regions_construction_matrix_alternative = (
            nuts1_regions_construction_matrix_alternative
        )
        nuts_cropdata_matrix = nuts1_cropdata_matrix
        UAA_nuts_inverse = UAA_nuts1_inverse
    if level == 2:
        nuts_regions_construction_matrix_alternative = (
            nuts2_regions_construction_matrix_alternative
        )
        nuts_cropdata_matrix = nuts2_cropdata_matrix
        UAA_nuts_inverse = UAA_nuts2_inverse
    if level == 3:
        nuts_regions_construction_matrix_alternative = (
            nuts3_regions_construction_matrix_alternative
        )
        nuts_cropdata_matrix = nuts3_cropdata_matrix
        UAA_nuts_inverse = UAA_nuts3_inverse

    x_transposed = x.transpose()
    weighted_crops = np.multiply(weight_array, x_transposed)
    nuts_prediction = np.matmul(
        weighted_crops, nuts_regions_construction_matrix_alternative
    )

    relative_deviation = np.multiply(
        (nuts_prediction.transpose() - nuts_cropdata_matrix).transpose(),
        UAA_nuts_inverse,
    )
    relative_negative_deviation = np.where(
        relative_deviation > 0, 0, np.abs(relative_deviation)
    )
    return (
        relative_negative_deviation.max(),
        relative_deviation.min(),
        relative_deviation.max(),
    )
# %%
country_year_list[1]
#%%
UAA=pd.read_csv(UAA_input_path+str(country_year_list[1].min())+str(country_year_list[1].max())+".csv")
cropdata = pd.read_csv(cropdata_input_path+str(country_year_list[1].min())+str(country_year_list[1].max())+".csv")
crop_levels = pd.read_csv(crop_level_path+str(country_year_list[1].min())+str(country_year_list[1].max())+".csv")
#%%
if __name__ == "__main__":
    for c in range(len(country_year_list[0])):
        country = country_year_list[0][c]
        year = country_year_list[1][c]
        print(country + " " + str(year))
        #country specific input files
        

        cell_weight_path = (
            intermediary_data_path
            + "Regional_Aggregates/Cell_Weights/"
            + country
            + "/cell_weights_"+str(country_year_list[1].min())+str(country_year_list[1].max())+".csv"
        )
        

        """ import parameters"""
        nuts_info = pd.read_excel(parameter_path, sheet_name="NUTS")
        nuts_year = np.sort(nuts_info[nuts_info["crop_map_year"]==year]["nuts_year"].value_counts().keys())
        crop_info = pd.read_excel(parameter_path, sheet_name="crops")
        all_crops = np.array(crop_info["crop"])
        levl_info = pd.read_excel(parameter_path, sheet_name="lowest_agg_level")
        country_levls = {}
        for i in range(len(levl_info)):
            country_levls[levl_info.iloc[i].country_code] = levl_info.iloc[
                i
            ].lowest_agg_level
   
        excluded_NUTS_regions = pd.read_excel(excluded_NUTS_regions_path)
        excluded_NUTS_regions = np.array(
            excluded_NUTS_regions["excluded NUTS1 regions"]
        )
        try:
            other_settings = pd.read_excel(
                user_specifications_path, sheet_name="other_settings"
            )
            postsampling_reps = other_settings["postsampling_reps"].iloc[0]
            deviation_tolerance = other_settings["deviation_tolerance"].iloc[0]
            n_of_param_samples = other_settings["n_of_param_samples"].iloc[0]
        except:
            pass

        # %%

        nuts_input = pd.read_csv(nuts_path)
        cell_weight_info = pd.read_csv(cell_weight_path)
        posterior_probabilities = pd.read_parquet(
            posterior_probabilities_path
            + country
            + "/"
            + country
            + str(year)
            + "entire_country"
        )
        n_of_fields = pd.read_csv(n_of_fields_path+country+"/n_of_fields/n_of_fields_allcountry_"+
                                  str(country_year_list[1].min())+str(country_year_list[1].max())+".csv")
        # %%
        nuts_regions_relevant = nuts_input[
            (nuts_input["CNTR_CODE"] == country) & (nuts_input["year"] == year)
        ]

        nuts_regions_relevant = nuts_regions_relevant.iloc[
            np.where(
                np.isin(
                    np.array(nuts_regions_relevant["NUTS_ID"]).astype("U3"),
                    excluded_NUTS_regions,
                ).astype(int)
                == 0
            )[0]
        ]
        # %%
       
        #some cellcodes appear in multiple NUTS2 or NUTS3 regions (those at the border). Generate unique identifiers
        #for each cell in each region
        n_of_fields_selected = n_of_fields[n_of_fields["year"] == year]

        cellcode_unique = list(
            map(
                "".join,
                zip(
                    np.array(n_of_fields_selected["CELLCODE"]),
                    np.array(n_of_fields_selected.iloc[:,np.where(np.array(n_of_fields_selected.columns).astype("U4")=="nuts")[0]]).squeeze(),
                ),
            )
        )

        n_of_fields_selected["CELLCODE_unique"] = cellcode_unique
        
        # %%
        crop_levels_selected=crop_levels[(crop_levels["country"]==country)&(crop_levels["year"]==year)]
        if crop_levels_selected["NUTS3"].sum()>0:
            nuts_level_country_year=3
        elif crop_levels_selected["NUTS2"].sum()>0:
            nuts_level_country_year=2
        elif crop_levels_selected["NUTS1"].sum()>0:
            nuts_level_country_year=1
        else:
            nuts_level_country_year=0
        #%%
        UAA["NUTS_LEVL"] = np.vectorize(len)(UAA["NUTS_ID"]) - 2
        cropdata["NUTS_LEVL"] = np.vectorize(len)(cropdata["NUTS_ID"]) - 2
        cropdata_relevant = cropdata[
            (cropdata["country"] == country) & (cropdata["year"] == year)
        ]
        # prepare data on UAA
        relevant_UAA = UAA[
            (UAA["country"] == country)
            & (UAA["year"] == year)
            & (UAA["NUTS_LEVL"] == nuts_level_country_year)
        ]
        # write UAA in ha
        relevant_UAA["UAA_in_ha"] = relevant_UAA["UAA_corrected"] * 1000
        # %%

        beta = 0

        posterior_probabilities_selected = posterior_probabilities[
            posterior_probabilities["beta"] == beta
        ]
        posterior_probabilities_selected.sort_values(
            by=["lowest_relevant_NUTS_level", "CELLCODE_unique", "crop"], inplace=True
        )
        # %%
        C = len(all_crops)
        I = int(len(posterior_probabilities_selected) / C)

        ordered_cellcodes_unique = (
            np.array(posterior_probabilities_selected["CELLCODE_unique"])
            .reshape((I, C))
            .transpose()[0]
        )

        ordered_nuts1_each_cell = (
            np.array(posterior_probabilities_selected["NUTS1"])
            .reshape((I, C))
            .transpose()[0]
        )

        cell_weight_info_selected = cell_weight_info[
            (cell_weight_info["CELLCODE_unique"].isin(ordered_cellcodes_unique))
            & (cell_weight_info["year"] == year)
        ]
        cell_weight_info_selected.sort_values(
            by=["lowest_relevant_NUTS_level", "CELLCODE"], inplace=True
        )
        weight_array = np.array(cell_weight_info_selected["weight"])
        # %%
        if nuts_level_country_year == 3:
            nuts3_cropdata = cropdata_relevant[cropdata_relevant["NUTS_LEVL"] == 3][
                ["NUTS_ID", "crop", "area"]
            ].sort_values(by=["NUTS_ID", "crop"])
            nuts3_cropdata_matrix = (
                np.array(
                    nuts3_cropdata.pivot(index="NUTS_ID", columns="crop", values="area")
                )
                * 1000
            )
            nuts3_regions_construction_matrix_alternative = (
                create_region_construction_matrix_alternative(3).transpose()
            )
            UAA_nuts3 = calculate_level_UAA(3)
            UAA_nuts3_inverse = 1 / UAA_nuts3
            if len(np.where(UAA_nuts3 == 0)) > 0:
                UAA_nuts3_inverse[np.where(UAA_nuts3 == 0)] = 0

        if nuts_level_country_year >= 2:
            nuts2_cropdata = cropdata_relevant[cropdata_relevant["NUTS_LEVL"] == 2][
                ["NUTS_ID", "crop", "area"]
            ].sort_values(by=["NUTS_ID", "crop"])
            nuts2_cropdata_matrix = (
                np.array(
                    nuts2_cropdata.pivot(index="NUTS_ID", columns="crop", values="area")
                )
                * 1000
            )
            nuts2_regions_construction_matrix_alternative = (
                create_region_construction_matrix_alternative(2).transpose()
            )
            UAA_nuts2 = calculate_level_UAA(2)
            UAA_nuts2_inverse = 1 / UAA_nuts2
            if len(np.where(UAA_nuts2 == 0)) > 0:
                UAA_nuts2_inverse[np.where(UAA_nuts2 == 0)] = 0

        if nuts_level_country_year >= 1:
            nuts1_cropdata = cropdata_relevant[cropdata_relevant["NUTS_LEVL"] == 1][
                ["NUTS_ID", "crop", "area"]
            ].sort_values(by=["NUTS_ID", "crop"])
            nuts1_cropdata_matrix = (
                np.array(
                    nuts1_cropdata.pivot(index="NUTS_ID", columns="crop", values="area")
                )
                * 1000
            )
            nuts1_regions_construction_matrix_alternative = (
                create_region_construction_matrix_alternative(1).transpose()
            )
            UAA_nuts1 = calculate_level_UAA(1)
            UAA_nuts1_inverse = 1 / UAA_nuts1
            if len(np.where(UAA_nuts1 == 0)) > 0:
                UAA_nuts1_inverse[np.where(UAA_nuts1 == 0)] = 0

        nuts0_cropdata = cropdata_relevant[cropdata_relevant["NUTS_LEVL"] == 0][
            ["NUTS_ID", "crop", "area"]
        ].sort_values(by=["NUTS_ID", "crop"])
        nuts0_cropdata_matrix = (
            np.array(
                nuts0_cropdata.pivot(index="NUTS_ID", columns="crop", values="area")
            )
            * 1000
        )
        nuts0_regions_construction_matrix_alternative = (
            create_region_construction_matrix_alternative(0).transpose()
        )
        UAA_nuts0 = calculate_level_UAA(0)
        UAA_nuts0_inverse = 1 / UAA_nuts0

        # %%
        
        #%%
        n_of_fields_selected = n_of_fields_selected[
            n_of_fields_selected["CELLCODE_unique"].isin(ordered_cellcodes_unique)
        ]
        n_of_fields_selected.sort_values(
            by=[n_of_fields_selected.columns[np.where(np.array(n_of_fields_selected.columns).astype("U4")=="nuts")[0]][0], 
                "CELLCODE_unique"], inplace=True
        )

        n_of_fields_array = np.array(n_of_fields_selected["n_of_fields_assumed"])
        

        # %%
        deviation_dict = {
            "beta": [],
            "accepted": [],
            "NUTS0": [],
            "NUTS1": [],
            "NUTS2": [],
            "NUTS3": [],
        }

        for beta in range(n_of_param_samples):
            print(f"start random sampling for beta {beta}...")
            if (
                beta > 0
            ):  # otherwise, the df "posterior_probabilities_selected" has already been created
                posterior_probabilities_selected = posterior_probabilities[
                    posterior_probabilities["beta"] == beta
                ]
                posterior_probabilities_selected.sort_values(
                    by=["lowest_relevant_NUTS_level", "CELLCODE_unique", "crop"],
                    inplace=True,
                )

            random_results = generate_random_results(
                posterior_probabilities_selected, n_of_fields_array, postsampling_reps
            )
            random_results_T = random_results.transpose(1, 0, 2)
            # convert to float16 to save memory and delete previous arrays
            random_results_T_float16 = random_results_T.astype("float16")
            del random_results
            del random_results_T

            for i in range(postsampling_reps):
                sample_accepted = 1
                for j in range(4):
                    if j > (nuts_level_country_year):
                        deviation_dict[f"NUTS{j}"].append(np.nan)
                        continue
                    (
                        max_relative_negative_deviation,
                        min_relative_deviation,
                        max_relative_deviation,
                    ) = calculate_deviation_from_known_aggregates(
                        random_results_T_float16[i], j
                    )
                    relevant_deviation_measure = max_relative_negative_deviation
                    if j == 0:
                        relevant_deviation_measure = max(
                            max_relative_negative_deviation, max_relative_deviation
                        )
                    deviation_dict[f"NUTS{j}"].append(relevant_deviation_measure)
                    if relevant_deviation_measure > deviation_tolerance:
                        sample_accepted = 0
                deviation_dict["accepted"].append(sample_accepted)
                deviation_dict["beta"].append(beta)

            # save array and free up memory space --- will be loaded later again to create final parquet file
            print("save data...")
            Path(temporary_file_path).mkdir(parents=True, exist_ok=True)
            np.save(
                temporary_file_path + "beta" + str(beta) + ".npy",
                random_results_T_float16,
            )
            del random_results_T_float16
            gc.collect()
        deviation_df = pd.DataFrame(deviation_dict)
        # %%

        nuts1_regions_relevant = np.sort(
            np.array(
                nuts_regions_relevant[nuts_regions_relevant["LEVL_CODE"] == 1][
                    "NUTS_ID"
                ]
            )
        )

        for region in nuts1_regions_relevant:
            relevant_cellcodes_unique = ordered_cellcodes_unique[
                np.where(ordered_nuts1_each_cell == region)[0]
            ]
            splitted_cellcodes_unique = np.array(
                list(np.char.split(relevant_cellcodes_unique.astype(str), country))
            ).transpose()
            df_region = pd.DataFrame()
            for beta in range(n_of_param_samples):
                data = np.load(temporary_file_path + "beta" + str(beta) + ".npy")
                if ignore_consistency:
                    n_of_consistent_samples = len(
                        deviation_df[deviation_df["beta"] == beta]
                    )
                else:
                    n_of_consistent_samples = len(
                        np.where(
                            deviation_df[deviation_df["beta"] == beta]["accepted"] == 1
                        )[0]
                    )
                df = pd.DataFrame(
                    {
                        "CELLCODE": np.repeat(
                            splitted_cellcodes_unique[0], n_of_consistent_samples
                        ),
                        "NUTS_ID": np.repeat(
                            np.core.defchararray.add(
                                np.repeat(country, len(splitted_cellcodes_unique[1])),
                                splitted_cellcodes_unique[1],
                            ),
                            n_of_consistent_samples,
                        ),
                    }
                )
                if ignore_consistency:
                    data_selected = data

                else:
                    data_selected = data[
                        np.where(
                            deviation_df[deviation_df["beta"] == beta]["accepted"] == 1
                        )[0]
                    ]
                data_selected = data_selected.transpose(1, 0, 2)[
                    np.where(ordered_nuts1_each_cell == region)[0]
                ]
                cropshares = data_selected.reshape(
                    (data_selected.shape[0] * data_selected.shape[1], -1)
                )
                df["beta"] = np.repeat(beta, len(cropshares))
                # data must be at least float32 for parquet file...
                df_cropshares = pd.DataFrame(cropshares, columns=all_crops).astype(
                    "float32"
                )
                df_region = pd.concat(
                    (df_region, pd.concat((df, df_cropshares), axis=1))
                )

            print(f"export data for {region}...")
            Path(results_postsampling_path + country + "/" +str(year)).mkdir(
                parents=True, exist_ok=True
            )

            df_region.to_parquet(
                results_postsampling_path
                + country
                + "/"
                + str(year)
                + "/"
                + country
                + str(year)
                + "_"
                + region
            )

        # export deviation csv
        deviation_df.to_csv(
            results_postsampling_path
            + country
            + "/"
            + str(year)
            + "/"
            + country
            + str(year)
            + "_deviation_from_aggregated.csv"
        )

        # finally, clean up temporary directory by deleting temporary files...
        for file in os.listdir(temporary_file_path):
            os.remove(temporary_file_path + file)
# %%

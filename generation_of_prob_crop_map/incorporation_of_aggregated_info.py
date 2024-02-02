# %%
import argparse
from cProfile import label
import jax.numpy as jnp  # it works almost exactly like numpy
from jax import grad, jit, vmap, hessian
from jax import random
from jax import jacfwd, jacrev
from jax.numpy import linalg

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
#DEFAULT SETTINGS IN THE OPTIMIZATION
minimize_relative_deviation = True
n_of_param_samples = 10
deviation_tolerance_aggregates = 0.01
deviation_tolerance_cells = 0.001

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
prior_crop_probability_estimates_path=data_main_path+"Results/Prior_crop_probability_estimates/"

UAA_input_path = (
    intermediary_data_path+ "Regional_Aggregates/coherent_UAA_"
)
cropdata_input_path = (
    intermediary_data_path+ "Regional_Aggregates/cropdata_"
)

parameter_path = delineation_and_parameter_path + "DGPCM_user_parameters.xlsx"

crop_level_path=intermediary_data_path+"Regional_Aggregates/crop_levels_selected_countries_"
#
#output paths
output_path=data_main_path+"Results/Posterior_crop_probability_estimates/"
#%%
# import parameters
selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])
# %%
parser = argparse.ArgumentParser()
parser.add_argument("-cc", "--ccode", type=str, required=False)
parser.add_argument("-y", "--year", type=int, required=False)
parser.add_argument("--from_xlsx", type=str, required=False)

args = parser.parse_args()
if args.from_xlsx == "True":
    country_year_list = [np.repeat(country_codes_relevant,len(selected_years)),np.tile(selected_years,len(country_codes_relevant))]
else:
    country_year_list = [[args.ccode], [args.year]]

country_year_list = [np.repeat(country_codes_relevant,len(selected_years)),np.tile(selected_years,len(country_codes_relevant))]



#%%
""" SET UP FUNCTIONS NEEDED IN THE OPTIMIZATION"""

def set_up_optimization_problem(beta, all_crops, init_p_values="prior"):
    reference_df_agg = prepare_prior_info_new(beta, all_crops)
    reference_df_agg.fillna(0, inplace=True)
    reference_df_agg.drop(columns=["beta"], inplace=True)

    reference_df_agg = pd.merge(
        weight_df, reference_df_agg, how="left", on=["NUTS1", "CELLCODE"]
    )

    """construct vector of prior probabilities"""
    p = np.array(reference_df_agg["probability"]).reshape((I, C))
    # verify that probabilities are correct
    print(f"pmin: {p.sum(axis=1).min()}, pmax: {p.sum(axis=1).max()}")
    # ensure that all values for p are slightly larger than 0 to avoid problems when takign logs
    p = np.where(p <= 0.00001, 0.00001, p)
    p = np.array([p_cell / p_cell.sum() for p_cell in p])

    if init_p_values == "prior":
        init_values = p
    else:
        init_values = np.tile(
            (nuts0_cropdata_matrix / nuts0_cropdata_matrix.sum()).flatten(), I
        )

    p_adjusted = np.zeros((newC, I))

    i = 0
    p_conversion_matrix = np.zeros((C, newC))
    for c, crop in enumerate(all_crops):
        crop_freq = len(np.where(relevant_crop_array == crop)[0])
       # init_values_array_crop = init_values.transpose()[c] / crop_freq
        relevant_info_shares_crop=relevant_info_share_array[np.where(relevant_crop_array==crop)[0]]
        relevant_info_shares_crop=relevant_info_shares_crop+0.0001
        relevant_info_shares_crop=relevant_info_shares_crop/relevant_info_shares_crop.sum()
        for j in range(crop_freq):
            p_adjusted[i + j] = init_values.transpose()[c]*relevant_info_shares_crop[j]
            p_conversion_matrix[c][i + j] = 1
        i += j + 1
    p_conversion_matrix = p_conversion_matrix.transpose()

    q = p
    q_log = jnp.log(q).flatten()
    p_adjusted = p_adjusted.transpose()
    p_init = p_adjusted.flatten()
    onevector = jnp.ones(I)
    return (
        p_init,
        p_adjusted,
        q,
        q_log,
        onevector,
        p_conversion_matrix,
        reference_df_agg,
    )

def p_conversion_new(x):
    p_long = x.reshape(I, newC)
    return jnp.matmul(p_long, p_conversion_matrix).flatten()

def run_optimization_problem(maxiter_optimization, relative=True):
    if relative:
        pg_boxsection = ProjectedGradient(
            fun=complete_obj_function_relative,
            projection=projection_box,
            maxiter=maxiter_optimization,
            maxls=200,
            tol=0,  # 1e-2,
            stepsize=-0,
            decrease_factor=0.8,
            implicit_diff=True,
            verbose=False,
        )
    else:
        pg_boxsection = ProjectedGradient(
            fun=complete_obj_function_alternative,
            projection=projection_box,
            maxiter=maxiter_optimization,
            maxls=200,
            tol=0,  # 1e-2,
            stepsize=-0.001,
            decrease_factor=0.8,
            implicit_diff=True,
            verbose=False,
        )

    pg_sol_boxsection = pg_boxsection.run(p_init, hyperparams_proj=proj_params)
    return pg_sol_boxsection

def nuts3_crop_constraint_alternative(x):
    x_reshaped = x.reshape(I, newC).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    weighted_aggregated_crops = jnp.matmul(M[3], weighted_crops)
    return jnp.square(
        jnp.matmul(
            weighted_aggregated_crops,
            nuts3_regions_construction_matrix_alternative,
        )
        - nuts3_cropdata_matrix.transpose()
    ).sum()

def nuts3_crop_constraint_relative(x):
    x_reshaped = x.reshape(I, newC).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    weighted_aggregated_crops = jnp.matmul(M[3], weighted_crops)

    return jnp.square(
        jnp.multiply(
            (
                jnp.matmul(
                    weighted_aggregated_crops,
                    nuts3_regions_construction_matrix_alternative,
                )
                - nuts3_cropdata_matrix.transpose()
            ),
            UAA_nuts3_inverse,
        )
        * penalty_nuts3_rel
    ).sum()

def nuts2_crop_constraint_alternative(x):
    x_reshaped = x.reshape(I, newC).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    weighted_aggregated_crops = jnp.matmul(M[2], weighted_crops)
    return jnp.square(
        jnp.matmul(
            weighted_aggregated_crops,
            nuts2_regions_construction_matrix_alternative,
        )
        - nuts2_cropdata_matrix.transpose()
    ).sum()

def nuts2_crop_constraint_relative(x):
    x_reshaped = x.reshape(I, newC).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    weighted_aggregated_crops = jnp.matmul(M[2], weighted_crops)

    return jnp.square(
        jnp.multiply(
            (
                jnp.matmul(
                    weighted_aggregated_crops,
                    nuts2_regions_construction_matrix_alternative,
                )
                - nuts2_cropdata_matrix.transpose()
            ),
            UAA_nuts2_inverse,
        )
        * penalty_nuts2_rel
    ).sum()

def nuts1_crop_constraint_alternative(x):
    x_reshaped = x.reshape(I, newC).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    weighted_aggregated_crops = jnp.matmul(M[1], weighted_crops)
    return jnp.square(
        jnp.matmul(
            weighted_aggregated_crops,
            nuts1_regions_construction_matrix_alternative,
        )
        - nuts1_cropdata_matrix.transpose()
    ).sum()

def nuts1_crop_constraint_relative(x):
    x_reshaped = x.reshape(I, newC).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    weighted_aggregated_crops = jnp.matmul(M[1], weighted_crops)

    return jnp.square(
        jnp.multiply(
            (
                jnp.matmul(
                    weighted_aggregated_crops,
                    nuts1_regions_construction_matrix_alternative,
                )
                - nuts1_cropdata_matrix.transpose()
            ),
            UAA_nuts1_inverse,
        )
        * penalty_nuts1_rel
    ).sum()

def nuts0_crop_constraint_alternative(x):
    x_reshaped = x.reshape(I, newC).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    weighted_aggregated_crops = jnp.matmul(M[0], weighted_crops)
    return jnp.square(
        jnp.matmul(
            weighted_aggregated_crops,
            nuts0_regions_construction_matrix_alternative,
        )
        - nuts0_cropdata_matrix.transpose()
    ).sum()

def nuts0_crop_constraint_relative(x):
    x_reshaped = x.reshape(I, newC).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    weighted_aggregated_crops = jnp.matmul(M[0], weighted_crops)

    return jnp.square(
        jnp.multiply(
            (
                jnp.matmul(
                    weighted_aggregated_crops,
                    nuts0_regions_construction_matrix_alternative,
                )
                - nuts0_cropdata_matrix.transpose()
            ),
            UAA_nuts0_inverse,
        )
        * penalty_nuts0_rel
    ).sum()

def cell_constraint(x):
    x = x.reshape(I, newC)
    return jnp.square(x.sum(axis=1) - onevector).sum()

def obj_function_new(x):
    p = p_conversion_new(x)

    p = jax.nn.relu(p) + 0.0000001
    # return (jnp.dot(-(p * q_log) + p * jnp.log(p), jnp.ones_like(p)))
    return -(jnp.dot(p * q_log - p * jnp.log(p), weight_array_long_C))
    # return -((q_log*p).sum()-(jnp.log(p)*p).sum())

def combined_constraints_lowestlevel3_alternative(x):
    return (
        nuts3_crop_constraint_alternative(x) * penalty_nuts3
        + nuts2_crop_constraint_alternative(x) * penalty_nuts2
        + nuts1_crop_constraint_alternative(x) * penalty_nuts1
        + nuts0_crop_constraint_alternative(x) * penalty_nuts0
        + cell_constraint(x) * penalty_cell
    )

def combined_constraints_lowestlevel3_relative(x):
    return (
        nuts3_crop_constraint_relative(x)
        + nuts2_crop_constraint_relative(x)
        + nuts1_crop_constraint_relative(x)
        + nuts0_crop_constraint_relative(x)
        + cell_constraint(x) * penalty_cell
    )

def combined_constraints_lowestlevel2_alternative(x):
    return (
        nuts2_crop_constraint_alternative(x) * penalty_nuts2
        + nuts1_crop_constraint_alternative(x) * penalty_nuts1
        + nuts0_crop_constraint_alternative(x) * penalty_nuts0
        + cell_constraint(x) * penalty_cell
    )

def combined_constraints_lowestlevel2_relative(x):
    return (
        nuts2_crop_constraint_relative(x)
        + nuts1_crop_constraint_relative(x)
        + nuts0_crop_constraint_relative(x)
        + cell_constraint(x) * penalty_cell
    )

def combined_constraints_lowestlevel1_alternative(x):
    return (
        nuts1_crop_constraint_alternative(x) * penalty_nuts1
        + nuts0_crop_constraint_alternative(x) * penalty_nuts0
        + cell_constraint(x) * penalty_cell
    )

def combined_constraints_lowestlevel1_relative(x):
    return (
        nuts1_crop_constraint_relative(x)
        + nuts0_crop_constraint_relative(x)
        + cell_constraint(x) * penalty_cell
    )

def combined_constraints_lowestlevel0_alternative(x):
    return (
        nuts0_crop_constraint_alternative(x) * penalty_nuts0
        + cell_constraint(x) * penalty_cell
    )

def combined_constraints_lowestlevel0_relative(x):
    return nuts0_crop_constraint_relative(x) + cell_constraint(x) * penalty_cell

def nuts_crop_deviation_alternative(x, level):
    x_reshaped = x.reshape(I, newC).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    weighted_aggregated_crops = jnp.matmul(M[level], weighted_crops)
    if level == 3:
        nuts_region_construction_matrix = (
            nuts3_regions_construction_matrix_alternative
        )
        nuts_cropdata_matrix = nuts3_cropdata_matrix

    if level == 2:
        nuts_region_construction_matrix = (
            nuts2_regions_construction_matrix_alternative
        )
        nuts_cropdata_matrix = nuts2_cropdata_matrix

    if level == 1:
        nuts_region_construction_matrix = (
            nuts1_regions_construction_matrix_alternative
        )
        nuts_cropdata_matrix = nuts1_cropdata_matrix

    if level == 0:
        nuts_region_construction_matrix = (
            nuts0_regions_construction_matrix_alternative
        )
        nuts_cropdata_matrix = nuts0_cropdata_matrix

    return (
        jnp.matmul(weighted_aggregated_crops, nuts_region_construction_matrix)
        - nuts_cropdata_matrix.transpose()
    )

def nuts_crop_area(x, level):
    x_reshaped = x.reshape(I, newC).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    weighted_aggregated_crops = jnp.matmul(M[level], weighted_crops)
    if level == 3:
        nuts_region_construction_matrix = (
            nuts3_regions_construction_matrix_alternative
        )

    if level == 2:
        nuts_region_construction_matrix = (
            nuts2_regions_construction_matrix_alternative
        )

    if level == 1:
        nuts_region_construction_matrix = (
            nuts1_regions_construction_matrix_alternative
        )

    if level == 0:
        nuts_region_construction_matrix = (
            nuts0_regions_construction_matrix_alternative
        )

    return jnp.matmul(
        weighted_aggregated_crops, nuts_region_construction_matrix
    )

def max_cell_deviation(x):
    x = x.reshape(I, newC)
    return jnp.abs(x.sum(axis=1) - onevector).max()

def evaluate_quality_alternative(x, level):
    max_dev_list = []
    nuts0_UAA = calculate_level_UAA(0)
    nuts0_dev = nuts_crop_deviation_alternative(x, 0).transpose()
    max_dev = np.abs(
        [nuts0_dev[i] * UAA_nuts0_inverse[i] for i in range(len(nuts0_UAA))]
    ).max()
    print(f"maximal relative_deviation at nuts0 level: {max_dev}")
    max_dev_list.append(max_dev)
    if level > 0:
        nuts1_UAA = calculate_level_UAA(1)
        nuts1_dev = nuts_crop_deviation_alternative(x, 1).transpose()
        max_dev = np.abs(
            [nuts1_dev[i] * UAA_nuts1_inverse[i] for i in range(len(nuts1_UAA))]
        ).max()
        print(f"maximal relative_deviation at nuts1 level: {max_dev}")
        max_dev_list.append(max_dev)
    if level > 1:
        nuts2_UAA = calculate_level_UAA(2)
        nuts2_dev = nuts_crop_deviation_alternative(x, 2).transpose()
        max_dev = np.abs(
            [nuts2_dev[i] * UAA_nuts2_inverse[i] for i in range(len(nuts2_UAA))]
        ).max()
        print(f"maximal relative_deviation at nuts2 level: {max_dev}")
        max_dev_list.append(max_dev)
    if level > 2:
        nuts3_UAA = calculate_level_UAA(3)
        nuts3_dev = nuts_crop_deviation_alternative(x, 3).transpose()
        max_dev = np.abs(
            [nuts3_dev[i] * UAA_nuts3_inverse[i] for i in range(len(nuts3_UAA))]
        ).max()
        print(f"maximal relative_deviation at nuts3 level: {max_dev}")
        max_dev_list.append(max_dev)

    max_cell_dev = max_cell_deviation(x)
    print(f"maximal cell deviation: {max_cell_dev}")

    r2 = r2_score(q.flatten(), p_conversion_new(x))
    print(f"R2 score: {r2}")
    return max_dev_list, max_cell_dev, r2

def prepare_prior_info_new(beta, relevant_crops):
    """IMPORT AND STRUCTURE PRIOR INFO"""
    reference_df_agg = pd.DataFrame()

    for nuts1 in nuts1_unique:
        nuts1_priorprob_path = priorprob_path + nuts1 + "_" + str(year)
        beta_selected = f"betarand{beta}"

        reference_df = ffp.calculate_reference_df(
            nuts1_priorprob_path, beta_selected, relevant_crops
        )

        reference_df["NUTS1"] = np.repeat(nuts1, len(reference_df))

        reference_df_agg = pd.concat((reference_df_agg, reference_df))

    return reference_df_agg

def calculate_level_UAA(level):
    level_UAA = relevant_UAA[["NUTS_ID", "UAA_in_ha"]].copy()
    level_UAA["level_NUTS_ID"] = np.array(level_UAA["NUTS_ID"]).astype(
        f"U{2+level}"
    )
    level_UAA = level_UAA.groupby("level_NUTS_ID").sum().reset_index()
    return np.array(level_UAA["UAA_in_ha"])

def create_region_construction_matrix(level):
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
        weight_df["lowest_relevant_NUTS_level"]
    ).astype(f"U{level+2}")
    for r, region in enumerate(regions):
        region_construction_matrix[r][
            np.where(weight_df_regions == region)[0]
        ] = 1
    region_construction_matrix = np.repeat(
        region_construction_matrix, newC, axis=1
    )
    return region_construction_matrix

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
        weight_df["lowest_relevant_NUTS_level"]
    ).astype(f"U{level+2}")
    for r, region in enumerate(regions):
        region_construction_matrix[r][
            np.where(weight_df_regions == region)[0]
        ] = 1
    return region_construction_matrix

def create_level_crop_weight_matrix(level):
    # create a matrix of shape (newC x newC*I) which indicates for each crop if the respective crop is relevant at the regional level
    # and where the p_init entries are that indicate the respective crop
    diag = np.zeros(newC)
    diag[np.where(relevant_level_array >= level)[0]] = 1
    level_crop_construction_matrix = np.tile(np.diag(diag), I)

    # multiply the weight array with the level_crop_construction_matrix element-wise to receive a (newC*I x newC) matrix which indicates the
    # weight of the respective cell for those crops that are relevant at the regional level
    level_crop_weight_matrix = np.multiply(
        weight_array_long_newC, level_crop_construction_matrix
    ).transpose()
    return level_crop_weight_matrix

def adjust_penalty(
    max_dev_list,
    deviation_tolerance_aggregates,
    max_cell_dev,
    deviation_tolerance_cells,
    penalty_nuts0_rel,
    penalty_nuts1_rel,
    penalty_nuts2_rel,
    penalty_nuts3_rel,
    penalty_cell,
    nuts_level_country_year,
):
    # set default for penalty adjusted to false and only change if adjustment is sufficient
    penalty_adjusted = False
    if (np.max(max_dev_list) <= deviation_tolerance_aggregates) & (
        max_cell_dev < deviation_tolerance_cells
    ):
        penalty_adjusted = True
    elif (np.max(max_dev_list) <= deviation_tolerance_aggregates) & (
        max_cell_dev > deviation_tolerance_cells
    ):
        penalty_cell = penalty_cell * 5

    elif (
        len(
            np.where(np.array(max_dev_list) > deviation_tolerance_aggregates)[0]
        )
        < nuts_level_country_year
    ):
        increase_penalty_array = np.ones(4)
        increase_penalty_array[
            np.where(np.array(max_dev_list) > deviation_tolerance_aggregates)[0]
        ] = 5
        (
            penalty_nuts0_rel,
            penalty_nuts1_rel,
            penalty_nuts2_rel,
            penalty_nuts3_rel,
        ) = (
            penalty_nuts0_rel * increase_penalty_array[0],
            penalty_nuts1_rel * increase_penalty_array[1],
            penalty_nuts2_rel * increase_penalty_array[2],
            penalty_nuts3_rel * increase_penalty_array[3],
        )
        penalty_cell = penalty_cell * 5

    else:
        penalty_cell = penalty_cell * 10
        (
            penalty_nuts0_rel,
            penalty_nuts1_rel,
            penalty_nuts2_rel,
            penalty_nuts3_rel,
        ) = (
            penalty_nuts0_rel * 10,
            penalty_nuts1_rel * 10,
            penalty_nuts2_rel * 10,
            penalty_nuts3_rel * 10,
        )

    return (
        penalty_nuts0_rel,
        penalty_nuts1_rel,
        penalty_nuts2_rel,
        penalty_nuts3_rel,
        penalty_cell,
        penalty_adjusted,
    )


#%%
UAA = pd.read_csv(UAA_input_path+str(country_year_list[1].min())+str(country_year_list[1].max())+".csv")
UAA["NUTS_LEVL"]=np.char.str_len(np.array(UAA["NUTS_ID"]).astype(str))-2
cropdata = pd.read_csv(cropdata_input_path+str(country_year_list[1].min())+str(country_year_list[1].max())+".csv")
cropdata["NUTS_LEVL"]=np.char.str_len(np.array(cropdata["NUTS_ID"]).astype(str))-2
nuts_input = pd.read_csv(nuts_path)
#%%

crop_levels=pd.read_csv(crop_level_path+str(country_year_list[1].min())+str(country_year_list[1].max())+".csv")
#crop_levels=pd.read_csv("/home/baumert/research/Project-1/data/Intermediary Data/Eurostat/optimization_constraints/crop_levels_20102020_final.csv")

#%%
crop_levels
#%%

if __name__ == "__main__":
    for c in range(len(country_year_list[0]))[:1]:
        country = country_year_list[0][c]
        year = country_year_list[1][c]
        print(f"starting optimization for {country} in {year}...")

        

        priorprob_path = (
            prior_crop_probability_estimates_path + country + "/"
        )
        cellsize_input_path = (
            intermediary_data_path
            + "/Zonal_Stats/"
            + country
            + "/inferred_UAA/1kmgrid_"
        )
 
        cell_weight_path = (
            intermediary_data_path
            + "Regional_Aggregates/Cell_Weights/"
            + country
            + "/cell_weights_"
            +str(country_year_list[1].min())+str(country_year_list[1].max())
            +".csv"
        )

        


        # %%
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # %%
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
            n_of_param_samples = other_settings["n_of_param_samples"].iloc[0]
            deviation_tolerance_aggregates = other_settings["deviation_tolerance"].iloc[
                0
            ]
            deviation_tolerance_cells = other_settings[
                "deviation_tolerance_cells"
            ].iloc[0]
        except:
            pass
        """import data"""

        cell_weight_info = pd.read_csv(cell_weight_path)

        # %%
        cropdata_relevant=cropdata[(cropdata["country"]==country)&(cropdata["year"]==year)]
        UAA_relevant=UAA[(UAA["country"]==country)&(UAA["year"]==year)]
        
        #%%
        UAA_relevant["NUTS_LEVL"] = np.vectorize(len)(UAA_relevant["NUTS_ID"]) - 2
        cropdata_relevant["NUTS_LEVL"] = np.vectorize(len)(cropdata_relevant["NUTS_ID"]) - 2

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

        nuts1_unique = np.sort(
            np.array(
                nuts_regions_relevant[nuts_regions_relevant["LEVL_CODE"] == 1][
                    "NUTS_ID"
                ]
            )
        )

        nuts1_unique = nuts1_unique[
            np.where(np.isin(nuts1_unique, excluded_NUTS_regions).astype(int) == 0)[0]
        ]
               
        #%%
        """find out at which level data on crop area is available for respective country and year"""
        
        relevant_crop_levels = crop_levels[
            (crop_levels["country"] == country) & (crop_levels["year"] == year)
        ]
        #%%
        #find the lowest NUTS level of the respective year and country for which data is available
        if relevant_crop_levels["NUTS3"].sum()>0:
            lowest_level_country_year=3
        elif relevant_crop_levels["NUTS2"].sum()>0:
            lowest_level_country_year=2
        elif relevant_crop_levels["NUTS1"].sum()>0:
            lowest_level_country_year=1
        else:
            lowest_level_country_year=0
        
        #%%
        relevant_crop_array, relevant_level_array, relevant_info_share_array = [], [], []
        for crop in all_crops:
            selection = relevant_crop_levels[relevant_crop_levels["crop"] == crop]
            # if a crop has 0 values for all levels (i.e., if it doesn't appear in a country):
            if selection.iloc[0, -4:].sum() == 0:
                relevant_crop_array.append(crop)
                relevant_level_array.append(lowest_level_country_year)
                relevant_info_share_array.append(1.)

            write_to_level = False
            for i in range(lowest_level_country_year + 1):
                if selection[f"NUTS{i}"].iloc[0] > 0:
                    write_to_level = True
                if write_to_level:
                    relevant_crop_array.append(crop)
                    relevant_level_array.append(i)
                    relevant_info_share_array.append(selection[f"{i}level_information_share"].iloc[0])

        relevant_crop_array = np.array(relevant_crop_array)
        relevant_level_array = np.array(relevant_level_array)
        relevant_info_share_array=np.array(relevant_info_share_array)
        
        
        #%%
        C = len(all_crops)
        newC = len(relevant_crop_array)
        nuts_level_country_year = relevant_level_array.max()
        if "NUTS1" in np.array(cell_weight_info.columns):
            weight_df = cell_weight_info[(cell_weight_info["year"] == year)][
                [
                    "NUTS1",
                    "lowest_relevant_NUTS_level",
                    "CELLCODE",
                    "CELLCODE_unique",
                    "weight",
                ]
            ]
        else:
            weight_df = cell_weight_info[(cell_weight_info["year"] == year)][
                ["lowest_relevant_NUTS_level", "CELLCODE", "CELLCODE_unique", "weight"]
            ]
        weight_df.drop_duplicates("CELLCODE_unique", inplace=True)
        weight_df = weight_df[weight_df["weight"] > 0]
        weight_df.sort_values(
            by=["lowest_relevant_NUTS_level", "CELLCODE"], inplace=True
        )
        if not "NUTS1" in np.array(weight_df.columns):
            weight_df["NUTS1"] = np.array(
                weight_df["lowest_relevant_NUTS_level"]
            ).astype("U3")
        weight_array = np.array(weight_df["weight"])
        weight_array_long_newC = np.repeat(weight_array, newC)
        weight_array_long_C = np.repeat(weight_array, C)

        # %%

        aggregated_crop_conversion_matrix = np.zeros((C, newC))
        for c, crop in enumerate(all_crops):
            aggregated_crop_conversion_matrix[c][
                np.where(relevant_crop_array == crop)[0]
            ] = 1
        aggregated_crop_conversion_matrix = (
            aggregated_crop_conversion_matrix.transpose()
        )

        cellcode_unique_array = np.array(
            weight_df["CELLCODE_unique"].value_counts().keys()
        )
        I = len(cellcode_unique_array)

        # build other functions needed for the optimization that depend on the nuts_level_country_year
        if nuts_level_country_year == 3:

            def complete_obj_function_alternative(x):
                return jnp.log(
                    obj_function_new(x) * obj_function_factor
                    + combined_constraints_lowestlevel3_alternative(x)
                )

            def complete_obj_function_relative(x):
                return jnp.log(
                    obj_function_new(x) * obj_function_factor
                    + combined_constraints_lowestlevel3_relative(x)
                )

        if nuts_level_country_year == 2:

            def complete_obj_function_alternative(x):
                return jnp.log(
                    obj_function_new(x) * obj_function_factor
                    + combined_constraints_lowestlevel2_alternative(x)
                )

            def complete_obj_function_relative(x):
                return jnp.log(
                    obj_function_new(x) * obj_function_factor
                    + combined_constraints_lowestlevel2_relative(x)
                )

        if nuts_level_country_year == 1:

            def complete_obj_function_alternative(x):
                return jnp.log(
                    obj_function_new(x) * obj_function_factor
                    + combined_constraints_lowestlevel1_alternative(x)
                )

            def complete_obj_function_relative(x):
                return jnp.log(
                    obj_function_new(x) * obj_function_factor
                    + combined_constraints_lowestlevel1_relative(x)
                )

        if nuts_level_country_year == 0:

            def complete_obj_function_alternative(x):
                return jnp.log(
                    obj_function_new(x) * obj_function_factor
                    + combined_constraints_lowestlevel0_alternative(x)
                )

            def complete_obj_function_relative(x):
                return jnp.log(
                    obj_function_new(x) * obj_function_factor
                    + combined_constraints_lowestlevel0_relative(x)
                )

        # build other matrices needed in the optimization
        M = np.zeros((nuts_level_country_year + 1, C, newC))
        for l in np.arange(nuts_level_country_year + 1):
            print(l)
            for c, crop in enumerate(all_crops):
                for j in np.where(
                    (relevant_crop_array == crop) & (relevant_level_array >= l)
                )[0]:
                    M[l][c][j] = 1

        
        #%%
        # prepare data on UAA
        relevant_UAA = UAA[
            (UAA["country"] == country)
            & (UAA["year"] == year)
            & (UAA["NUTS_LEVL"] == nuts_level_country_year)
        ]

        # write UAA in ha
        relevant_UAA["UAA_in_ha"] = relevant_UAA["UAA_corrected"] * 1000

        """preprare crop data"""
        cropdata_relevant = cropdata[
            (cropdata["country"] == country) & (cropdata["year"] == year)
        ]

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
        #%%

    
        """ RUN OPTIMIZATION PROBLEM
        for the first run of each country and year the penalties and max iter are manually adjusted until a satisfactory result is received. 
        the following runs (different parameter values) of the same country and year are run with the same penalties and max iter
        """

        """Load and preprocess prior crop probabilities"""
        beta = 0
        (
            p_init,
            p_adjusted,
            q,
            q_log,
            onevector,
            p_conversion_matrix,
            reference_df_agg,
        ) = set_up_optimization_problem(beta, all_crops)

        
        
        
        
        #%%
        penalty_adjusted = False

        penalty_cell, obj_function_factor = 10e2, 0.01
        penalty_nuts3_rel, penalty_nuts2_rel, penalty_nuts1_rel, penalty_nuts0_rel = (
            1,
            1,
            1,
            1,
        )
        # %%

        while not penalty_adjusted:
            box_lower = jnp.zeros_like(p_init)
            box_upper = jnp.ones_like(p_init)
            proj_params = (box_lower, box_upper)

            max_iter = 5000
            result = run_optimization_problem(
                maxiter_optimization=max_iter, relative=True
            )

            p_result = result.params

            max_dev_list, max_cell_dev, r2 = evaluate_quality_alternative(
                p_result, nuts_level_country_year
            )
            (
                max_dev_list_init,
                max_cell_dev_init,
                r2_init,
            ) = evaluate_quality_alternative(p_init, nuts_level_country_year)
            print(f"[---{country}--{year}---]")

            if (
                np.array(
                    (
                        penalty_cell,
                        penalty_nuts3_rel,
                        penalty_nuts2_rel,
                        penalty_nuts1_rel,
                        penalty_nuts0_rel,
                    )
                ).max()
                < 10e10
            ):  # to avoid numerical overflow
                (
                    penalty_nuts0_rel,
                    penalty_nuts1_rel,
                    penalty_nuts2_rel,
                    penalty_nuts3_rel,
                    penalty_cell,
                    penalty_adjusted,
                ) = adjust_penalty(
                    max_dev_list,
                    deviation_tolerance_aggregates,
                    max_cell_dev,
                    deviation_tolerance_cells,
                    penalty_nuts0_rel,
                    penalty_nuts1_rel,
                    penalty_nuts2_rel,
                    penalty_nuts3_rel,
                    penalty_cell,
                    nuts_level_country_year,
                )
            else:
                penalty_adjusted = True

        # %%
        p_final = p_conversion_new(p_result)

        optimization_hyperparameter_results_dict = {
            "minimize_relative_deviation": [],
            "beta": [],
            "max_iter": [],
            "penalty_nuts3_rel": [],
            "penalty_nuts2_rel": [],
            "penalty_nuts1_rel": [],
            "penalty_nuts0_rel": [],
            "penalty_cell": [],
            "max_dev_nuts3": [],
            "max_dev_nuts2": [],
            "max_dev_nuts1": [],
            "max_dev_nuts0": [],
            "max_dev_nuts3_init": [],
            "max_dev_nuts2_init": [],
            "max_dev_nuts1_init": [],
            "max_dev_nuts0_init": [],
            "max_cell_dev": [],
            "r2": [],
        }

        optimization_hyperparameter_results_dict["minimize_relative_deviation"].append(
            minimize_relative_deviation
        )
        optimization_hyperparameter_results_dict["beta"].append(beta)
        optimization_hyperparameter_results_dict["max_iter"].append(max_iter)
        optimization_hyperparameter_results_dict["penalty_nuts3_rel"].append(
            penalty_nuts3_rel
        )
        optimization_hyperparameter_results_dict["penalty_nuts2_rel"].append(
            penalty_nuts2_rel
        )
        optimization_hyperparameter_results_dict["penalty_nuts1_rel"].append(
            penalty_nuts1_rel
        )
        optimization_hyperparameter_results_dict["penalty_nuts0_rel"].append(
            penalty_nuts0_rel
        )
        optimization_hyperparameter_results_dict["penalty_cell"].append(penalty_cell)
        for i, max_dev in enumerate(max_dev_list):
            optimization_hyperparameter_results_dict[f"max_dev_nuts{i}"].append(max_dev)
            optimization_hyperparameter_results_dict[f"max_dev_nuts{i}_init"].append(
                max_dev_list_init[i]
            )
        for j in range(3 - i):
            optimization_hyperparameter_results_dict[f"max_dev_nuts{i+j+1}"].append(
                np.nan
            )
            optimization_hyperparameter_results_dict[
                f"max_dev_nuts{i+j+1}_init"
            ].append(np.nan)
        optimization_hyperparameter_results_dict["max_cell_dev"].append(max_cell_dev)
        optimization_hyperparameter_results_dict["r2"].append(r2)

        results_df = reference_df_agg.copy()
        results_df.rename(columns={"probability": "prior_probability"}, inplace=True)
        results_df["posterior_probability"] = p_final
        results_df["beta"] = np.repeat(beta, len(p_final))
        all_results_df = results_df.copy()

        min_penalty_nuts0_rel = penalty_nuts0_rel
        min_penalty_nuts1_rel = penalty_nuts1_rel
        min_penalty_nuts2_rel = penalty_nuts2_rel
        min_penalty_nuts3_rel = penalty_nuts3_rel
        min_penalty_cell = penalty_cell

        for beta in range(1, n_of_param_samples):
            penalty_nuts0_rel = min_penalty_nuts0_rel
            penalty_nuts1_rel = min_penalty_nuts1_rel
            penalty_nuts2_rel = min_penalty_nuts2_rel
            penalty_nuts3_rel = min_penalty_nuts3_rel
            penalty_cell = min_penalty_cell
            print(f"solving optimization problem for beta {beta}")
            (
                p_init,
                p_adjusted,
                q,
                q_log,
                onevector,
                p_conversion_matrix,
                reference_df_agg,
            ) = set_up_optimization_problem(beta, all_crops)
            penalty_adjusted = False
            while not penalty_adjusted:
                result = run_optimization_problem(maxiter_optimization=max_iter)
                p_result = result.params
                p_final = p_conversion_new(p_result)
                max_dev_list, max_cell_dev, r2 = evaluate_quality_alternative(
                    p_result, nuts_level_country_year
                )
                (
                    max_dev_list_init,
                    max_cell_dev_init,
                    r2_init,
                ) = evaluate_quality_alternative(p_init, nuts_level_country_year)

                if (
                    np.array(
                        (
                            penalty_cell,
                            penalty_nuts3_rel,
                            penalty_nuts2_rel,
                            penalty_nuts1_rel,
                            penalty_nuts0_rel,
                        )
                    ).max()
                    < 10e10
                ):  # to avoid numerical overflow
                    (
                        penalty_nuts0_rel,
                        penalty_nuts1_rel,
                        penalty_nuts2_rel,
                        penalty_nuts3_rel,
                        penalty_cell,
                        penalty_adjusted,
                    ) = adjust_penalty(
                        max_dev_list,
                        deviation_tolerance_aggregates,
                        max_cell_dev,
                        deviation_tolerance_cells,
                        penalty_nuts0_rel,
                        penalty_nuts1_rel,
                        penalty_nuts2_rel,
                        penalty_nuts3_rel,
                        penalty_cell,
                        nuts_level_country_year,
                    )
                else:
                    penalty_adjusted = True

            optimization_hyperparameter_results_dict[
                "minimize_relative_deviation"
            ].append(minimize_relative_deviation)
            optimization_hyperparameter_results_dict["beta"].append(beta)
            optimization_hyperparameter_results_dict["max_iter"].append(max_iter)
            optimization_hyperparameter_results_dict["penalty_nuts3_rel"].append(
                penalty_nuts3_rel
            )
            optimization_hyperparameter_results_dict["penalty_nuts2_rel"].append(
                penalty_nuts2_rel
            )
            optimization_hyperparameter_results_dict["penalty_nuts1_rel"].append(
                penalty_nuts1_rel
            )
            optimization_hyperparameter_results_dict["penalty_nuts0_rel"].append(
                penalty_nuts0_rel
            )
            optimization_hyperparameter_results_dict["penalty_cell"].append(
                penalty_cell
            )
            for i, max_dev in enumerate(max_dev_list):
                optimization_hyperparameter_results_dict[f"max_dev_nuts{i}"].append(
                    max_dev
                )
                optimization_hyperparameter_results_dict[
                    f"max_dev_nuts{i}_init"
                ].append(max_dev_list_init[i])
            for j in range(3 - i):
                optimization_hyperparameter_results_dict[f"max_dev_nuts{i+j+1}"].append(
                    np.nan
                )
                optimization_hyperparameter_results_dict[
                    f"max_dev_nuts{i+j+1}_init"
                ].append(np.nan)
            optimization_hyperparameter_results_dict["max_cell_dev"].append(
                max_cell_dev
            )
            optimization_hyperparameter_results_dict["r2"].append(r2)

            results_df = reference_df_agg.copy()
            results_df.rename(
                columns={"probability": "prior_probability"}, inplace=True
            )
            results_df["posterior_probability"] = p_final
            results_df["beta"] = np.repeat(beta, len(p_final))
            all_results_df = pd.concat((all_results_df, results_df))
        # %%
        optimization_hyperparameter_results_df = pd.DataFrame(
            optimization_hyperparameter_results_dict
        )
       

        # %%
        Path(output_path + country + "/").mkdir(parents=True, exist_ok=True)
        all_results_df.to_parquet(
            output_path + country + "/" + country + str(year) + "entire_country"
        )

        optimization_hyperparameter_results_df.to_csv(
            output_path
            + country
            + "/"
            + country
            + str(year)
            + "entire_country_hyperparameters.csv"
        )
# %%

# %%
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pyarrow as pa
import pyarrow.parquet as pq
import math
from joblib import load
from sklearn.neighbors import KernelDensity
import joblib
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

import modules.functions_for_prediction as fufop
#%%

# %%
LUCAS_observation_inclusion_threshold=20
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
feature_input_path=intermediary_data_path+"Zonal_Stats/"
nuts_path=intermediary_data_path+"Preprocessed_Inputs/NUTS/NUTS_all_regions_all_years.csv"
excluded_NUTS_regions_path=delineation_and_parameter_path+ "excluded_NUTS_regions.xlsx"
regression_param_path = estimated_parameter_path+ "multinomial_logit_"
scaler_path = estimated_parameter_path+"scale_factors/standardscaler_multinom_logit_"

#
output_path=data_main_path+"Results/Prior_crop_probability_estimates/"

# import parameters
selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])


#%%
parser = argparse.ArgumentParser()
parser.add_argument("-cc", "--ccode", type=str, required=False)
parser.add_argument("-y", "--year", type=int, required=False)
parser.add_argument("--from_xlsx", type=str, required=False)

args = parser.parse_args()
if args.from_xlsx == "True":
    country_year_list = [np.repeat(country_codes_relevant,len(selected_years)),np.tile(selected_years,len(country_codes_relevant))]
else:
    country_year_list = [[args.ccode], [args.year]]
#%%
country_year_list = [np.repeat(country_codes_relevant,len(selected_years)),np.tile(selected_years,len(country_codes_relevant))]
country_year_list
# training_countries = ["UK"]
# reference_crop_estimation = "APPL+OFRU"
#%%

excluded_nuts_regions = pd.read_excel(
    excluded_NUTS_regions_path
)  # some nuts regions are excluded (mainly France overseas)
excluded_nuts_regions = np.array(
    excluded_nuts_regions["excluded NUTS1 regions"]
)
nuts_input = pd.read_csv(nuts_path)
#%%
# params_from_sample = False  # if True parameteres are drawn from a random distribution, if False the parameter mean is taken
# testruns = False
nuts_level = 1
# training_countries_str = "_".join(training_countries)
if __name__ == "__main__":
    for c in range(len(country_year_list[0])):
        country = country_year_list[0][c]
        year = country_year_list[1][c]
        print(f"start prediction for {country} in {year}")
        # input files
       
        cellsize_input_path = feature_input_path+country + "/inferred_UAA/1kmgrid_"



        # specify features to be used in the prediction
        feature_name_dict = {
            "elevation": "elevation",
            "sand_content": "sand",
            "clay_content": "clay",
            "silt_content": "silt",
            "organic_carbon": "oc_content",
            "slope_in_degree": "slope",
            "avg_annual_veg_period": "avg_annual_veg_period",
            "avg_annual_precipitation": "avg_annual_precipitation",
            "avg_annual_temp_sum": "avg_annual_temp_sum",
            "latitude4326": "latitude4326",
            "bulk_density": "bulk_density",
            "coarse_fragments": "coarse_fragments",
            "awc": "awc",
        }

        climate_features = [
            "avg_annual_veg_period",
            "avg_annual_precipitation",
            "avg_annual_temp_sum",
        ]

        """ import parameters"""
        corine_info = pd.read_excel(parameter_path, sheet_name="CORINE")
        corine_year = corine_info[corine_info["crop_map_year"] == year][
            "corine_year"
        ].iloc[0]
        nuts_info = pd.read_excel(parameter_path, sheet_name="NUTS")
        nuts_boundary_year = int(nuts_info[nuts_info["crop_map_year"] == year]["nuts_year"].iloc[0])

        crop_info = pd.read_excel(parameter_path, sheet_name="crops")
        all_crops = np.array(crop_info["crop"])

        levl_info = pd.read_excel(parameter_path, sheet_name="lowest_agg_level")
        country_levls = {}
        for i in range(len(levl_info)):
            country_levls[levl_info.iloc[i].country_code] = levl_info.iloc[
                i
            ].lowest_agg_level
        # lowest_level_country_year_info = pd.read_csv(lowest_level_country_year_path)


        training_country_info = pd.read_excel(
            parameter_path, sheet_name="parameter_training_countries"
        )
        training_country = training_country_info[
            training_country_info["country"] == country
        ]["training_country"].iloc[0]
        substitution_feature_info=pd.read_excel(parameter_path, sheet_name="substitution_features_nuts")
        try:
            relevant_substition_feature_nuts=substitution_feature_info[substitution_feature_info["country"]==country]["substitution_nuts"].iloc[0]
            relevant_substitution_feature_country=relevant_substition_feature_nuts[:2]
        except:
            pass
        
        #%%
        # generate a string that indicates the three years previous to the prediction year
        climate_years = ""
        for y in range(year - 3, year):
            climate_years = climate_years + str(y)[-2:]

        
        #%%
        excluded_nuts_regions
        #%%
        try:
            relevant_nuts_regs = np.sort(
                nuts_input[
                    (nuts_input["CNTR_CODE"] == country)
                    & (nuts_input["year"] == year)
                    & (nuts_input["LEVL_CODE"] == nuts_level)
                ]["NUTS_ID"].values
            )
        except:
            relevant_nuts_regs = np.sort(
                nuts_input[
                    (nuts_input["CNTR_CODE"] == country)
                    & (nuts_input["LEVL_CODE"] == nuts_level)
                ]["NUTS_ID"].values
            )

        # %%
        relevant_nuts_regs = relevant_nuts_regs[
            np.where(np.isin(relevant_nuts_regs, excluded_nuts_regions, invert=True))
        ]

        #
        """
        one crop is excluded from the parameter estimation, because it serves as the reference crop.
        Since there may be several crops for which no parameter estimates exist (as only those crops for which the number of LUCAS observations exceeds
        'LUCAS_observation_inclusion_threshold'), we need to identify the first crop (in alphabetical order) which is not in all_crops, as this is the reference crop.
        """

        # %%

        params = pd.read_excel(
            regression_param_path
            + training_country
            + "_statsmodel_params_obsthreshold"
            + str(LUCAS_observation_inclusion_threshold)
            + ".xlsx"
        )
        covariance = pd.read_excel(
            regression_param_path
            + training_country
            + "_statsmodel_covariance_obsthreshold"
            + str(LUCAS_observation_inclusion_threshold)
            + ".xlsx"
        )
        
        #%%
        reference_crop_estimation = all_crops[
            np.where(np.isin(all_crops, np.array(params.columns), invert=True))[0][0]
        ]

        
        
        for nuts in relevant_nuts_regs:
            # if country=="EL":
            #    nuts="GR"+nuts[-1:]
            # if probabilities have already been calculated don't calculate it again
            # if os.path.isfile(output_path + nuts + "_" + str(prediction_year)):
            #     continue

            """IMPORTS"""
            # import statsmodel params and covariance matrix

            # import data on cellsize and agshare
            grid = pd.read_csv(
                feature_input_path + country+"/cell_size/1kmgrid_" + nuts + "_all_years.csv"
            )

            relevant_grid = fufop.get_relevant_grid(grid, nuts_boundary_year)

            # get crop classes and keep only those that appear in the training samples
            considered_crops = np.delete(np.array(params.columns), 0)
            considered_crops = np.concatenate(
                (np.array([reference_crop_estimation]), considered_crops)
            )

            with open(scaler_path + training_country + "_columns.txt") as t:
                features = t.readlines()
            feature_array = np.array([features[f][:-1] for f, _ in enumerate(features)])

            # feature data is aggregated at NUTS1 level. Even if the level at which the predictions are made is NUTS2 or NUTS3 level, data is still imported at NUTS1 level
            nuts_feature_import = nuts[:3]

            feature_df = relevant_grid[["CELLCODE"]]
            feature_column_name = []
            for feature in feature_array:
                if (
                    (feature == "slope_in_degree")
                    | (feature == "sand_content")
                    | (feature == "silt_content")
                    | (feature == "clay_content")
                ):
                    feature = feature.split("_")[0]
                if feature in climate_features:
                    data = pd.read_csv(
                        feature_input_path
                        +country
                        +"/"
                        + feature
                        + "_"
                        + climate_years
                        + "/1kmgrid_"
                        + nuts_feature_import
                        + ".csv"
                    )
                else:
                    data = pd.read_csv(
                        feature_input_path
                        +country
                        +"/"
                        + feature
                        + "/1kmgrid_"
                        + nuts_feature_import
                        + ".csv"
                    )
                if feature == "latitude4326":
                    data_relevant = data.iloc[:, [1, 2]]

                else:
                    data_relevant = data.iloc[:, [1, 3]]
                feature_df = pd.merge(
                    feature_df, data_relevant, how="left", on="CELLCODE"
                )
                # the names of the relevant columns sometimes are not exactly the same as the file name, therefore collect those column names
                feature_column_name.append(data_relevant.columns[1])

            #%%
            """REPLACE NAN VALUES"""
            lat3035, lon3035 = fufop.get_lat_lon_from_cellcode(feature_df["CELLCODE"])
            feature_df["lat3035"] = lat3035
            feature_df["lon3035"] = lon3035

            for number, nanvals in enumerate(feature_df.isna().sum()):
                if (nanvals > 0)&(nanvals<len(lat3035)):
                    column_name = feature_df.isna().sum().index[number]
                    feature_df[column_name] = fufop.replace_nanvalues(
                        feature_df, column_name
                    )
                    print(
                        f"{nanvals/len(feature_df)*100}% of values for {column_name} replaced"
                    )
                elif nanvals==len(lat3035):
                    column_name = feature_df.isna().sum().index[number]
                    print(f"replace 100% of data for {column_name}")
                    input_data_name=feature_array[np.where(np.array(feature_column_name)==column_name)[0]][0]
                    substitute_data=pd.read_csv(
                        intermediary_data_path + "/Zonal Stats/"+relevant_substitution_feature_country+"/"
                        +input_data_name+"/1kmgrid_"+relevant_substition_feature_nuts+".csv")
                    substitute_data_mean=substitute_data[column_name].mean()
                    feature_df[column_name]=np.repeat(substitute_data_mean,len(feature_df))
            """SCALE DATA AND ADD CONSTANT"""
            # ensure that the order of features is the same as used to train the parameters
            X = feature_df[feature_column_name]
            # load scaler
            correct_scaler = load(scaler_path + training_country)
            # scale
            X = correct_scaler.transform(X)
            # add vector of ones at first position for intercept
            X_statsmodel = np.insert(np.transpose(X), 0, np.ones(len(X)), axis=0)

            """PARAMETER PREPARATION"""
            randomly_selected_params = fufop.parameter_preparation(
                params, covariance, len(considered_crops)
            )


            
            all_probas = fufop.probability_calculation(
                randomly_selected_params,
                X_statsmodel,
                len(considered_crops),
                X_statsmodel.shape[1],
                sample_params=True,
                nofreps=20,
            )

            #%%
            """CROP SHARE PREDICTION"""
            all_probas = fufop.probability_calculation(
                randomly_selected_params,
                X_statsmodel,
                len(considered_crops),
                X_statsmodel.shape[1],
                sample_params=True,
                nofreps=20,
            )

            all_prediction_df = pd.DataFrame()
            for i, p_set in enumerate(all_probas):
                prediction_df = pd.DataFrame(
                    {
                        "CELLCODE": np.repeat(
                            np.array(relevant_grid["CELLCODE"]), len(considered_crops)
                        ),
                        "beta": np.repeat(f"betarand{i}", len(p_set.flatten())),
                        "crop": np.tile(considered_crops, len(relevant_grid)),
                        "probability": p_set.flatten(),
                    }
                )
                all_prediction_df = pd.concat([all_prediction_df, prediction_df])
#%%
            table = pa.Table.from_pandas(all_prediction_df)
            Path(output_path+country+"/").mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output_path+country+"/" + nuts + "_" + str(year))
            print(f"NUTS region {nuts} successfully exported")


# %%

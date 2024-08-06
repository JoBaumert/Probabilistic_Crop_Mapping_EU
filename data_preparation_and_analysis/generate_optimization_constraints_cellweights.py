#%%
from timeit import repeat
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
from rasterio.plot import show
import rasterio as rio
import sys
from pathlib import Path
import os

# %%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]

delineation_and_parameter_path = (
    data_main_path+"delineation_and_parameters/"
)
raw_data_path=data_main_path+"Raw_Data/"

# input files
regional_aggregates_path = (
    data_main_path+ "Intermediary_Data/Regional_Aggregates/"
)


excluded_NUTS_regions_path = delineation_and_parameter_path + "excluded_NUTS_regions.xlsx"
parameter_path = delineation_and_parameter_path + "DGPCM_user_parameters.xlsx"
nuts_input_path = data_main_path + "Intermediary_Data/Preprocessed_Inputs/NUTS/NUTS_all_regions_all_years.csv"
cellsize_input_path = data_main_path + "Intermediary_Data/Zonal_Stats/"

# output files
cell_weight_path = regional_aggregates_path+"Cell_Weights/"

# %%
# import parameters
selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])

nuts_year_info=pd.read_excel(parameter_path,sheet_name="NUTS")
relevant_nuts_years=np.array(nuts_year_info[nuts_year_info["crop_map_year"].isin(selected_years)]["nuts_year"])
crop_info = pd.read_excel(parameter_path, sheet_name="crops")
all_crops = np.array(crop_info["crop"])
levl_info = pd.read_excel(parameter_path, sheet_name="lowest_agg_level")
country_levls = {}
for i in range(len(levl_info)):
    country_levls[levl_info.iloc[i].country_code] = levl_info.iloc[i].lowest_agg_level

#%%
UAA = pd.read_csv(regional_aggregates_path+"coherent_UAA_"+str(selected_years.min())+str(selected_years.max())+".csv")
cropdata=pd.read_csv(regional_aggregates_path+"cropdata_"+str(selected_years.min())+str(selected_years.max())+".csv")


#%%
#cropdata is only needed to get information on the lowest level at which crop information is available for a country in a year
cropdata["NUTS_LEVL"]=np.vectorize(len)(np.array(cropdata["NUTS_ID"]))-2
lowest_level_country_year=cropdata[["country","year","NUTS_LEVL"]].groupby(["country","year"]).max().reset_index()

#%%
excluded_NUTS_regions = pd.read_excel(excluded_NUTS_regions_path)
excluded_NUTS_regions = np.array(excluded_NUTS_regions["excluded NUTS1 regions"])

# load data on NUTS regions in a year
NUTS_data = pd.read_csv(nuts_input_path)
# %%
NUTS_data=NUTS_data[(NUTS_data["CNTR_CODE"].isin(country_codes_relevant))&(NUTS_data["year"].isin(selected_years))]

#%%

#%%


#%%

#%%
if __name__ == "__main__":

    for country in country_codes_relevant:
        lowest_level_selected_country_allyears=lowest_level_country_year[
                (lowest_level_country_year["country"] == country)
            ]["NUTS_LEVL"].max()
        print(f"starting for {country}")
        weight_df_complete = pd.DataFrame()
        for year in selected_years:
            relevant_UAA = UAA[(UAA["country"] == country) & (UAA["year"] == year)]
            relevant_UAA["UAA_in_ha"] = relevant_UAA["UAA_corrected"] * 1000

            cellsize_country_df = pd.DataFrame()
            nuts1_regs = np.array(
                NUTS_data[
                    (NUTS_data["CNTR_CODE"] == country)
                    & (NUTS_data["LEVL_CODE"] == 1)
                    & (NUTS_data["year"] == year)
                ]["NUTS_ID"]
            )
            # discard those nuts regions that are overseas
            nuts1_regs = nuts1_regs[
                np.where(np.isin(nuts1_regs, excluded_NUTS_regions).astype(int) == 0)[0]
            ]
            lowest_level_selected_country_year = lowest_level_country_year[
                (lowest_level_country_year["country"] == country)
                & (lowest_level_country_year["year"] == year)
            ]["NUTS_LEVL"].iloc[0]

            for nuts1 in nuts1_regs:
                relevant_cells_df = pd.read_csv(
                    cellsize_input_path
                    + country
                    + "/inferred_UAA/1kmgrid_"
                    + nuts1
                    + ".csv"
                )
                relevant_cells_df = relevant_cells_df[relevant_cells_df["year"] == year]
                cellsize_country_df = pd.concat(
                    (cellsize_country_df, relevant_cells_df)
                )

            cellsize_country_df = cellsize_country_df[
                ["CELLCODE", f"nuts{country_levls[country]}", "inferred_UAA"]
            ]

            cellsize_country_df["CELLCODE_unique"] = [
                str(cellsize_country_df["CELLCODE"].iloc[i])
                + str(cellsize_country_df[f"nuts{country_levls[country]}"].iloc[i])
                for i in range(len(cellsize_country_df))
            ]

            weight_df = cellsize_country_df.copy()
            weight_df["inferred_UAA_in_ha"] = weight_df["inferred_UAA"] / 10000

            weight_df["lowest_relevant_NUTS_level"] = np.array(
                weight_df[f"nuts{lowest_level_selected_country_allyears}"]
            ).astype(f"U{lowest_level_selected_country_year+2}")

            weight_array = np.ndarray(len(weight_df))

            for reg in np.sort(
                np.array(weight_df["lowest_relevant_NUTS_level"].value_counts().keys())
            ):
                region_index = np.where(weight_df["lowest_relevant_NUTS_level"] == reg)[
                    0
                ]
                inferred_UAA = np.array(weight_df["inferred_UAA_in_ha"])[region_index]
                true_regional_UAA = relevant_UAA[relevant_UAA["NUTS_ID"] == reg][
                    "UAA_in_ha"
                ].iloc[0]
                inferred_regional_UAA = inferred_UAA.sum()
                factor = true_regional_UAA / inferred_regional_UAA
                cell_weight = inferred_UAA * factor
                weight_array[region_index] = cell_weight
            weight_df["weight"] = weight_array
            weight_df["country"] = np.repeat(country, len(weight_df))
            weight_df["year"] = np.repeat(year, len(weight_df))
            weight_df_complete = pd.concat((weight_df_complete, weight_df))
            weight_df_complete["NUTS1"] = np.array(
                weight_df_complete[f"nuts{lowest_level_selected_country_allyears}"]
            ).astype("U3")


        # export data for country
        Path(cell_weight_path + country ).mkdir(
            parents=True, exist_ok=True
        )
        weight_df_complete.to_csv(
            cell_weight_path + country + "/cell_weights_"+str(selected_years.min())+str(selected_years.max())+".csv"
        )
        print("data for " + country + " exported")
# %%
relevant_cells_df
# %%
relevant_cells_df = pd.read_csv(
    cellsize_input_path
    + country
    + "/inferred_UAA/1kmgrid_"
    + nuts1
    + ".csv"
)
# %%
cellsize_country_df
# %%
country_levls
# %%
lowest_level_country_year
# %%
weight_df_complete
# %%
grid=gpd.read_file("/home/baumert/fdiexchange/baumert/project1/Data/Raw_Data/Grid/PL_1km.zip!/pl_1km.shp")
# %%
weight_df_complete=pd.merge(weight_df_complete,grid[["CELLCODE","geometry"]],how="left",on="CELLCODE")
# %%
weight_df_complete=gpd.GeoDataFrame(weight_df_complete)
# %%
weight_df_complete[weight_df_complete["year"]==2014].plot(column="weight")
# %%
weight_df_complete[weight_df_complete["year"]==2020]
# %%
nuts1="HR0"
relevant_cells_df = pd.read_csv(
                    cellsize_input_path
                    + country
                    + "/inferred_UAA/1kmgrid_"
                    + nuts1
                    + ".csv"
                )
# %%
relevant_cells_df
# %%
reg
# %%

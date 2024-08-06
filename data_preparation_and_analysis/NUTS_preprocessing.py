#%%


# from gettext import find

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

#%%
#data_main_path=open(str(Path(__file__).parents[1])+"/data_main_path.txt")
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]


#%%
#inputs
nuts_path=data_main_path+"Raw_Data/NUTS/NUTS_RG_01M_"
parameter_path = (
    data_main_path+"delineation_and_parameters/DGPCM_user_parameters.xlsx"
)

#outputs
nuts_output_path = data_main_path+ "Intermediary_Data/Preprocessed_Inputs/NUTS/"



#%%
# ===============================================================================================================================================
"""
1) COMPILATION OF INFORMATION ON NUTS BOUNDARIES FOR THE YEARS 2006-2020
"""
# import parameters
nuts_info = pd.read_excel(parameter_path, sheet_name="NUTS")
all_years = np.array(nuts_info["crop_map_year"])
nuts_years = np.sort(nuts_info["nuts_year"].value_counts().keys())
relevant_nuts_years = np.array(nuts_info["nuts_year"])


# import information which countries are considered
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])

# import NUTS data
nuts_year_dict = {}
for year in nuts_years:
    nuts_year_dict[year] = gpd.read_file(nuts_path + str(year) + "_3035.shp.zip")


nuts_allyear_df = pd.DataFrame()
for y, year in enumerate(all_years):
    year_df = nuts_year_dict[relevant_nuts_years[y]][
        ["CNTR_CODE", "LEVL_CODE", "NUTS_ID", "NUTS_NAME", "geometry"]
    ]
    year_df["year"] = np.repeat(year, len(year_df))
    year_df["nuts_boundary_year"] = np.repeat(relevant_nuts_years[y], len(year_df))
    nuts_allyear_df = pd.concat((nuts_allyear_df, year_df))


nuts_allyear_df = nuts_allyear_df[
    nuts_allyear_df["CNTR_CODE"].isin(country_codes_relevant)
]
nuts_allyear_gdf = gpd.GeoDataFrame(nuts_allyear_df)

#%%
nuts_allyear_df
#%%
# export data
Path(nuts_output_path).mkdir(parents=True, exist_ok=True)
nuts_allyear_df.to_csv(nuts_output_path + "NUTS_all_regions_all_years.csv")
nuts_allyear_gdf.to_file(nuts_output_path + "NUTS_all_regions_all_years.shp")
print("successfully completed task")
#%%

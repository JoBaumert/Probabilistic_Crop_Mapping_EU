#%%
import argparse
from cgi import test
from re import S
import geopandas as gpd
import numpy as np
import pyproj
from shapely.geometry import Point
from shapely import wkt
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show
from rasterstats import zonal_stats, point_query
from rasterio.windows import from_bounds
import os
import richdem as rd
import time
from os.path import exists
import zipfile
from pathlib import Path


#%%
"""
This python script calculates the size of each cell in a region. While each cell originally has a size of 1km2, the cells at the boundary of NUTS region 
are smaller than 1km2 as they are divided by the boundary. A cell that is within a NUTS1 region and therefoer has a size of 1km2 when 
looking at the NUTS1 level may have a smaller size when looking, e.g, at NUTS2 level, if it is divided by a NUTS2 border. This is why for each cell and every (relevant) level the size
is calculated.
"""

#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]

raw_data_path = data_main_path+"Raw_Data/"
intermediary_data_path = data_main_path+"Intermediary_Data/"
delineation_and_parameter_path = data_main_path+"delineation_and_parameters/"

# input files
grid_1km_path=raw_data_path+"Grid/"
parameter_path = delineation_and_parameter_path + "DGPCM_user_parameters.xlsx"
nuts_path = raw_data_path + "NUTS/NUTS_RG_01M_"

excluded_NUTS_regions_path = delineation_and_parameter_path + "excluded_NUTS_regions.xlsx"

# output files
out_path = intermediary_data_path + "Zonal_Stats/"


# %%

# import parameters
nuts_info = pd.read_excel(parameter_path, sheet_name="NUTS")
all_years = np.array(nuts_info["crop_map_year"])
nuts_years = np.sort(nuts_info["nuts_year"].value_counts().keys())
relevant_nuts_years = np.array(nuts_info["nuts_year"])
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])
nuts_levels=pd.read_excel(parameter_path,sheet_name="lowest_agg_level")
#%%


#%%
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

#%%
nuts_allyear_gdf = gpd.GeoDataFrame(nuts_allyear_df)
#%%

#%%
# countries = {"UK": "United_Kingdom"}
# level1, level2, level3 = False, False, True
"""
parser = argparse.ArgumentParser()
parser.add_argument("-cc", "--ccode", type=str, required=True)
parser.add_argument("-cn", "--cname", type=str, required=True)
parser.add_argument("-l", "--level", nargs="+", required=True)

args = parser.parse_args()
countries = {args.ccode: args.cname}
level1, level2, level3 = False, False, False
if "1" in args.level:
    level1 = True
if "2" in args.level:
    level2 = True
if "3" in args.level:
    level3 = True

"""

#%%


if __name__ == "__main__":

    for country in country_codes_relevant[:1]:
       
        print(
            f"grid preparation starting for {country}"
        )
        
        #find name for relevant 1km grid file 
        grid_1km_path_country = (
       # "zip+file://"
         grid_1km_path
        + country
        +"_1km.zip"     
        )
        zip=zipfile.ZipFile(grid_1km_path_country)
        for file in zip.namelist():
            if file[-3:]=="shp":
                break

        grid_1km_country = gpd.read_file(grid_1km_path_country+"!/"+file)

        out_path_country = out_path + country + "/cell_size/"

        # all_nuts_data_df = pd.concat(all_nuts_data)
        NUTS1_regs = np.sort(
            np.array(
                nuts_allyear_gdf[
                    (nuts_allyear_gdf["CNTR_CODE"] == country)
                    & (nuts_allyear_gdf["LEVL_CODE"] == 1)
                ]["NUTS_ID"]
                .value_counts()
                .keys()
            )
        )
        NUTS2_regs = np.sort(
            np.array(
                nuts_allyear_gdf[
                    (nuts_allyear_gdf["CNTR_CODE"] == country)
                    & (nuts_allyear_gdf["LEVL_CODE"] == 2)
                ]["NUTS_ID"]
                .value_counts()
                .keys()
            )
        )
        NUTS3_regs = np.sort(
            np.array(
                nuts_allyear_gdf[
                    (nuts_allyear_gdf["CNTR_CODE"] == country)
                    & (nuts_allyear_gdf["LEVL_CODE"] == 3)
                ]["NUTS_ID"]
                .value_counts()
                .keys()
            )
        )
        
        
        # some regions (e.g., French overseas) are excluded
        excluded_NUTS_regions = pd.read_excel(excluded_NUTS_regions_path)
        excluded_NUTS_regions = np.array(
            excluded_NUTS_regions["excluded NUTS1 regions"]
        )
        
        nuts_regs = [NUTS1_regs, NUTS2_regs, NUTS3_regs]
        highest_NUTS_level=nuts_levels[nuts_levels["country_code"]==country]["lowest_agg_level"].iloc[0]

       
        print("loop starts")
        for level in range(highest_NUTS_level):
            for nuts in nuts_regs[level]:
                print(nuts)
                if (
                    not os.path.isfile(
                        out_path_country + "1kmgrid_" + nuts + "_all_years.csv"
                    )
                    and not nuts
                    in excluded_NUTS_regions  # some overseas NUTS1 regions and their corresponding subregions are excluded
                ):
                    if level > 1:
                        # if nuts2 or nuts3 regions are considered, load grid for nuts1 region to accelerate overlaying
                        grid_1km_relevant_df = pd.read_csv(
                            out_path_country
                            + "1kmgrid_"
                            + nuts[:3]
                            + "_all_years.csv"
                        )
                        grid_1km_relevant_df["geometry"] = grid_1km_relevant_df[
                            "geometry"
                        ].apply(wkt.loads)
                        grid_1km_relevant_allyears = gpd.GeoDataFrame(
                            grid_1km_relevant_df, crs="epsg:3035"
                        )
                    else:
                        grid_1km_relevant = grid_1km_country

                    grid_years_dict = {"year": [], "grid": []}
                    for y, year in enumerate(nuts_years):
                        if level > 1:
                            grid_1km_relevant = grid_1km_relevant_allyears[
                                grid_1km_relevant_allyears["year"] == year
                            ]
                        nutsregion = nuts_allyear_gdf[
                            (nuts_allyear_gdf["nuts_boundary_year"] == year)
                            & (nuts_allyear_gdf["NUTS_ID"] == nuts)
                        ]
                        print(f"start overlaying grids for {nuts} in {year}")
                        st = time.time()
                        grid_1km_nutsregion = grid_1km_relevant.overlay(
                            nutsregion, how="intersection"
                        )
                        et = time.time()
                        # print(f"overlaying took {et-st} seconds")
                        grid_1km_nutsregion["area"] = grid_1km_nutsregion.area
                        grid_1km_nutsregion = grid_1km_nutsregion[
                            ["CELLCODE", "geometry", "area"]
                        ]
                        grid_years_dict["year"].append(year)
                        grid_years_dict["grid"].append(grid_1km_nutsregion)

                    grid_years = []
                    for g, grid in enumerate(grid_years_dict["grid"]):
                        grid["year"] = np.repeat(
                            grid_years_dict["year"][g], len(grid)
                        )
                        grid_years.append(grid)

                    grid_years = pd.concat(grid_years)
                    # for some reason, in some cases a cell is overlaid multiple times for the same year (with the same resulting value)
                    # --> drop these duplicates
                    grid_years.drop_duplicates(["CELLCODE", "year"], inplace=True)
                    # export
                    Path(out_path_country).mkdir(parents=True, exist_ok=True)
                    grid_years.to_csv(
                        out_path_country + "1kmgrid_" + nuts + "_all_years.csv"
                    )
                    print("created csv file for region " + nuts)

    # %%

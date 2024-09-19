#%%
from cgi import test

# from re import S
import geopandas as gpd
import math
import numpy as np

# import pyproj
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import from_bounds
import os
import richdem as rd
from os.path import exists
import zipfile
from pathlib import Path

# from O2B_feature_grid_calculation import NUTS1

import modules.functions_for_data_preparation as ffd

#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]



#%%

"""
SETTINGS
"""


selected_years_climate = [
    [2003, 2004, 2005], #for LUCAS 2006
    [2006, 2007, 2008], #for LUCAS 2009
    [2009, 2010, 2011], #for LUCAS 2012
    [2012, 2013, 2014], #for LUCAS 2015
    [2015, 2016, 2017], #for LUCAS 2018
]


find_elevation = True
find_slope_and_aspect = True
find_temperature = True
find_veg_period = True
find_precipitation = True
find_sand_content = True
find_clay_content = True
find_coarse_fragments = True
find_silt_content = True
find_awc = True
find_bulk_density = True
find_latitude4326 = True
#%%
# determine path to the project folder

raw_data_path=data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
delineation_and_parameter_path = (
    data_main_path+"delineation_and_parameters/"
)
"""input paths"""
parameter_path = delineation_and_parameter_path + "DGPCM_user_parameters.xlsx"
crop_delineation_path = delineation_and_parameter_path + "DGPCM_crop_delineation.xlsx"
excluded_NUTS_regions_path=delineation_and_parameter_path+ "excluded_NUTS_regions.xlsx"
nuts_path=intermediary_data_path+"Preprocessed_Inputs/NUTS/NUTS_all_regions_all_years.shp"
grid_25km_path = raw_data_path+"Grid/grid25.zip"
#explanatory vars:
elev_path = raw_data_path+"DEM/eudem_dem_3035_europe.tif"

slope_path = "zip+file:///"+raw_data_path+"DEM/eudem_slop_3035_europe.zip!/eudem_slop_3035_europe/eudem_slop_3035_europe.tif"

sand_path = "zip+file:///"+raw_data_path+"Soil/Sand_Extra.zip!/Sand1.tif" 
clay_path = "zip+file:///"+raw_data_path+"Soil/Clay_Extra.zip!/Clay.tif"
silt_path = "zip+file:///"+raw_data_path+"Soil/Silt_Extra.zip!/Silt1.tif"
bulk_density_path = "zip+file:///"+raw_data_path+"Soil/BulkDensity_Extra.zip!/Bulk_density.tif"
coarse_fragments_path = "zip+file:///"+raw_data_path+"Soil/CoarseFragments_Extra.zip!/Coarse_fragments.tif"
awc_path = "zip+file:///"+raw_data_path+"Soil/AWC_Extra.zip!/AWC.tif"
temp_path = intermediary_data_path+"Preprocessed_Inputs/Climate/all_temperature_data.parquet"
precipit_path = intermediary_data_path+"Preprocessed_Inputs/Climate/all_precipitation_data.parquet"
#LUCAS path:
lucas_path=intermediary_data_path+"Preprocessed_Inputs/LUCAS/LUCAS_preprocessed.csv"
"""Output path:"""
out_path = intermediary_data_path+"/LUCAS_feature_merges/"
#%%

selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])

NUTS_gdf = gpd.read_file(nuts_path)

# for some countries (e.g., Spain and France) some NUTS regions are not in Europe --> exclude those regions mentioned in the excel file
excluded_NUTS_file = pd.read_excel(excluded_NUTS_regions_path)
excluded_NUTS_country = np.array(excluded_NUTS_file["country"].value_counts().keys())
excluded_NUTS_regions = np.array(excluded_NUTS_file["excluded NUTS1 regions"])


#%%
"""
MERGING
"""

if __name__ == "__main__":

    print("get explanatory variables for locations of LUCAS (LUCAS feature merges)...")
    for country in country_codes_relevant:

        #when reproducing the maps without the original slope and elevation data this if clause will be activated
        if not os.path.isfile(elev_path):
            elev_path_relevant=raw_data_path+"DEM/eudem_dem_3035_"+country+".tif"
            slope_path_relevant=raw_data_path+"DEM/eudem_slope_3035_"+country+".tif"

        else:
            elev_path_relevant=elev_path
            slope_path_relevant=slope_path

        
        # load files
        NUTS_dict = ffd.get_NUTS_regions(NUTS_gdf, country)
        NUTS1_regs = NUTS_dict["NUTS1"]
        NUTS2_regs = NUTS_dict["NUTS2"]
        NUTS3_regs = NUTS_dict["NUTS3"]

        NUTS_country = NUTS_gdf[NUTS_gdf["CNTR_CODE"] == country]

        # for some countries (e.g., Spain and France) some NUTS regions are not in Europe --> exclude them
        if country in excluded_NUTS_country:
            NUTS_country = NUTS_country[
                (~NUTS_country["FID"].isin(excluded_NUTS_regions))
                & (NUTS_country["LEVL_CODE"] == 1)
            ]

        if (find_temperature) or (find_veg_period) or (find_precipitation):
            grid_25km = gpd.read_file(grid_25km_path)
            # keep only those cells that are on land (not a country's sea territory)
            grid_25km_country = grid_25km.overlay(
                NUTS_country[NUTS_country["NUTS_ID"].isin(NUTS1_regs)],
                how="intersection",
            )
            grid_25km_country = grid_25km_country.merge(
                grid_25km, how="left", on="FID_1"
            )
            grid_25km_country.drop(columns="geometry_x", inplace=True)
            grid_25km_country.rename(columns={"geometry_y": "geometry"}, inplace=True)
            grid_25km_country = gpd.GeoDataFrame(grid_25km_country)
            grid_25km_country.rename(columns={"Grid_Code_x": "GRID_NO"}, inplace=True)

        # if climate data is used, load it for the entire country:
        if (find_temperature) or (find_veg_period):
            temperature_data = pd.read_parquet(temp_path)
            temperature_data_relevant = temperature_data[
                temperature_data["COUNTRY"] == country
            ]

        if find_precipitation:
            precipitation_data = pd.read_parquet(precipit_path)
            precipitation_data_relevant = precipitation_data[
                precipitation_data["COUNTRY"] == country
            ]

        """import needed data"""

        LUCAS_preprocessed = pd.read_csv(lucas_path)

        LUCAS_selected = LUCAS_preprocessed[
            LUCAS_preprocessed["nuts1"].isin(NUTS1_regs)
        ]
        LUCAS_geom = [
            Point(xy) for xy in zip(LUCAS_selected["th_long"], LUCAS_selected["th_lat"])
        ]
        LUCAS_selected_epsg4326 = gpd.GeoDataFrame(
            LUCAS_selected, geometry=LUCAS_geom, crs="epsg:4326"
        )
        LUCAS_selected_epsg3035 = LUCAS_selected_epsg4326.to_crs(crs="epsg:3035")
        rel_nuts2 = sorted(list(LUCAS_selected_epsg4326["nuts2"].value_counts().keys()))

        lon4326, lat4326 = (
            LUCAS_selected_epsg4326["geometry"].x,
            LUCAS_selected_epsg4326["geometry"].y,
        )
        lon3035, lat3035 = (
            LUCAS_selected_epsg3035["geometry"].x,
            LUCAS_selected_epsg3035["geometry"].y,
        )

       
        
        #%%
        """SLOPE"""
        if find_slope_and_aspect and not os.path.isfile(
            out_path + "/" + country + "/slope.csv"
        ):
            # get bounding boxes of NUTS region to load only relevant part of raster file
            left, bottom, right, top = NUTS_country.total_bounds
            # add some buffer to ensure that matrix really contains all cells
            left, bottom, right, top = (
                left - 10000,
                bottom - 10000,
                right + 10000,
                top + 10000,
            )
            window = [left, bottom, right, top]

            slope_in_DN = ffd.get_elevation_LUCAS(
                slope_path_relevant, lon3035, lat3035, window=window
            )

            if not os.path.isfile(elev_path):
                slope_in_degree=slope_in_DN

            else:
                slope_in_degree = np.arccos(slope_in_DN / 250) * 180 / np.pi

            df_slope = LUCAS_selected[["year", "id", "point_id"]]
            df_slope["slope_in_degree"] = slope_in_degree
            # export data
            Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
            df_slope.to_csv(out_path + "/" + country + "/slope.csv")
            print(f"{country}: slope data exported")
        
        #%%
        """ELEVATION"""
        if find_elevation and not os.path.isfile(
            out_path + "/" + country + "/elevation.csv"
        ):
            # get bounding boxes of NUTS region to load only relevant part of raster file
            left, bottom, right, top = NUTS_country.total_bounds
            # add some buffer to ensure that matrix really contains all cells
            left, bottom, right, top = (
                left - 10000,
                bottom - 10000,
                right + 10000,
                top + 10000,
            )
            window = [left, bottom, right, top]

            elevation = ffd.get_elevation_LUCAS(
                elev_path_relevant, lon3035, lat3035, window=window
            )

            df_elevation = LUCAS_selected[["year", "id", "point_id"]]
            df_elevation["elevation"] = elevation
            # export data
            Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
            df_elevation.to_csv(out_path + "/" + country + "/elevation.csv")
            print(f"{country}: elevation data exported")
        #%%

        """SAND"""
        if find_sand_content and not os.path.isfile(
            out_path + "/" + country + "/sand.csv"
        ):
            sand = ffd.get_soil_content_LUCAS(sand_path, lon3035, lat3035)
            df_sand = LUCAS_selected[["year", "id", "point_id", "sand"]]
            df_sand["sand_content_calculated"] = sand
            # for some LUCAS points the sand content was measured. If it is available, take the sand content that was measured at the spot, otherwise use the map
            values_exist = np.where(df_sand["sand"] >= 0)
            sand_corrected = sand
            sand_corrected[values_exist] = np.array(df_sand["sand"])[values_exist]
            df_sand["sand_content"] = sand_corrected
            # export data
            Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
            df_sand.to_csv(out_path + "/" + country + "/sand.csv")
            print(f"{country}: sand data exported")
        ##%%

        """CLAY"""
        if find_clay_content and not os.path.isfile(
            out_path + "/" + country + "/clay.csv"
        ):
            clay = ffd.get_soil_content_LUCAS(clay_path, lon3035, lat3035)
            df_clay = LUCAS_selected[["year", "id", "point_id", "clay"]]
            df_clay["clay_content_calculated"] = clay
            # for some LUCAS points the clay content was measured. If it is available, take the clay content that was measured at the spot, otherwise use the map
            values_exist = np.where(df_clay["clay"] >= 0)
            clay_corrected = clay
            clay_corrected[values_exist] = np.array(df_clay["clay"])[values_exist]
            df_clay["clay_content"] = clay_corrected
            # export data
            Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
            df_clay.to_csv(out_path + "/" + country + "/clay.csv")
            print(f"{country}: clay data exported")
        ##%%

        """SILT"""
        if find_silt_content and not os.path.isfile(
            out_path + "/" + country + "/silt.csv"
        ):
            silt = ffd.get_soil_content_LUCAS(silt_path, lon3035, lat3035)
            df_silt = LUCAS_selected[["year", "id", "point_id", "silt"]]
            df_silt["silt_content_calculated"] = silt
            # for some LUCAS points the silt content was measured. If it is available, take the silt content that was measured at the spot, otherwise use the map
            values_exist = np.where(df_silt["silt"] >= 0)
            silt_corrected = silt
            silt_corrected[values_exist] = np.array(df_silt["silt"])[values_exist]
            df_silt["silt_content"] = silt_corrected
            # export data
            Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
            df_silt.to_csv(out_path + "/" + country + "/silt.csv")
            print(f"{country}: silt data exported")
       

        """COARSE FRAGMENTS"""
        if find_coarse_fragments and not os.path.isfile(
            out_path + "/" + country + "/coarse_fragments.csv"
        ):
            cf = ffd.get_soil_content_LUCAS(coarse_fragments_path, lon3035, lat3035)
            df_cf = LUCAS_selected[["year", "id", "point_id", "coarse"]]
            df_cf["coarse_fragments_calculated"] = cf

            # for some LUCAS points the coarse_fragments were measured. If it is available, take the coarse fragments that was measured at the spot, otherwise use the map
            values_exist = np.where(df_cf["coarse"] >= 0)
            cf_corrected = cf
            cf_corrected[values_exist] = np.array(df_cf["coarse"])[values_exist]
            df_cf["coarse_fragments"] = cf_corrected
            # export data
            Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
            df_cf.to_csv(out_path + "/" + country + "/coarse_fragments.csv")
            print(f"{country}: coarse fragments data exported")
        

        """AWC"""
        if find_awc and not os.path.isfile(out_path + "/" + country + "/awc.csv"):
            awc = ffd.get_soil_content_LUCAS(awc_path, lon3035, lat3035)
            df_awc = LUCAS_selected[["year", "id", "point_id"]]
            df_awc["awc"] = awc
            # export data
            Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
            df_awc.to_csv(out_path + "/" + country + "/awc.csv")
            print(f"{country}: awc data exported")
        

        """BULK DENSITY"""
        if find_bulk_density and not os.path.isfile(
            out_path + "/" + country + "/bulk_density.csv"
        ):
            bulk_density = ffd.get_soil_content_LUCAS(
                bulk_density_path, lon3035, lat3035
            )
            df_bulk_density = LUCAS_selected[["year", "id", "point_id"]]
            df_bulk_density["bulk_density"] = bulk_density
            # export data
            Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
            df_bulk_density.to_csv(out_path + "/" + country + "/bulk_density.csv")
            print(f"{country}: bulk density data exported")
        #%%

        """CLIMATE"""
        if (find_temperature) or (find_veg_period) or (find_precipitation):
            # connect LUCAS points with 25km grid cells
            LUCAS_grid_link_df = ffd.link_grid_and_points(
                grid_25km, lon3035, lat3035, LUCAS_selected
            )

            if (find_temperature) or (find_veg_period):
                temperature = temperature_data_relevant
                temperature["year"] = np.array(
                    np.array(temperature["DAY"]).astype("U4")
                ).astype(int)
                df_temperature = LUCAS_selected[["year", "id", "point_id"]]
                df_veg_period = LUCAS_selected[["year", "id", "point_id"]]

            if find_precipitation:
                precipitation = precipitation_data_relevant
                precipitation["year"] = np.array(
                    np.array(precipitation["DAY"]).astype("U4")
                ).astype(int)
                df_precipitation = LUCAS_selected[["year", "id", "point_id"]]

            for year_group in selected_years_climate:
                yearstr = ""
                for year in year_group:
                    yearstr += str(year)[-2:]

                """TEMPERATURE RELATED DATA"""
                if (find_temperature) or (find_veg_period):
                    temperature_selected_years = temperature[
                        temperature["year"].isin(year_group)
                    ]

                    """TEMPERATURE SUM"""
                    if find_temperature and not os.path.isfile(
                        out_path + "/" + country + "/avg_annual_temp_sum.csv"
                    ):
                        temperature_annual_sum = ffd.get_annual_temp_sum(
                            temperature_selected_years
                        )

                        temperature_annual_sum_mean = (
                            temperature_annual_sum[["GRID_NO", "TEMPERATURE_AVG"]]
                            .groupby("GRID_NO")
                            .mean()
                            .reset_index()
                        )

                        temperature_annual_sum_mean.rename(
                            columns={
                                "TEMPERATURE_AVG": "tempsum_annual_mean_" + yearstr
                            },
                            inplace=True,
                        )
                        # merge gridded climate data with LUCAS points and save it in df_temperature
                        tempsum_LUCAS_selected = pd.merge(
                            LUCAS_grid_link_df,
                            temperature_annual_sum_mean,
                            how="left",
                            left_on="Grid_Code",
                            right_on="GRID_NO",
                        )
                        df_temperature = pd.merge(
                            df_temperature,
                            tempsum_LUCAS_selected[
                                ["id", "tempsum_annual_mean_" + yearstr]
                            ],
                            how="left",
                            on="id",
                        )

                    """VEGETATION PERIOD"""
                    if find_veg_period and not os.path.isfile(
                        out_path + "/" + country + "/avg_annual_veg_period.csv"
                    ):
                        temperature_veg_period = ffd.get_annual_veg_period(
                            temperature_selected_years
                        )
                        temperature_veg_period_mean = (
                            temperature_veg_period[["GRID_NO", "TEMPERATURE_AVG"]]
                            .groupby("GRID_NO")
                            .mean()
                            .reset_index()
                        )
                        temperature_veg_period_mean.rename(
                            columns={
                                "TEMPERATURE_AVG": "vegperiod_annual_mean_" + yearstr
                            },
                            inplace=True,
                        )
                        # merge gridded climate data with LUCAS points and save it in df_veg_period
                        veg_period_LUCAS_selected = pd.merge(
                            LUCAS_grid_link_df,
                            temperature_veg_period_mean,
                            how="left",
                            left_on="Grid_Code",
                            right_on="GRID_NO",
                        )
                        df_veg_period = pd.merge(
                            df_veg_period,
                            veg_period_LUCAS_selected[
                                ["id", "vegperiod_annual_mean_" + yearstr]
                            ],
                            how="left",
                            on="id",
                        )

                """PRECIPITATION DATA"""
                if find_precipitation and not os.path.isfile(
                    out_path + "/" + country + "/avg_annual_precipitation.csv"
                ):
                    precipitation_selected_years = precipitation[
                        precipitation["year"].isin(year_group)
                    ]
                    precipitation_annual_sum = ffd.get_annual_precipitation_sum(
                        precipitation_selected_years
                    )
                    precipitation_annual_sum_mean = (
                        precipitation_annual_sum[["GRID_NO", "PRECIPITATION"]]
                        .groupby("GRID_NO")
                        .mean()
                        .reset_index()
                    )
                    precipitation_annual_sum_mean.rename(
                        columns={
                            "PRECIPITATION": "precipitation_annual_mean_" + yearstr
                        },
                        inplace=True,
                    )
                    # merge gridded climate data with LUCAS points and save it in df_veg_period
                    precipitation_LUCAS_selected = pd.merge(
                        LUCAS_grid_link_df,
                        precipitation_annual_sum_mean,
                        how="left",
                        left_on="Grid_Code",
                        right_on="GRID_NO",
                    )
                    df_precipitation = pd.merge(
                        df_precipitation,
                        precipitation_LUCAS_selected[
                            ["id", "precipitation_annual_mean_" + yearstr]
                        ],
                        how="left",
                        on="id",
                    )

            # export data

            if not os.path.isfile(
                out_path + "/" + country + "/avg_annual_temp_sum.csv"
            ):
                Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
                df_temperature.to_csv(
                    out_path + "/" + country + "/avg_annual_temp_sum.csv"
                )
                print(f"{country}: temperature data exported")

            if not os.path.isfile(
                out_path + "/" + country + "/avg_annual_veg_period.csv"
            ):
                Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
                df_veg_period.to_csv(
                    out_path + "/" + country + "/avg_annual_veg_period.csv"
                )
                print(f"{country}: veg period data exported")

            if not os.path.isfile(
                out_path + "/" + country + "/avg_annual_precipitation.csv"
            ):
                Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
                df_precipitation.to_csv(
                    out_path + "/" + country + "/avg_annual_precipitation.csv"
                )
                print(f"{country}: precipitation data exported")

        """LATITUDE"""
        if find_latitude4326 and not os.path.isfile(
            out_path + "/" + country + "/latitude4326.csv"
        ):
            df_latitude4326 = LUCAS_selected[["year", "id", "point_id"]]
            df_latitude4326["latitude4326"] = lat4326
            # export data
            Path(out_path + "/" + country).mkdir(parents=True, exist_ok=True)
            df_latitude4326.to_csv(out_path + "/" + country + "/latitude4326.csv")
            print(f"{country}: latitude data exported")

# %%
# %%

#%%
"""
for the interpretation of the corine code see https://land.copernicus.eu/user-corner/technical-library/clc-product-user-manual
page 67
"""

from cgi import test
from re import S
import geopandas as gpd
import math
import numpy as np
import pyproj
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import from_bounds
import os, zipfile
import richdem as rd

from os.path import exists
import zipfile
from pathlib import Path

import modules.functions_for_data_preparation as ffd

#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]


raw_data_path=data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
delineation_and_parameter_path = (
    data_main_path+"delineation_and_parameters/"
)
"""input paths"""
parameter_path = delineation_and_parameter_path + "DGPCM_user_parameters.xlsx"
excluded_NUTS_regions_path=delineation_and_parameter_path+ "excluded_NUTS_regions.xlsx"
nuts_path=intermediary_data_path+"Preprocessed_Inputs/NUTS/NUTS_all_regions_all_years.shp"
grid_1km_path=raw_data_path+"Grid/"
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
zonal_stats_path=intermediary_data_path + "Zonal_Stats/"
corine_basic_path=raw_data_path+"CORINE/"
# output files
out_path = zonal_stats_path


# assign user specific parameters and values


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
find_corine_class=True
calculate_UAA=True
"""
find_elevation = False
find_slope_and_aspect = False
find_temperature = False
find_veg_period = False
find_precipitation = False
find_sand_content = False
find_clay_content = False
find_coarse_fragments = False
find_silt_content = False
find_awc = False
find_bulk_density = False
find_latitude4326 = False
find_corine_class=True
calculate_UAA=True
"""
corine_years = [2006, 2012, 2018]  # all years for which corine data is available
climate_mean_nofyears = 3


#%%
# import nuts data
NUTS_gdf = gpd.read_file(nuts_path)

# import parameters
selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])

corine_info = pd.read_excel(parameter_path, sheet_name="CORINE")
relevant_corine_years = np.array(corine_info[corine_info["crop_map_year"].isin(selected_years)]["corine_year"])
corine_years = np.sort(np.unique(relevant_corine_years))
nuts_year_info=pd.read_excel(parameter_path,sheet_name="NUTS")
relevant_nuts_years=np.array(nuts_year_info[nuts_year_info["crop_map_year"].isin(selected_years)]["nuts_year"])

corine_ag_classes=np.array(pd.read_excel(parameter_path, sheet_name="CORINE_ag_classes")["ag_classes"])
#%%

levl_info = pd.read_excel(parameter_path, sheet_name="lowest_agg_level")
country_levls = {}
for i in range(len(levl_info)):
    country_levls[levl_info.iloc[i].country_code] = levl_info.iloc[i].lowest_agg_level

#%%
climate_mean_firstyear = selected_years.min()-climate_mean_nofyears
climate_mean_lastyear = selected_years.max()-1


#%%
if __name__ == "__main__":

    print("get feature values for each grid cell...")
    for country in country_codes_relevant:
        #when reproducing the maps without the original slope and elevation data this if clause will be activated
        if not os.path.isfile(elev_path):
            elev_path_relevant=raw_data_path+"DEM/eudem_dem_3035_"+country+".tif"
            slope_path_relevant=raw_data_path+"DEM/eudem_slope_3035_"+country+".tif"

        else:
            elev_path_relevant=elev_path
            slope_path_relevant=slope_path
    
        print(country)


        NUTS_dict = ffd.get_NUTS_regions(NUTS_gdf, country)

        NUTS1_regs = NUTS_dict["NUTS1"]
        NUTS2_regs = NUTS_dict["NUTS2"]
        NUTS3_regs = NUTS_dict["NUTS3"]


        # assign file paths relevant for the specific country only
        
        grid_1km_regionshape_path = zonal_stats_path + country + "/cell_size/"


        # load files
        # NUTS = gpd.read_file(nuts_path)
        grid_1km_path_country = (
        # "zip+file://"
            grid_1km_path
            + country
            +"_1km.zip"     
            )

        zip=zipfile.ZipFile(grid_1km_path_country)
        for file in zip.namelist():
            if (file[-3:]=="shp")&(file[3:6]=="1km"):
                break


        grid_1km_country = gpd.read_file(grid_1km_path_country+"!/"+file)
        grid_25km = gpd.read_file(grid_25km_path)

        NUTS_country = NUTS_gdf[NUTS_gdf["CNTR_CODE"] == country]

        
        # keep only those cells that are on land (not a country's sea territory)
        grid_25km_country = grid_25km.overlay(
            NUTS_country[NUTS_country["NUTS_ID"].isin(NUTS1_regs)], how="intersection"
        )
        grid_25km_country = grid_25km_country.merge(grid_25km, how="left", on="FID_1")
        grid_25km_country.drop(columns="geometry_x", inplace=True)
        grid_25km_country.rename(columns={"geometry_y": "geometry"}, inplace=True)
        grid_25km_country = gpd.GeoDataFrame(grid_25km_country)
        grid_25km_country.rename(columns={"Grid_Code_x": "GRID_NO"}, inplace=True)

        # some regions (e.g., French overseas) are excluded
        excluded_NUTS_regions = pd.read_excel(excluded_NUTS_regions_path)
        excluded_NUTS_regions = np.array(excluded_NUTS_regions["excluded NUTS1 regions"])

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



        #%%
        """merge data"""
        

        for NUTS1 in NUTS1_regs:
            if not NUTS1 in excluded_NUTS_regions:
                NUTS_country_selected = NUTS_country[NUTS_country["NUTS_ID"] == NUTS1]
                # grid_1km_selected=grid_1km_country.overlay(NUTS_country_selected, how='intersection')
                grid_1km_selected = pd.read_csv(
                    grid_1km_regionshape_path + "1kmgrid_" + NUTS1 + "_all_years.csv"
                )
                # for all cells that intersect with the NUTS region in any year, get the whole cell
                all_cells = sorted(
                    list(grid_1km_selected["CELLCODE"].value_counts().keys())
                )
                grid_1km_selected = grid_1km_country[
                    grid_1km_country["CELLCODE"].isin(all_cells)
                ]
                grid_1km_selected.sort_values(by="CELLCODE", inplace=True)
                # grid_1km_selected.drop(columns='geometry_x',inplace=True)
                # grid_1km_selected.rename(columns={'geometry_y':'geometry'},inplace=True)
                # grid_1km_selected=gpd.GeoDataFrame(grid_1km_selected)
                # position of upper left corner of cell:
                cells_lon = np.array(grid_1km_selected["EOFORIGIN"])
                cells_lat = np.array(grid_1km_selected["NOFORIGIN"]) + 1000

                if find_slope_and_aspect and not os.path.isfile(
                    out_path + "/" + country + "/slope/1kmgrid_" + NUTS1 + ".csv"
                ):

                    """SLOPE"""
                    # get bounding boxes of NUTS region to load only relevant part of raster file
                    left, bottom, right, top = NUTS_country_selected.total_bounds
                    # add some buffer to ensure that matrix really contains all cells
                    left, bottom, right, top = (
                        left - 10000,
                        bottom - 10000,
                        right + 10000,
                        top + 10000,
                    )
                    window = [left, bottom, right, top]
                    slope_in_DN = ffd.get_elevation(
                        slope_path_relevant, grid_1km_selected, window=window
                    )
                    if not os.path.isfile(elev_path):
                        slope_in_degree=slope_in_DN

                    else:
                        slope_in_degree=np.arccos(slope_in_DN / 250) * 180 / np.pi
                    slope_grid = pd.DataFrame(
                        {
                            "CELLCODE": grid_1km_selected["CELLCODE"].values,
                            # calculation of slope in degrees: see here: https://land.copernicus.eu/user-corner/technical-library/slope-conversion-table
                            "slope_degree": slope_in_degree,
                        }
                    )
                    slope_grid_1km_selected = pd.merge(
                        slope_grid, grid_1km_selected, how="left", on="CELLCODE"
                    )
                    slope_grid_1km_selected = gpd.GeoDataFrame(slope_grid_1km_selected)
                    slope_grid_1km_selected = slope_grid_1km_selected[
                        ["CELLCODE", "geometry", "slope_degree"]
                    ]

                    # export slope data for NUTS1 region
                    Path(out_path + "/" + country + "/slope").mkdir(
                        parents=True, exist_ok=True
                    )
                    slope_grid_1km_selected.to_csv(
                        out_path + "/" + country + "/slope/1kmgrid_" + NUTS1 + ".csv"
                    )
                    print(
                        f"file '{out_path}/{country}/slope/1kmgrid_{NUTS1}.csv' was created successfully"
                    )

                if find_elevation and not os.path.isfile(
                    out_path + "/" + country + "/elevation/1kmgrid_" + NUTS1 + ".csv"
                ):
                    """ELEVATION"""
                    # get bounding boxes of NUTS region to load only relevant part of raster file
                    left, bottom, right, top = NUTS_country_selected.total_bounds
                    # add some buffer to ensure that matrix really contains all cells
                    left, bottom, right, top = (
                        left - 10000,
                        bottom - 10000,
                        right + 10000,
                        top + 10000,
                    )
                    window = [left, bottom, right, top]
                    elevation_grid = pd.DataFrame(
                        {
                            "CELLCODE": grid_1km_selected["CELLCODE"].values,
                            "elevation": ffd.get_elevation(
                                elev_path_relevant, grid_1km_selected, window=window
                            ),
                        }
                    )
                    elevation_grid_1km_selected = pd.merge(
                        elevation_grid, grid_1km_selected, how="left", on="CELLCODE"
                    )
                    elevation_grid_1km_selected = gpd.GeoDataFrame(
                        elevation_grid_1km_selected
                    )
                    elevation_grid_1km_selected = elevation_grid_1km_selected[
                        ["CELLCODE", "geometry", "elevation"]
                    ]
                    # export elevation data for NUTS1 region
                    Path(out_path + "/" + country + "/elevation").mkdir(
                        parents=True, exist_ok=True
                    )
                    elevation_grid_1km_selected.to_csv(
                        out_path
                        + "/"
                        + country
                        + "/elevation/1kmgrid_"
                        + NUTS1
                        + ".csv"
                    )
                    print(
                        f"file '{out_path}/{country}/elevation/1kmgrid_{NUTS1}.csv' was created successfully"
                    )

                if (find_temperature) or (find_veg_period) or (find_precipitation):
                    """CLIMATE"""
                    cellcode1km_gridno25km_df = ffd.grid25km_to_grid1km(
                        grid_25km_country, grid_1km_selected, cells_lon, cells_lat
                    )

                    if (find_temperature) or (find_veg_period):
                        temperature = temperature_data_relevant
                        temperature["year"] = temperature["DAY"].apply(
                            lambda x: int(str(x)[:4])
                        )

                    if find_precipitation:
                        precipitation = precipitation_data_relevant
                        precipitation["year"] = precipitation["DAY"].apply(
                            lambda x: int(str(x)[:4])
                        )
                    for year in np.arange(
                        climate_mean_firstyear,
                        climate_mean_lastyear - climate_mean_nofyears + 2,
                    ):
                        climate_mean_years = list(
                            np.arange(year, year + climate_mean_nofyears)
                        )
                        yearstr = ""
                        for year in climate_mean_years:
                            yearstr += str(year)[-2:]

                        if (find_temperature) or (find_veg_period):
                            temperature_selected_years = temperature[
                                temperature["year"].isin(climate_mean_years)
                            ]

                        if (
                            find_temperature
                        ):  # and not os.path.isfile( out_path + "/" + country + "/avg_annual_temp_sum_" + yearstr + "/1kmgrid_" + NUTS1 + ".csv"):
                            temperature_annual_sum = ffd.get_annual_temp_sum(
                                temperature_selected_years
                            )

                            temperature_annual_sum_mean = (
                                temperature_annual_sum[
                                    temperature_annual_sum["year"].isin(
                                        climate_mean_years
                                    )
                                ][["GRID_NO", "TEMPERATURE_AVG"]]
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
                            tempsum_grid_1km_selected = pd.merge(
                                cellcode1km_gridno25km_df,
                                temperature_annual_sum_mean,
                                how="left",
                                on="GRID_NO",
                            )

                            
                            tempsum_grid_1km_selected = pd.merge(
                                tempsum_grid_1km_selected,
                                grid_1km_selected,
                                how="left",
                                on="CELLCODE",
                            )
                            tempsum_grid_1km_selected = gpd.GeoDataFrame(
                                tempsum_grid_1km_selected
                            )
                            # export temperature sum data for NUTS1 region
                            Path(
                                out_path
                                + "/"
                                + country
                                + "/avg_annual_temp_sum_"
                                + yearstr
                            ).mkdir(parents=True, exist_ok=True)
                            tempsum_grid_1km_selected.to_csv(
                                out_path
                                + "/"
                                + country
                                + "/avg_annual_temp_sum_"
                                + yearstr
                                + "/1kmgrid_"
                                + NUTS1
                                + ".csv"
                            )
                            print(
                                f"file '{out_path}/{country}/avg_annual_temp_sum_{yearstr}/1kmgrid_{NUTS1}.csv' was created successfully"
                            )

                        if (
                            find_veg_period
                        ):  # and not os.path.isfile(out_path + "/"+ country + "/avg_annual_veg_period_"+ yearstr + "/1kmgrid_"+ NUTS1 + ".csv"):
                            temperature_veg_period = ffd.get_annual_veg_period(
                                temperature_selected_years
                            )
                            temperature_veg_period_mean = (
                                temperature_veg_period[
                                    temperature_veg_period["year"].isin(
                                        climate_mean_years
                                    )
                                ][["GRID_NO", "TEMPERATURE_AVG"]]
                                .groupby("GRID_NO")
                                .mean()
                                .reset_index()
                            )
                            temperature_veg_period_mean.rename(
                                columns={
                                    "TEMPERATURE_AVG": "vegperiod_annual_mean_"
                                    + yearstr
                                },
                                inplace=True,
                            )
                            vegperiod_grid_1km_selected = pd.merge(
                                cellcode1km_gridno25km_df,
                                temperature_veg_period_mean,
                                how="left",
                                on="GRID_NO",
                            )
                            vegperiod_grid_1km_selected = pd.merge(
                                vegperiod_grid_1km_selected,
                                grid_1km_selected,
                                how="left",
                                on="CELLCODE",
                            )
                            vegperiod_grid_1km_selected = gpd.GeoDataFrame(
                                vegperiod_grid_1km_selected
                            )
                            Path(
                                out_path
                                + "/"
                                + country
                                + "/avg_annual_veg_period_"
                                + yearstr
                            ).mkdir(parents=True, exist_ok=True)
                            vegperiod_grid_1km_selected.to_csv(
                                out_path
                                + "/"
                                + country
                                + "/avg_annual_veg_period_"
                                + yearstr
                                + "/1kmgrid_"
                                + NUTS1
                                + ".csv"
                            )
                            print(
                                f"file '{out_path}/{country}/avg_annual_veg_period_{yearstr}/1kmgrid_{NUTS1}.csv' was created successfully"
                            )

                        if (
                            find_precipitation
                        ):  # and not os.path.isfile(out_path + "/"+ country+ "/avg_annual_precipitation_" + yearstr+ "/1kmgrid_"+ NUTS1 + ".csv"):
                            precipitation_selected_years = precipitation[
                                precipitation["year"].isin(climate_mean_years)
                            ]
                            precipitation_annual_sum = ffd.get_annual_precipitation_sum(
                                precipitation_selected_years
                            )
                            precipitation_annual_sum_mean = (
                                precipitation_annual_sum[
                                    precipitation_annual_sum["year"].isin(
                                        climate_mean_years
                                    )
                                ][["GRID_NO", "PRECIPITATION"]]
                                .groupby("GRID_NO")
                                .mean()
                                .reset_index()
                            )
                            precipitation_annual_sum_mean.rename(
                                columns={
                                    "PRECIPITATION": "precipitation_annual_mean_"
                                    + yearstr
                                },
                                inplace=True,
                            )
                            precipitation_grid_1km_selected = pd.merge(
                                cellcode1km_gridno25km_df,
                                precipitation_annual_sum_mean,
                                how="left",
                                on="GRID_NO",
                            )
                            precipitation_grid_1km_selected = pd.merge(
                                precipitation_grid_1km_selected,
                                grid_1km_selected,
                                how="left",
                                on="CELLCODE",
                            )
                            precipitation_grid_1km_selected = gpd.GeoDataFrame(
                                precipitation_grid_1km_selected
                            )
                            Path(
                                out_path
                                + "/"
                                + country
                                + "/avg_annual_precipitation_"
                                + yearstr
                            ).mkdir(parents=True, exist_ok=True)
                            precipitation_grid_1km_selected.to_csv(
                                out_path
                                + "/"
                                + country
                                + "/avg_annual_precipitation_"
                                + yearstr
                                + "/1kmgrid_"
                                + NUTS1
                                + ".csv"
                            )
                            print(
                                f"file '{out_path}/{country}/avg_annual_precipitation_{yearstr}/1kmgrid_{NUTS1}.csv' was created successfully"
                            )

                """SAND CONTENT"""
                if find_sand_content and not os.path.isfile(
                    out_path + "/" + country + "/sand/1kmgrid_" + NUTS1 + ".csv"
                ):

                    sand_share_grid = {
                        "CELLCODE": grid_1km_selected["CELLCODE"].values,
                        "sand_share": ffd.get_soil_content(
                            sand_path, grid_1km_selected, resolution=500
                        ),
                    }

                    sand_share_grid_df = pd.DataFrame(sand_share_grid)
                    sand_grid_1km_selected = pd.merge(
                        sand_share_grid_df, grid_1km_selected, how="left", on="CELLCODE"
                    )
                    sand_grid_1km_selected = gpd.GeoDataFrame(sand_grid_1km_selected)
                    sand_grid_1km_selected = sand_grid_1km_selected[
                        ["CELLCODE", "geometry", "sand_share"]
                    ]
                    Path(out_path + "/" + country + "/sand").mkdir(
                        parents=True, exist_ok=True
                    )
                    sand_grid_1km_selected.to_csv(
                        out_path + "/" + country + "/sand/1kmgrid_" + NUTS1 + ".csv"
                    )
                    print(
                        f"file '{out_path}/{country}/sand/1kmgrid_{NUTS1}.csv' was created successfully"
                    )

                """CLAY CONTENT"""
                if find_clay_content and not os.path.isfile(
                    out_path + "/" + country + "/clay/1kmgrid_" + NUTS1 + ".csv"
                ):

                    clay_share_grid = {
                        "CELLCODE": grid_1km_selected["CELLCODE"].values,
                        "clay_share": ffd.get_soil_content(
                            clay_path, grid_1km_selected, resolution=500
                        ),
                    }

                    clay_share_grid_df = pd.DataFrame(clay_share_grid)
                    clay_grid_1km_selected = pd.merge(
                        clay_share_grid_df, grid_1km_selected, how="left", on="CELLCODE"
                    )
                    clay_grid_1km_selected = gpd.GeoDataFrame(clay_grid_1km_selected)
                    clay_grid_1km_selected = clay_grid_1km_selected[
                        ["CELLCODE", "geometry", "clay_share"]
                    ]
                    Path(out_path + "/" + country + "/clay").mkdir(
                        parents=True, exist_ok=True
                    )
                    clay_grid_1km_selected.to_csv(
                        out_path + "/" + country + "/clay/1kmgrid_" + NUTS1 + ".csv"
                    )
                    print(
                        f"file '{out_path}/{country}/clay/1kmgrid_{NUTS1}.csv' was created successfully"
                    )

                """COARSE FRAGMENTS"""
                if find_coarse_fragments and not os.path.isfile(
                    out_path
                    + "/"
                    + country
                    + "/coarse_fragments/1kmgrid_"
                    + NUTS1
                    + ".csv"
                ):

                    coarse_fragments_grid = {
                        "CELLCODE": grid_1km_selected["CELLCODE"].values,
                        "coarse_fragments": ffd.get_soil_content(
                            coarse_fragments_path, grid_1km_selected, resolution=500
                        ),
                    }

                    coarse_fragments_grid_df = pd.DataFrame(coarse_fragments_grid)
                    coarse_fragments_grid_1km_selected = pd.merge(
                        coarse_fragments_grid_df,
                        grid_1km_selected,
                        how="left",
                        on="CELLCODE",
                    )
                    coarse_fragments_grid_1km_selected = gpd.GeoDataFrame(
                        coarse_fragments_grid_1km_selected
                    )
                    coarse_fragments_grid_1km_selected = (
                        coarse_fragments_grid_1km_selected[
                            ["CELLCODE", "geometry", "coarse_fragments"]
                        ]
                    )
                    Path(out_path + "/" + country + "/coarse_fragments").mkdir(
                        parents=True, exist_ok=True
                    )
                    coarse_fragments_grid_1km_selected.to_csv(
                        out_path
                        + "/"
                        + country
                        + "/coarse_fragments/1kmgrid_"
                        + NUTS1
                        + ".csv"
                    )
                    print(
                        f"file '{out_path}/{country}/coarse_fragments/1kmgrid_{NUTS1}.csv' was created successfully"
                    )

                """SILT CONTENT"""
                if find_silt_content and not os.path.isfile(
                    out_path + "/" + country + "/silt/1kmgrid_" + NUTS1 + ".csv"
                ):

                    silt_grid = {
                        "CELLCODE": grid_1km_selected["CELLCODE"].values,
                        "silt_share": ffd.get_soil_content(
                            silt_path, grid_1km_selected, resolution=500
                        ),
                    }

                    silt_grid_df = pd.DataFrame(silt_grid)
                    silt_grid_1km_selected = pd.merge(
                        silt_grid_df, grid_1km_selected, how="left", on="CELLCODE"
                    )
                    silt_grid_1km_selected = gpd.GeoDataFrame(silt_grid_1km_selected)
                    silt_grid_1km_selected = silt_grid_1km_selected[
                        ["CELLCODE", "geometry", "silt_share"]
                    ]
                    Path(out_path + "/" + country + "/silt").mkdir(
                        parents=True, exist_ok=True
                    )
                    silt_grid_1km_selected.to_csv(
                        out_path + "/" + country + "/silt/1kmgrid_" + NUTS1 + ".csv"
                    )
                    print(
                        f"file '{out_path}/{country}/silt/1kmgrid_{NUTS1}.csv' was created successfully"
                    )

                """AVAILABLE WATER CAPACITY"""
                if find_awc and not os.path.isfile(
                    out_path + "/" + country + "/awc/1kmgrid_" + NUTS1 + ".csv"
                ):

                    awc_grid = {
                        "CELLCODE": grid_1km_selected["CELLCODE"].values,
                        "awc_share": ffd.get_soil_content(
                            awc_path, grid_1km_selected, resolution=500
                        ),
                    }

                    awc_grid_df = pd.DataFrame(awc_grid)
                    awc_grid_1km_selected = pd.merge(
                        awc_grid_df, grid_1km_selected, how="left", on="CELLCODE"
                    )
                    awc_grid_1km_selected = gpd.GeoDataFrame(awc_grid_1km_selected)
                    awc_grid_1km_selected = awc_grid_1km_selected[
                        ["CELLCODE", "geometry", "awc_share"]
                    ]
                    Path(out_path + "/" + country + "/awc").mkdir(
                        parents=True, exist_ok=True
                    )
                    awc_grid_1km_selected.to_csv(
                        out_path + "/" + country + "/awc/1kmgrid_" + NUTS1 + ".csv"
                    )
                    print(
                        f"file '{out_path}/{country}/awc/1kmgrid_{NUTS1}.csv' was created successfully"
                    )

                """BULK DENSITY"""
                if find_bulk_density and not os.path.isfile(
                    out_path + "/" + country + "/bulk_density/1kmgrid_" + NUTS1 + ".csv"
                ):

                    bulk_density_grid = {
                        "CELLCODE": grid_1km_selected["CELLCODE"].values,
                        "bulk_density_share": ffd.get_soil_content(
                            bulk_density_path, grid_1km_selected, resolution=500
                        ),
                    }

                    bulk_density_grid_df = pd.DataFrame(bulk_density_grid)
                    bulk_density_grid_1km_selected = pd.merge(
                        bulk_density_grid_df,
                        grid_1km_selected,
                        how="left",
                        on="CELLCODE",
                    )
                    bulk_density_grid_1km_selected = gpd.GeoDataFrame(
                        bulk_density_grid_1km_selected
                    )
                    bulk_density_grid_1km_selected = bulk_density_grid_1km_selected[
                        ["CELLCODE", "geometry", "bulk_density_share"]
                    ]
                    Path(out_path + "/" + country + "/bulk_density").mkdir(
                        parents=True, exist_ok=True
                    )
                    bulk_density_grid_1km_selected.to_csv(
                        out_path
                        + "/"
                        + country
                        + "/bulk_density/1kmgrid_"
                        + NUTS1
                        + ".csv"
                    )
                    print(
                        f"file '{out_path}/{country}/bulk_density/1kmgrid_{NUTS1}.csv' was created successfully"
                    )

                if find_corine_class:
                    
                    """CORINE CLASS"""
                    for year in corine_years:
                        CORINE_CLASS = {
                            "CELLCODE": [],
                            "class": [],
                            "frequency": [],
                            "area": [],
                        }

                        corine_path = (
                            corine_basic_path+"clc"
                            + str(year)
                            + "_v2020_20u1_raster100m/DATA/"
                        )
                        for file in os.listdir(corine_path):
                            if file[-3:]=="tif":
                                break
                        corine_path=corine_path+file

                        CORINE_CLASS = {
                            "CELLCODE": grid_1km_selected["CELLCODE"].values,
                            # assign nan values to the following as long as it's not clear where they are needed
                            "class": np.repeat(np.nan, len(grid_1km_selected)),
                            "frequency": np.repeat(np.nan, len(grid_1km_selected)),
                            "area": np.array(grid_1km_selected.area),
                        }

                        corine_class_grid_df = pd.DataFrame(CORINE_CLASS)
                        corine_grid_1km_selected = pd.merge(
                            corine_class_grid_df,
                            grid_1km_selected,
                            how="left",
                            on="CELLCODE",
                        )
        


                        """AGRICULTURAL SHARE"""

                        # agricultural_classes=[12,13,14,15,16,17,18,19,20,21,22,26] <-- this was used previously but then discarded
                        # agricultural_classes = [12, 13, 14, 15, 16, 17, 18, 19, 20, 22] <-- this is used now
                        all_cells = list(
                            corine_class_grid_df["CELLCODE"].value_counts().keys()
                        )
                        agshare_dict = {
                            "CELLCODE": grid_1km_selected["CELLCODE"].values,
                            "agshare": ffd.get_agshare(corine_path, grid_1km_selected,corine_ag_classes),
                        }
                        agshare_grid_df = pd.DataFrame(agshare_dict)
                        agshare_grid_1km_selected = pd.merge(
                            agshare_grid_df,
                            grid_1km_selected,
                            how="left",
                            on="CELLCODE",
                        )
                        agshare_grid_1km_selected = agshare_grid_1km_selected[
                            ["CELLCODE", "geometry", "agshare"]
                        ]
                        Path(out_path + "/" + country + "/CORINE_agshare").mkdir(
                            parents=True, exist_ok=True
                        )
                        agshare_grid_1km_selected.to_csv(
                            out_path
                            + "/"
                            + country
                            + "/CORINE_agshare/1kmgrid_"
                            + NUTS1
                            + "_"
                            + str(year)
                            + ".csv"
                        )
                        print(
                            f"file '{out_path}/{country}/CORINE_agshare/1kmgrid_{NUTS1}_{str(year)}.csv' was created successfully"
                        )

                if find_latitude4326:
                    "LATITUDE EPSG:4326"
                    lat3035 = grid_1km_selected["NOFORIGIN"]
                    lon3035 = grid_1km_selected["EOFORIGIN"]
                    coords_grid_1km_selected_3035 = gpd.points_from_xy(
                        x=lon3035, y=lat3035, crs="epsg:3035"
                    )
                    coords_grid_1km_selected_4326 = (
                        coords_grid_1km_selected_3035.to_crs(crs="epsg:4326")
                    )
                    lon4326 = coords_grid_1km_selected_4326.x
                    lat4326 = coords_grid_1km_selected_4326.y
                    grid_1km_selected["lat4326"] = lat4326
                    latitude4326_grid_1km_selected = grid_1km_selected[
                        ["CELLCODE", "lat4326"]
                    ]
                    Path(out_path + "/" + country + "/latitude4326").mkdir(
                        parents=True, exist_ok=True
                    )
                    latitude4326_grid_1km_selected.to_csv(
                        out_path
                        + "/"
                        + country
                        + "/latitude4326/1kmgrid_"
                        + NUTS1
                        + ".csv"
                    )
                    print(
                        f"file '{out_path}/{country}/latitude4326/1kmgrid_{NUTS1}.csv' was created successfully"
                    )

                """INFER UAA"""
                if calculate_UAA and not os.path.isfile(
                    out_path + "/" + country + "/inferred_UAA/1kmgrid_" + NUTS1 + ".csv"
                ):
                    if country_levls[country] == 3:
                        relevant_nuts = NUTS3_regs[
                            np.where(NUTS3_regs.astype("U3") == NUTS1)[0]
                        ]

                    elif country_levls[country] == 2:
                        relevant_nuts = NUTS2_regs[
                            np.where(NUTS2_regs.astype("U3") == NUTS1)[0]
                        ]

                    cellsize_nuts1_df = pd.DataFrame()
                    for nuts in relevant_nuts:
                        df = pd.read_csv(
                            zonal_stats_path
                            + country
                            + "/cell_size/1kmgrid_"
                            + nuts
                            + "_all_years.csv"
                        )
                        df[f"nuts{country_levls[country]}"] = np.repeat(nuts, len(df))
                        cellsize_nuts1_df = pd.concat((cellsize_nuts1_df, df))

                    agshare_nuts1_df = pd.DataFrame()
                    for year in corine_years:
                        df = pd.read_csv(
                            zonal_stats_path
                            + country
                            + "/CORINE_agshare/1kmgrid_"
                            + NUTS1
                            + "_"
                            + str(year)
                            + ".csv"
                        )
                        df["year"] = np.repeat(year, len(df))
                        agshare_nuts1_df = pd.concat((agshare_nuts1_df, df))

                    inferred_UAA_df = pd.DataFrame()
                    for nuts in relevant_nuts:
                        for y, year in enumerate(selected_years):
                            cyear = relevant_corine_years[y]
                            nyear = relevant_nuts_years[y]
                            selected_cellsize_df = cellsize_nuts1_df[
                                (cellsize_nuts1_df["year"] == nyear)
                                & (
                                    cellsize_nuts1_df[f"nuts{country_levls[country]}"]
                                    == nuts
                                )
                            ]
                            if len(selected_cellsize_df) > 0:

                                cellcodes = np.sort(
                                    selected_cellsize_df["CELLCODE"]
                                    .value_counts()
                                    .keys()
                                )

                                selected_agshare_df = agshare_nuts1_df[
                                    (agshare_nuts1_df["year"] == cyear)
                                    & (agshare_nuts1_df["CELLCODE"].isin(cellcodes))
                                ]

                                df = pd.merge(
                                    selected_agshare_df[["CELLCODE", "agshare"]],
                                    selected_cellsize_df[["CELLCODE", "area"]],
                                    how="left",
                                    on="CELLCODE",
                                )
                                df[f"nuts{country_levls[country]}"] = np.repeat(
                                    nuts, len(df)
                                )
                                df["year"] = np.repeat(year, len(df))
                                inferred_UAA_df = pd.concat((inferred_UAA_df, df))

                    if len(selected_cellsize_df)>0:

                        inferred_UAA_df["inferred_UAA"] = (
                            inferred_UAA_df["agshare"] * inferred_UAA_df["area"]
                        )

                        Path(out_path + "/" + country + "/inferred_UAA").mkdir(
                            parents=True, exist_ok=True
                        )
                        inferred_UAA_df.to_csv(
                            out_path
                            + "/"
                            + country
                            + "/inferred_UAA/1kmgrid_"
                            + NUTS1
                            + ".csv"
                        )
                        print(
                            f"file '{out_path}/{country}/inferred_UAA/1kmgrid_{NUTS1}.csv' was created successfully"
                        )

# %%

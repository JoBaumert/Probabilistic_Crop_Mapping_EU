#%%
import argparse
from cgi import test
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly import data
from shapely.geometry import Point
from sklearn.utils import column_or_1d
from scipy.special import gamma
from pathlib import Path
import os, zipfile
#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]

country = "FR" #default is France in 2018
year = 2018

raw_data_path = data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
parameter_path=data_main_path+"delineation_and_parameters/DGPCM_user_parameters.xlsx"
nuts_path=intermediary_data_path+"Preprocessed_Inputs/NUTS/NUTS_all_regions_all_years.shp"
excluded_NUTS_regions_path = data_main_path+"delineation_and_parameters/excluded_NUTS_regions.xlsx"
crop_delineation_path=data_main_path+"delineation_and_parameters/DGPCM_crop_delineation.xlsx"
IACS_data_path = raw_data_path+"IACS/"+country+"_"+str(year)+".zip!"
grid_path=raw_data_path+"Grid/"
cellsize_path=intermediary_data_path+"Zonal_Stats/"

output_path=intermediary_data_path+"Preprocessed_Inputs/IACS/"
#%%

parser = argparse.ArgumentParser()
parser.add_argument("-cc", "--ccode", type=str, required=False)
parser.add_argument("-y", "--year", type=int, required=False)

args = parser.parse_args()
if args.ccode is not None:
    country=args.ccode
    year=args.year
    

# %%
data_gpd = gpd.read_file(IACS_data_path+country)

# %%
HCAT_conversion = pd.read_excel(crop_delineation_path, sheet_name="HCAT_DGPCM")
#%%

data_gpd.rename(columns={"EC_hcat_c": "HCAT2_code"}, inplace=True)
#%%
HCAT_conversion["HCAT2_code"] = np.char.mod(
    "%d", np.array(HCAT_conversion["HCAT2_code"])
)
# %%
data_gpd = pd.merge(
    data_gpd, HCAT_conversion[["HCAT2_code", "DGPCM_code"]], how="left", on="HCAT2_code"
)
#%%
nuts_years=pd.read_excel(parameter_path,sheet_name="NUTS")
relevant_nuts_year=nuts_years[nuts_years["crop_map_year"]==year]["nuts_year"].iloc[0]
#%%
"""attribute each field to (at least) one km2 cell"""

NUTS = gpd.read_file(nuts_path)
# load all 1km grid-cells that belong to selected country
grid_1km_path_country = (
            # "zip+file://"
                grid_path
                + country
                +"_1km.zip"     
                )

zip=zipfile.ZipFile(grid_1km_path_country)
for file in zip.namelist():
    if file[-3:]=="shp":
        break


grid_1km_country = gpd.read_file(grid_1km_path_country+"!/"+file)


# %%
# assign unique ID to each cell to avoid confusion of cells
data_gpd["unique_ID"] = ["ID_" + str(x) for x in np.arange(len(data_gpd))]

#%%
# for France, the original crs is 2154
data_gdf_epsg2154 = gpd.GeoDataFrame(data_gpd, crs=2154, geometry="geometry")
data_gdf_epsg3035 = data_gdf_epsg2154.to_crs("epsg:3035")
# %%
data_gdf_epsg3035.head()
#%%
data_gdf_epsg3035.rename(columns={"SURF_PARC": "area_original_ha"}, inplace=True)
data_gdf_epsg3035.dropna(subset=["geometry"], inplace=True)
#%%
# find the relevant NUTSx regions and exclude those belonging to regions oversears (e.g., for France)
excluded_NUTS_regions = pd.read_excel(excluded_NUTS_regions_path)
excluded_NUTS1_regions = np.array(excluded_NUTS_regions["excluded NUTS1 regions"])
NUTS2_df = NUTS[(NUTS["CNTR_CODE"] == country) & (NUTS["LEVL_CODE"] == 2)&(NUTS["year"]==year)]
NUTS2_df["NUTS1"] = np.array(NUTS2_df["NUTS_ID"]).astype("U3")
relevant_NUTS2_regs = np.array(
    NUTS2_df[~NUTS2_df["NUTS1"].isin(excluded_NUTS1_regions)]["NUTS_ID"]
)


#%%
for region in relevant_NUTS2_regs:
    cells = pd.read_csv(cellsize_path + country+"/cell_size/1kmgrid_"+region + "_all_years.csv")

    selected_cells = grid_1km_country[
        grid_1km_country["CELLCODE"].isin(cells[cells["year"] == relevant_nuts_year]["CELLCODE"])
    ]

    """ONLY SUBSET OF CELLS IN SELECTED REGION"""
    cellcodes_df = selected_cells.overlay(data_gdf_epsg3035, how="intersection")

    cellcodes_df["area_in_cell"] = cellcodes_df.area
    #%%
    """calculate cropshare for each cell"""
    crops_in_cells = cellcodes_df.groupby(["CELLCODE", "DGPCM_code"])["area_in_cell"].sum().reset_index()

    total_area_in_cells = crops_in_cells.groupby("CELLCODE").sum().reset_index()
    total_area_in_cells.rename(
        columns={"area_in_cell": "total_area_in_cells"}, inplace=True
    )

    crops = sorted(list(cellcodes_df["DGPCM_code"].value_counts().keys()))
    all_cells = selected_cells["CELLCODE"].value_counts().keys()
    all_cells_array = np.array(
        [np.repeat(cell, len(crops)) for cell in all_cells]
    ).flatten()
    all_crops_array = np.tile(crops, len(all_cells))
    all_crops_and_cells_df = pd.DataFrame(
        np.array(np.array([all_cells_array, all_crops_array]).transpose()),
        columns=["CELLCODE", "DGPCM_code"],
    )

    crops_in_cells = pd.merge(
        all_crops_and_cells_df,
        crops_in_cells,
        how="left",
        on=["CELLCODE", "DGPCM_code"],
    )

    # replace nan values in the "area_in_cell" column by 0, since nan here means 0
    crops_in_cells["area_in_cell"] = crops_in_cells["area_in_cell"].fillna(0)
    crops_in_cells = pd.merge(
        crops_in_cells, total_area_in_cells, how="left", on="CELLCODE"
    )

    crops_in_cells["cropshare_true"] = (
        crops_in_cells["area_in_cell"] / crops_in_cells["total_area_in_cells"]
    )

    """merge with geo-coordinate data"""

    crops_in_cells = pd.merge(crops_in_cells, selected_cells, how="left", on="CELLCODE")
    crops_in_cells = gpd.GeoDataFrame(crops_in_cells)

    """export data"""
    print("export data for region "+region)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    crops_in_cells.to_csv(output_path +region + "_" + str(year) + ".csv")
# %%
"""
plt.figure()
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
crops_in_cells[crops_in_cells["DGPCM_code"] == "LMAIZ"].plot(
    ax=ax, column="cropshare_true", legend=True, cmap="YlOrRd", vmin=0, vmax=1
)
NUTS[NUTS["NUTS_ID"] == "FRF1"].plot(ax=ax, facecolor="None")
"""
# %%

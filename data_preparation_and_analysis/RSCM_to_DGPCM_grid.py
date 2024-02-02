# %%
from cgi import test

# from re import S
import geopandas as gpd
import math
import numpy as np

import argparse
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show
from rasterstats import zonal_stats, point_query
from rasterio.windows import from_bounds
import os
import richdem as rd
from os.path import exists
import zipfile
from pathlib import Path
from scipy import stats
# %%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]

""" THIS CODE IS USED TO AGREGATE THE CROP DATA FROM THE REMOTE SENSING CROP MAP (RSCM) BY D'ANDRIMONT ET AL (2021) TO OUR GRID CELLS
their crop map has a resolution of 10m so in this code we identify for each of our 1km grid cells
 those pixels that are within the boundaries of our grid cell and aggregate them.
"""
year=2018 # the only year considered by RSCM
country="FR" #default, as France is the only country for which IACS data in 2018 is available
country_list=[country]
raw_data_path = data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
parameter_path=data_main_path+"delineation_and_parameters/DGPCM_user_parameters.xlsx"
nuts_path=intermediary_data_path+"Preprocessed_Inputs/NUTS/NUTS_all_regions_all_years.csv"
excluded_NUTS_regions_path = data_main_path+"delineation_and_parameters/excluded_NUTS_regions.xlsx"
crop_delineation_path=data_main_path+"delineation_and_parameters/DGPCM_crop_delineation.xlsx"
cellsize_path=intermediary_data_path+"Zonal_Stats/"
RSCM_path=raw_data_path+"RSCM/EUCROPMAP_2018.tif"
grid_path=raw_data_path+"Grid/"
#output path
output_path=intermediary_data_path+"Preprocessed_Inputs/RSCM/"
#%%
# import parameters
selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])
#%%

parser = argparse.ArgumentParser()
parser.add_argument("-cc", "--ccode", type=str, required=False)
parser.add_argument("--from_xlsx", type=str, required=False)

args = parser.parse_args()
if args.from_xlsx == "True":
    country_list = [country_codes_relevant]
elif args.ccode is not None:
    country_list = [args.ccode]

#%%
cropname_conversion_file=pd.read_excel(crop_delineation_path)
#%%
#%%
cropname_conversion=cropname_conversion_file[["DGPCM_code","RSCM"]].drop_duplicates()
#%%
RSCM_codes_unique=np.sort(np.unique(np.array(cropname_conversion["RSCM"])[np.where(cropname_conversion["RSCM"]>0)[0]]))
DGPCM_codes=np.array(cropname_conversion[cropname_conversion["RSCM"].isin(RSCM_codes_unique)].drop_duplicates("RSCM")["DGPCM_code"])
DGPCM_codes_unique=np.array(cropname_conversion[cropname_conversion["DGPCM_code"].isin(DGPCM_codes)].drop_duplicates("DGPCM_code")["DGPCM_code"])
#%%
nuts_years=pd.read_excel(parameter_path,sheet_name="NUTS")
relevant_nuts_year=nuts_years[nuts_years["crop_map_year"]==year]["nuts_year"].iloc[0]
excluded_NUTS_regions = pd.read_excel(excluded_NUTS_regions_path)
excluded_NUTS_regions = np.array(
    excluded_NUTS_regions["excluded NUTS1 regions"]
)
nuts_input = pd.read_csv(nuts_path)


# %%
# file paths
for c in range(len(country_list)):
    country = country_list[c]

    cell_size_path_country=cellsize_path+country+"/cell_size/1kmgrid_"
    #%%
    
 
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
    def blockshaped(arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
        assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                .swapaxes(1,2)
                .reshape(-1, nrows, ncols))

    

    #%%
    if __name__ == "__main__":
        for selected_region in nuts1_unique:
            
            print(f"start remapping data for region {selected_region}")
            cell_size_data=pd.read_csv(cell_size_path_country+selected_region+"_all_years.csv")

            relevant_cells=np.unique(cell_size_data[cell_size_data["year"]==relevant_nuts_year]["CELLCODE"]).astype(str)


            """
            IMPORTANT: THE CELLCODE REFERS TO THE LEFT BOTTOM CORNER OF A CELL
            """
            intermediate_str=np.array([i[1] for i in np.array(np.char.split(relevant_cells,"E"))]).astype(str)
            east=np.array([i[0] for i in np.char.split(intermediate_str,"N")]).astype(int)
            north=np.array([i[1] for i in np.char.split(intermediate_str,"N")]).astype(int)


            """
            use the minum and maximum east and north values as the boundaries for loadingt the raster file
            (loading the entire raster file exceeds memory)
            """
            left,bottom,right,top=east.min(),north.min(),east.max()+1,north.max()+1

            with rio.open(
                RSCM_path
            ) as src:
                rst = src.read(
                    1, window=from_bounds(left*1000, bottom*1000, right*1000, top*1000, src.transform)
                )


            """
            Divide the raster evenly in blocks of 100x100 pixel. The reference cells have a length and a width of 1000m, the 
            Remote Sensing based Map has a pixel size of 10x10m. Therefore, each block contains the pixels that belong to one 1km grid cell
            """
            blocks=blockshaped(rst,100,100)

            east_positions=np.tile(np.arange(left,right,1),int(rst.shape[0]/100))
            north_positions=np.repeat(np.arange(top-1,bottom-1,-1),int(rst.shape[1]/100))
            rst_cells=np.char.add(east_positions.astype(str),north_positions.astype(str))
            rst_cells_cellcode=np.char.add(np.char.add(np.char.add("1kmE",east_positions.astype(str)),"N"),north_positions.astype(str))
            reference_cells_region=np.char.add(east.astype(str),north.astype(str))


            relevant_blocks_indices=np.where(np.isin(rst_cells,reference_cells_region))[0]

            relevant_blocks=blocks[relevant_blocks_indices]
            relevant_cells_cellcode=rst_cells_cellcode[relevant_blocks_indices]

            #%%
            """
            from the relevant blocks, retrieve the number of pixels in each block (=cell) and then divide them by all crop pixels to get the share
            """
            B=np.zeros((len(RSCM_codes_unique),len(reference_cells_region)))
            for c,crop in enumerate(RSCM_codes_unique):
                cell,counts=np.unique(np.where(relevant_blocks==crop)[0],return_counts=True)
                B[c][cell]=counts
        
            B=B.transpose()
            eucm_crop_shares=np.array([b/(np.nansum(b)+0.00000001) for b in B])
            df_eucm=pd.DataFrame(eucm_crop_shares,columns=RSCM_codes_unique)
        
            
            df_eucm.insert(loc=0,column="CELLCODE",value=relevant_cells_cellcode)

            #%%            
            #export the data
            print("export data")
            Path(output_path + country).mkdir(parents=True, exist_ok=True)
            df_eucm.to_csv(
                output_path + country +"/"+ selected_region+ "_1km_reference_grid.csv"
            )


# %%

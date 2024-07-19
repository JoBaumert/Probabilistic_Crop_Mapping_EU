#%%
import rasterio as rio
from rasterio import features
from rasterio.plot import show
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
from rasterio.windows import from_bounds
import geopandas as gpd
import pandas as pd
from pathlib import Path
import os
import numpy as np
import zipfile
import xarray
import gc
import matplotlib.pyplot as plt

#%%
#just some stuff to find the files in my file system... can be commented out
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
results_path=data_main_path+"Results/Simulated_consistent_crop_shares/"
#%%
#read one existing raster file in to get the "transform", i.e., corners and pixel sizes
# indicating the desired shape of the file you want to create
raster_reference_cells=rio.open(
    data_main_path+"Intermediary_Data/Preprocessed_Inputs/NUTS/NUTS2_2016_raster.tif"
)
transform=raster_reference_cells.transform
#%%
#look at coordinate reference system like this:
raster_reference_cells.crs
#%%
#look at bounds like this:
raster_reference_cells.bounds
#%%
# read the data (i.e., get numpy array with the content of the raster file) like this:
raster_reference_cells_content=raster_reference_cells.read()
#%%
raster_reference_cells_content.shape # --> has one "band", (the first dimension)
#%%
#load CORINE data...


# %%
"""
this is how you export raster files:

with rio.open(data_main_path+"Intermediary_Data/Preprocessed_Inputs/NUTS/NUTS2_2016_raster.tif", 'w',
            width=int(raster_reference_cells.shape[1]),height=int(raster_reference_cells.shape[0]),
            transform=transform,count=1,dtype=rio.int16,crs="EPSG:3035") as dst:
    dst.write(NUTS2_raster.astype(rio.int16))


"""
# %%

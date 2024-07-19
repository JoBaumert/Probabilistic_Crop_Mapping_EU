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
# %%
"""
NUTS=gpd.read_file(r"N:\ds\priv\baumert\Data\NUTS\NUTS_RG_01M_2016_3035.shp.zip")
# %%
EU_raster=rio.open(
    r"F:\baumert\project1\Data\Results\Simulated_consistent_crop_shares\EU\expected_crop_share_entire_EU_2020.tif"
)
# %%
transform=EU_raster.transform

# %%

# %%
EU_raster_data=EU_raster.read()
#%%
fig, ax = plt.subplots( figsize=(10,10))
raster=show(EU_raster_data[1],ax=ax,cmap="YlOrRd")
#NUTS0.plot(ax=raster,color="grey")
plt.show()

#%%
NUTS0.transpose
# %%
"""
#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
results_path=data_main_path+"Results/Simulated_consistent_crop_shares/"
#%%
EU_raster=rio.open(
    results_path+"EU/expected_crop_share_entire_EU_2010.tif"
)
transform=EU_raster.transform

EU_raster_read=EU_raster.read()
#%%
#import NUTS data
NUTS=gpd.read_file(data_main_path+"Raw_Data/NUTS/NUTS_RG_01M_2016_3035.shp.zip")
#%%
NUTS0=gpd.GeoDataFrame(NUTS[NUTS["LEVL_CODE"]==0])
NUTS1=gpd.GeoDataFrame(NUTS[NUTS["LEVL_CODE"]==1])
NUTS2=gpd.GeoDataFrame(NUTS[NUTS["LEVL_CODE"]==2])
NUTS3=gpd.GeoDataFrame(NUTS[NUTS["LEVL_CODE"]==3])
# %%
NUTS0_regs=np.unique(NUTS0["NUTS_ID"])
NUTS0_index=np.arange(len(NUTS0_regs))+1
NUTS0.sort_values(by="NUTS_ID",inplace=True)
NUTS0["country_index"]=NUTS0_index
NUTS1_regs=np.unique(NUTS1["NUTS_ID"])
NUTS1_index=np.arange(len(NUTS1_regs))+1
NUTS1.sort_values(by="NUTS_ID",inplace=True)
NUTS1["country_index"]=NUTS1_index
NUTS2_regs=np.unique(NUTS2["NUTS_ID"])
NUTS2_index=np.arange(len(NUTS2_regs))+1
NUTS2.sort_values(by="NUTS_ID",inplace=True)
NUTS2["country_index"]=NUTS2_index
NUTS3_regs=np.unique(NUTS3["NUTS_ID"])
NUTS3_index=np.arange(len(NUTS3_regs))+1
NUTS3.sort_values(by="NUTS_ID",inplace=True)
NUTS3["country_index"]=NUTS3_index


#%%
geom_value = ((geom,value) for geom, value in zip(NUTS0.geometry, NUTS0.country_index))
NUTS0_raster=features.rasterize(
    geom_value,
    out_shape=EU_raster.shape,
    transform=transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS1.geometry, NUTS1.country_index))
NUTS1_raster=features.rasterize(
    geom_value,
    out_shape=EU_raster.shape,
    transform=transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS2.geometry, NUTS2.country_index))
NUTS2_raster=features.rasterize(
    geom_value,
    out_shape=EU_raster.shape,
    transform=transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS3.geometry, NUTS3.country_index))
NUTS3_raster=features.rasterize(
    geom_value,
    out_shape=EU_raster.shape,
    transform=transform,
)
#%%
#get relevasnt grid as shapefile

"""create one large grid shapefile"""
grid_path="/home/baumert/fdiexchange/baumert/project1/Data/Raw_Data/Grid/"
all_grids=pd.DataFrame()
i=0
#test=["Switzerland_shapefile.zip","DE_1km.zip","AT_1km.zip"]
for directory in os.listdir(grid_path):
   # directory=test[i]
   # i+=1
    if directory[-3:]=="zip":
        for file in zipfile.ZipFile(grid_path+directory).namelist():
            if file[-7:]=="1km.shp":
                print(directory)
                all_grids=pd.concat((
                    all_grids,
                    gpd.read_file(grid_path+directory+"!/"+file)
                ))
        

all_grids.drop_duplicates("CELLCODE",inplace=True)


#%%
crop=24
east=((np.array(all_grids.EOFORIGIN)-EU_raster.bounds[0])/1000).astype(int)
north=(np.abs((np.array(all_grids.NOFORIGIN)-EU_raster.bounds[3])/1000)).astype(int)
all_grids["on_land"]=np.where(NUTS0_raster[north,east]>0,1,0)
all_grids["crop_share"]=EU_raster_read[crop][north,east]
all_grids["UAA"]=EU_raster_read[0][north,east]
all_grids=all_grids[all_grids["on_land"]==1]
all_grids=all_grids[all_grids["UAA"]>0]

all_grids=gpd.GeoDataFrame(all_grids)
#%%
NUTS0_regs
#%%
selected_cmap="YlOrRd"
max_val=600
plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
NUTS0[NUTS0["CNTR_CODE"].isin(["DE","AT","CH"])].plot(ax=ax,facecolor="lightgrey")
gpd.GeoDataFrame(all_grids).plot(ax=ax,column="crop_share",
            legend=True,
            cmap=selected_cmap,  # YlGn "YlOrRd"
            vmin=0,
            vmax=max_val,)
NUTS0[NUTS0["CNTR_CODE"].isin(["DE","AT","CH"])].boundary.plot(ax=ax,edgecolor="darkgrey",linewidth=0.5)

#%%
"""LOAD HDI DATA FOR SELECTED CROP AND YEAR"""
crop,year="SWHE",2010
HDI_data=rio.open(results_path+"HDIs/"+crop+"_90%HDIs_entire_EU_"+str(year)+".tif").read()
#%%
HDI_width=HDI_data[1]-HDI_data[0]
#%%
east=((np.array(all_grids.EOFORIGIN)-EU_raster.bounds[0])/1000).astype(int)
north=(np.abs((np.array(all_grids.NOFORIGIN)-EU_raster.bounds[3])/1000)).astype(int)
all_grids["HDI_width"]=HDI_width[north,east]
#%%
HDI_width.max()
#%%
selected_cmap="YlOrRd"
max_val=800
plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
NUTS0[NUTS0["CNTR_CODE"].isin(["DE","AT","CH"])].plot(ax=ax,facecolor="lightgrey")
gpd.GeoDataFrame(all_grids).plot(ax=ax,column="HDI_width",
            legend=True,
            cmap=selected_cmap,  # YlGn "YlOrRd"
            vmin=0,
            vmax=max_val,)
NUTS0[NUTS0["CNTR_CODE"].isin(["DE","AT","CH"])].boundary.plot(ax=ax,edgecolor="darkgrey",linewidth=0.5)
#%%
gpd.GeoDataFrame(all_grids).plot(column="crop_share")
# %%
east_lower_left=np.tile(np.arange(EU_raster.bounds[0],EU_raster.bounds[2],step=1000),EU_raster.shape[0]).reshape(EU_raster.shape)
north_lower_left=np.repeat(np.arange(EU_raster.bounds[3],EU_raster.bounds[1],step=-1000),EU_raster.shape[1]).reshape(EU_raster.shape)
cellcode_grid=np.char.add(east_lower_left.astype("U4"),north_lower_left.astype("U4"))

#%%
""""""
NUTS0_regs
#%%
NUTS0.plot(facecolor="lightgrey")
#%%
NUTS0[NUTS0["CNTR_CODE"].isin(["DE","AT","CH"])].plot(ax=ax,facecolor="lightgrey")
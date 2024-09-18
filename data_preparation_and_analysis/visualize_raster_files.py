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
"""
this script allows to visualize the raster files for the entire EU at once
"""

data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
results_path=data_main_path+"Results/Simulated_consistent_crop_shares/"
poterior_proba_path=data_main_path+"Results/Posterior_crop_probability_estimates/"
raw_data_path = data_main_path+"Raw_Data/"
output_path=data_main_path+"Results/Validations_and_Visualizations/Expected_crop_shares/"
albania_shapefile_path=raw_data_path+"NUTS/albania_shapefile.zip!/ALB_adm0.shp"
bosnia_shapefile_path=raw_data_path+"NUTS/bosnia_shapefile.zip!/BIH_adm0.shp"
kosovo_shapefile_path=raw_data_path+"NUTS/kosovo_shapefile.zip!/XKO_adm0.shp"
serbia_shapefile_path=raw_data_path+"NUTS/serbia_shapefile.zip!/SRB_adm0.shp"
#%%

EU_raster=rio.open(
    results_path+"EU/expected_crop_share_entire_EU_2015.tif"
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
#add shapefiles for albania, bosnia, kosovo, serbia
albania_boundary=gpd.read_file(albania_shapefile_path)
bosnia_boundary=gpd.read_file(bosnia_shapefile_path)
kosovo_boundary=gpd.read_file(kosovo_shapefile_path)
serbia_boundary=gpd.read_file(serbia_shapefile_path)

albania_boundary_epsg3035=albania_boundary.to_crs("epsg:3035")
bosnia_boundary_epsg3035=bosnia_boundary.to_crs("epsg:3035")
kosovo_boundary_epsg3035=kosovo_boundary.to_crs("epsg:3035")
serbia_boundary_epsg3035=serbia_boundary.to_crs("epsg:3035")

albania_boundary_epsg3035=albania_boundary_epsg3035["geometry"]
bosnia_boundary_epsg3035=bosnia_boundary_epsg3035["geometry"]
kosovo_boundary_epsg3035=kosovo_boundary_epsg3035["geometry"]
serbia_boundary_epsg3035=serbia_boundary_epsg3035["geometry"]

albania_boundary_epsg3035_df=pd.DataFrame(albania_boundary_epsg3035)
albania_boundary_epsg3035_df.insert(0,"CNTR_CODE","AL")
bosnia_boundary_epsg3035_df=pd.DataFrame(bosnia_boundary_epsg3035)
bosnia_boundary_epsg3035_df.insert(0,"CNTR_CODE","BA")
kosovo_boundary_epsg3035_df=pd.DataFrame(kosovo_boundary_epsg3035)
kosovo_boundary_epsg3035_df.insert(0,"CNTR_CODE","XK")
serbia_boundary_epsg3035_df=pd.DataFrame(serbia_boundary_epsg3035)
serbia_boundary_epsg3035_df.insert(0,"CNTR_CODE","RS")

#%%
NUTS0=NUTS0[["CNTR_CODE","geometry"]]
NUTS0=pd.concat((NUTS0,albania_boundary_epsg3035_df))
NUTS0=pd.concat((NUTS0,kosovo_boundary_epsg3035_df))
NUTS0=pd.concat((NUTS0,bosnia_boundary_epsg3035_df))
NUTS0=pd.concat((NUTS0,serbia_boundary_epsg3035_df))
#%%
#get relevasnt grid as shapefile

"""create one large grid shapefile"""
grid_path="/home/baumert/fdiexchange/baumert/project1/Data/Raw_Data/Grid/"
all_grids=pd.DataFrame()
i=0
#test=["DK_1km.zip","DE_1km.zip","AT_1km.zip"]
for directory in os.listdir(grid_path):
 #   directory=test[i]
  #  i+=1
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
#bands are the same for all countries and years, so any can be used
bands=pd.read_csv(results_path+"/AT/AT2010simulated_cropshare_10reps_bands.csv")

all_crops=[]
for i in np.char.split(np.array(bands["name"].iloc[2:30]).astype(str)):
    #crop name starts after 15th character
    all_crops.append(i[0][15:])
all_crops=np.array(all_crops)


#%%
all_grids=all_grids[(all_grids["EOFORIGIN"]>EU_raster.bounds[0])&
          (all_grids["EOFORIGIN"]<EU_raster.bounds[2])&
          (all_grids["NOFORIGIN"]>EU_raster.bounds[1])&
          (all_grids["NOFORIGIN"]<EU_raster.bounds[3])]
#%%
EU_raster_read.shape
#%%
selected_crop="LMAIZ"
year=2010
if year!=2010:
    EU_raster_read=rio.open(
    results_path+"EU/expected_crop_share_entire_EU_"+str(year)+".tif"
    ).read()
crop_grid=all_grids.copy()
east=((np.array(crop_grid.EOFORIGIN)-EU_raster.bounds[0])/1000).astype(int)
north=(np.abs((np.array(crop_grid.NOFORIGIN)-EU_raster.bounds[3])/1000)).astype(int)
crop_grid["on_land"]=np.where(NUTS0_raster[north,east]>0,1,0)
crop_grid["crop_share"]=EU_raster_read[np.where(all_crops==selected_crop)[0][0]+1][north,east]
crop_grid["UAA"]=EU_raster_read[0][north,east]
crop_grid=crop_grid[crop_grid["on_land"]==1]
crop_grid=crop_grid[crop_grid["UAA"]>0]

crop_grid=gpd.GeoDataFrame(crop_grid)

#%%
if selected_crop=="GRAS":
    selected_cmap="YlGn"
    max_val=1000
elif selected_crop=="SWHE":
    selected_cmap="YlOrRd"
    max_val=600
elif selected_crop=="LMAIZ":
    selected_cmap="Blues"
    max_val=600



plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
NUTS0.plot(ax=ax,facecolor="lightgrey")
gpd.GeoDataFrame(crop_grid).plot(ax=ax,column="crop_share",
            legend=True,
            cmap=selected_cmap,  # YlGn "YlOrRd"
            vmin=0,
            vmax=max_val,)
NUTS0.boundary.plot(ax=ax,edgecolor="darkgrey",linewidth=0.5)
ax.set_xlim(EU_raster.bounds[0],EU_raster.bounds[2])
ax.set_ylim(EU_raster.bounds[1],EU_raster.bounds[3])

plt.axis("off")
Path(output_path).mkdir(parents=True, exist_ok=True)
plt.savefig(output_path+"share_of_"+selected_crop+"_"+str(year)+".png")
plt.close(fig)
#%%
show(EU_raster_read[np.where(all_crops==selected_crop)[0][0]+1])
#%%

#%%
show(np.where(EU_raster_read[6]>100,1,0))

#%%
"""LOAD MEAN HDI DATA"""

HDI_data=rio.open(results_path+"HDIs/mean_HDI_width_all_crops_all_years.tif").read()
#%%
show(HDI_data)
#%%
HDI_grid=all_grids.copy()
east=((np.array(all_grids.EOFORIGIN)-EU_raster.bounds[0])/1000).astype(int)
north=(np.abs((np.array(all_grids.NOFORIGIN)-EU_raster.bounds[3])/1000)).astype(int)
HDI_grid["HDI_width"]=HDI_data[0][north,east]/1000
#remove all observations that are zero (no agricultural land)
HDI_grid=HDI_grid[HDI_grid["HDI_width"]>0]
#%%
plt.hist(HDI_grid.HDI_width,bins=50)
#%%
np.quantile(HDI_grid.HDI_width,0.95)
#%%
selected_cmap="YlOrRd"
max_val=0.18
plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
NUTS0.plot(ax=ax,facecolor="lightgrey")
gpd.GeoDataFrame(HDI_grid).plot(ax=ax,column="HDI_width",
            legend=True,
            cmap=selected_cmap,  # YlGn "YlOrRd"
            vmin=0,
            vmax=max_val,)
NUTS0.boundary.plot(ax=ax,edgecolor="darkgrey",linewidth=0.5)
ax.set_xlim(EU_raster.bounds[0],EU_raster.bounds[2])
ax.set_ylim(EU_raster.bounds[1],EU_raster.bounds[3])

plt.axis("off")
Path(output_path).mkdir(parents=True, exist_ok=True)
plt.savefig(output_path+"mean_HDI_width_all_years.png")
plt.close(fig)
#%%
gpd.GeoDataFrame(all_grids).plot(column="crop_share")
# %%
east_lower_left=np.tile(np.arange(EU_raster.bounds[0],EU_raster.bounds[2],step=1000),EU_raster.shape[0]).reshape(EU_raster.shape)
north_lower_left=np.repeat(np.arange(EU_raster.bounds[3],EU_raster.bounds[1],step=-1000),EU_raster.shape[1]).reshape(EU_raster.shape)
cellcode_grid=np.char.add(east_lower_left.astype("U4"),north_lower_left.astype("U4"))

#%%
"""posterior proba uncertainty"""
posterior_range=rio.open(poterior_proba_path+"Posterior_range/EU_posterior_range_all_crops_2020.tif").read()

# %%
posterior_range=posterior_range/1000
# %%
posterior_range_grid=all_grids.copy()
east=((np.array(all_grids.EOFORIGIN)-EU_raster.bounds[0])/1000).astype(int)
north=(np.abs((np.array(all_grids.NOFORIGIN)-EU_raster.bounds[3])/1000)).astype(int)
posterior_range_grid["posterior_range"]=posterior_range[0][north,east]
posterior_range_grid["cellweight"]=EU_raster_read[0][north,east]
#remove all observations that are zero (no agricultural land)
posterior_range_grid=posterior_range_grid[posterior_range_grid["cellweight"]>0]
# %%
selected_cmap="YlOrRd"
max_val=0.05
plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
NUTS0.plot(ax=ax,facecolor="lightgrey")
gpd.GeoDataFrame(posterior_range_grid).plot(ax=ax,column="posterior_range",
            legend=True,
            cmap=selected_cmap,  # YlGn "YlOrRd"
            vmin=0,
            vmax=max_val,)
NUTS0.boundary.plot(ax=ax,edgecolor="darkgrey",linewidth=0.5)
ax.set_xlim(EU_raster.bounds[0],EU_raster.bounds[2])
ax.set_ylim(EU_raster.bounds[1],EU_raster.bounds[3])

plt.axis("off")
Path(output_path).mkdir(parents=True, exist_ok=True)
plt.savefig(output_path+"posterior_width_mean_allcrops_2020.png")
plt.close(fig)
# %%
np.quantile(posterior_range_grid.posterior_range,0.99)
# %%


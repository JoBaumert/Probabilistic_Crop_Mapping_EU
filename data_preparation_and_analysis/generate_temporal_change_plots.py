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
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]

#%%
EU_2010_rio=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/EU/expected_crop_share_entire_EU_2011.tif")
#%%
EU_2010=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/EU/expected_crop_share_entire_EU_2011.tif").read()
EU_2015=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/EU/expected_crop_share_entire_EU_2015.tif").read()
EU_2020=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/EU/expected_crop_share_entire_EU_2020.tif").read()
EU_2018=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/EU/expected_crop_share_entire_EU_2018.tif").read()
# %%

bands=pd.read_csv("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/AT/AT2010simulated_cropshare_10reps_bands.csv")
# %%

all_crops=[]
for i in np.char.split(np.array(bands["name"].iloc[2:30]).astype(str)):
    #crop name starts after 15th character
    all_crops.append(i[0][15:])
all_crops=np.array(all_crops)
# %%

# %%
NUTS=gpd.read_file(data_main_path+"Raw_Data/NUTS/NUTS_RG_01M_2010_3035.shp.zip")

NUTS2=gpd.GeoDataFrame(NUTS[NUTS["LEVL_CODE"]==2])

NUTS2_regs=np.unique(NUTS2["NUTS_ID"])
NUTS2_index=np.arange(len(NUTS2_regs))+1
NUTS2.sort_values(by="NUTS_ID",inplace=True)
NUTS2["country_index"]=NUTS2_index


#%%


geom_value = ((geom,value) for geom, value in zip(NUTS2.geometry, NUTS2.country_index))
NUTS2_raster=features.rasterize(
    geom_value,
    out_shape=EU_2010_rio.shape,
    transform=EU_2010_rio.transform,
)
# %%
NUTS2_raster
# %%


# %%
selected_crops=["SWHE","GRAS","LMAIZ"]
year_diffs=[[2015,2010],[2020,2015]]

all_crops_years_reg_level_diffs=[]
all_crops_years_cell_level_diffs=[]

for crop in selected_crops:
    selected_crop_2010=EU_2010[np.where(all_crops==crop)[0]+1]
    selected_crop_2015=EU_2015[np.where(all_crops==crop)[0]+1]
    selected_crop_2020=EU_2020[np.where(all_crops==crop)[0]+1]

    for year_dif in year_diffs:
        avg_weight=np.concatenate((np.expand_dims(eval(f"EU_{year_dif[0]}")[0],0),np.expand_dims(eval(f"EU_{year_dif[1]}")[0],0)),axis=0).mean(axis=0)
        diff=(eval(f"selected_crop_{year_dif[0]}")-eval(f"selected_crop_{year_dif[1]}"))/1000
        
        
        all_reg_level_diffs=np.ndarray(np.where(NUTS2_raster>0)[0].shape[0])
        all_cell_level_diffs=np.ndarray(np.where(NUTS2_raster>0)[0].shape[0])
        counter=0
        for reg in NUTS2_index:
            reg_level_diff=(diff[0][np.where(NUTS2_raster==reg)]*avg_weight[np.where(NUTS2_raster==reg)]/1000).mean()
            cell_level_diff=diff[0][np.where(NUTS2_raster==reg)]
            size_region=np.where(NUTS2_raster==reg)[0].shape[0]
            all_reg_level_diffs[counter:counter+size_region]=np.repeat(reg_level_diff,len(cell_level_diff))
            all_cell_level_diffs[counter:counter+size_region]=cell_level_diff
            counter+=size_region
        all_crops_years_reg_level_diffs.append(all_reg_level_diffs)
        all_crops_years_cell_level_diffs.append(all_cell_level_diffs)

# %%
plt.scatter(x=all_reg_level_diffs,y=all_cell_level_diffs,s=0.1,color="black")
# %%
fig, ax = plt.subplots(nrows=2, ncols=3,figsize=(10,6))
xmin=min(all_crops_years_reg_level_diffs[0].min(),all_crops_years_reg_level_diffs[1].min())
xmax=max(all_crops_years_reg_level_diffs[0].max(),all_crops_years_reg_level_diffs[1].max())
ymin=min(all_crops_years_cell_level_diffs[0].min(),all_crops_years_cell_level_diffs[1].min())
ymax=max(all_crops_years_cell_level_diffs[0].max(),all_crops_years_cell_level_diffs[1].max())
ax[0,0].scatter(x=all_crops_years_reg_level_diffs[0],y=all_crops_years_cell_level_diffs[0],s=0.1,color="royalblue")
ax[1,0].scatter(x=all_crops_years_reg_level_diffs[1],y=all_crops_years_cell_level_diffs[1],s=0.1,color="royalblue")
ax[0,0].set_xlim(xmin,xmax)
ax[1,0].set_xlim(xmin,xmax)
ax[0,0].set_ylim(ymin,ymax)
ax[1,0].set_ylim(ymin,ymax)
ax[0,0].axhline(c="firebrick",linewidth=0.5)
ax[0,0].axvline(c="firebrick",linewidth=0.5)
ax[1,0].axhline(c="firebrick",linewidth=0.5)
ax[1,0].axvline(c="firebrick",linewidth=0.5)

xmin=min(all_crops_years_reg_level_diffs[2].min(),all_crops_years_reg_level_diffs[3].min())
xmax=max(all_crops_years_reg_level_diffs[2].max(),all_crops_years_reg_level_diffs[3].max())
ymin=min(all_crops_years_cell_level_diffs[2].min(),all_crops_years_cell_level_diffs[3].min())
ymax=max(all_crops_years_cell_level_diffs[2].max(),all_crops_years_cell_level_diffs[3].max())
ax[0,1].scatter(x=all_crops_years_reg_level_diffs[2],y=all_crops_years_cell_level_diffs[2],s=0.1,color="royalblue")
ax[1,1].scatter(x=all_crops_years_reg_level_diffs[3],y=all_crops_years_cell_level_diffs[3],s=0.1,color="royalblue")
ax[0,1].set_xlim(xmin,xmax)
ax[1,1].set_xlim(xmin,xmax)
ax[0,1].set_ylim(ymin,ymax)
ax[1,1].set_ylim(ymin,ymax)
ax[0,1].axhline(c="firebrick",linewidth=0.5)
ax[0,1].axvline(c="firebrick",linewidth=0.5)
ax[1,1].axhline(c="firebrick",linewidth=0.5)
ax[1,1].axvline(c="firebrick",linewidth=0.5)

xmin=min(all_crops_years_reg_level_diffs[4].min(),all_crops_years_reg_level_diffs[5].min())
xmax=max(all_crops_years_reg_level_diffs[4].max(),all_crops_years_reg_level_diffs[5].max())
ymin=min(all_crops_years_cell_level_diffs[4].min(),all_crops_years_cell_level_diffs[5].min())
ymax=max(all_crops_years_cell_level_diffs[4].max(),all_crops_years_cell_level_diffs[5].max())
ax[0,2].scatter(x=all_crops_years_reg_level_diffs[4],y=all_crops_years_cell_level_diffs[4],s=0.1,color="royalblue")
ax[1,2].scatter(x=all_crops_years_reg_level_diffs[5],y=all_crops_years_cell_level_diffs[5],s=0.1,color="royalblue")
ax[0,2].set_xlim(xmin,xmax)
ax[1,2].set_xlim(xmin,xmax)
ax[0,2].set_ylim(ymin,ymax)
ax[1,2].set_ylim(ymin,ymax)
ax[0,2].axhline(c="firebrick",linewidth=0.5)
ax[0,2].axvline(c="firebrick",linewidth=0.5)
ax[1,2].axhline(c="firebrick",linewidth=0.5)
ax[1,2].axvline(c="firebrick",linewidth=0.5)

plt.show()
#%%
xmax
# %%

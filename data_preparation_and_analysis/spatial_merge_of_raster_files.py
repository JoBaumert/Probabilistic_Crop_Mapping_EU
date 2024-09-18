#%%
import rasterio as rio
from rasterio import features
from rasterio.plot import show
from rasterio.merge import merge
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
import arviz as az
#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
postsampling_reps = 10 
Simulated_cropshares_path=(data_main_path+"Results/Simulated_consistent_crop_shares/")
#%%

for year in range(2020,2021):
    
    #year=2020
    print(year)

    country_data=[]
    for i,country in enumerate(np.sort(os.listdir(Simulated_cropshares_path))):
        if (country=="HDIs")|(country=="EU")|(country=="Deviations"):
            continue
        country_data.append(rio.open(Simulated_cropshares_path+country+"/"+country+str(year)+"simulated_cropshare_"+str(postsampling_reps)+"reps_int.tif"))
        
    print("merge raster files...")
    country_data_map,out_trans=merge(country_data)

    country_data_map_expected_shares=country_data_map[:30]
    #delete band with n of fields
    country_data_map_expected_shares=np.delete(country_data_map_expected_shares,1,axis=0)

    print("export rsaterfile expected shares EU...")

    Path(Simulated_cropshares_path+"EU").mkdir(parents=True, exist_ok=True)
    with rio.open(Simulated_cropshares_path+"EU/expected_crop_share_entire_EU_"+str(year)+".tif", 'w',
                width=int(country_data_map_expected_shares.shape[2]),height=int(country_data_map_expected_shares.shape[1]),
                transform=out_trans,count=country_data_map_expected_shares.shape[0],dtype=rio.int16,crs="EPSG:3035") as dst:
        dst.write(country_data_map_expected_shares.astype(rio.int16))

    del country_data_map,country_data_map_expected_shares, country_data
    gc.collect()
#%%
"""
    bands=pd.read_csv(Simulated_cropshares_path+country+"/"+country+str(year)+"simulated_cropshare_"+str(postsampling_reps)+"reps_bands.csv")


#%%
    all_crops=[]
    for i in np.char.split(np.array(bands["name"].iloc[2:30]).astype(str)):
        #crop name starts after 15th character
        all_crops.append(i[0][15:])

    testcrops=["GRAS","LMAIZ","SWHE"]
    for c,crop in enumerate(all_crops):
    #for crop in testcrops:
        if crop in testcrops:
            continue
        print(f"calculating 90% hdi for {crop}")
        relevant_bands=country_data_map[np.where(np.char.find(np.array(bands["name"]).astype(str),crop)==0)[0]]
        hdis=az.hdi(np.expand_dims(relevant_bands,0),hdi_prob=0.9).T
        hdis=hdis.transpose(0,2,1)
    
        with rio.open(Simulated_cropshares_path+"HDIs/"+crop+"_90%HDIs_entire_EU_"+str(year)+".tif", 'w',
                    width=int(hdis.shape[2]),height=int(hdis.shape[1]),
                    transform=out_trans,count=hdis.shape[0],dtype=rio.int16,crs="EPSG:3035") as dst:
            dst.write(hdis.astype(rio.int16))

    del country_data_map
    gc.collect()
#%%
#merge all HDI files to one average HDI-width file
#should take 4- minutes
all_HDIs_list=[]
for file in os.listdir(Simulated_cropshares_path+"HDIs/"):
    all_HDIs_list.append(rio.open(Simulated_cropshares_path+"HDIs/"+file).read())

# %%
transform=rio.open(Simulated_cropshares_path+"HDIs/"+file).transform
#%%
all_HDIs_array=np.array(all_HDIs_list)
# %%
HDI_width=all_HDIs_array.transpose(1,0,2,3)[1]-all_HDIs_array.transpose(1,0,2,3)[0]
# %%
mean_HDI_width=HDI_width.mean(axis=0)
# %%
show(mean_HDI_width)
# %%
plt.hist(mean_HDI_width[np.where(mean_HDI_width>0)].flatten(),bins=50)

# %%
with rio.open(Simulated_cropshares_path+"HDIs/mean_HDI_width_all_crops_all_years.tif", 'w',
            width=int(mean_HDI_width.shape[1]),height=int(mean_HDI_width.shape[0]),
            transform=transform,count=1,dtype=rio.int16,crs="EPSG:3035") as dst:
    dst.write(np.expand_dims(mean_HDI_width,0).astype(rio.int16))
# %%
"""
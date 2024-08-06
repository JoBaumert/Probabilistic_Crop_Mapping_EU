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
import arviz as az
# %%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
postsampling_reps = 10 
results_path=data_main_path+"Results/Simulated_consistent_crop_shares/"

#%%
#Posterior_probability_path=(data_main_path+"Results/Posterior_crop_probability_estimates/")
Posterior_probability_path="/home/baumert/fdiexchange/baumert/posterior_probabilities/"
parameter_path = (
    data_main_path+"delineation_and_parameters/DGPCM_user_parameters.xlsx"
)
raw_data_path = data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
grid_1km_path=raw_data_path+"Grid/"
n_of_fields_path=intermediary_data_path+"Zonal_Stats/"
#%%
Simulated_cropshares_path=(data_main_path+"Results/Simulated_consistent_crop_shares/")
output_path=(data_main_path+"Results/Posterior_crop_probability_estimates/Posterior_range/")
# %%
#import parameters
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])
nuts_info = pd.read_excel(parameter_path, sheet_name="NUTS")
all_years = pd.read_excel(parameter_path, sheet_name="selected_years")
all_years=np.array(all_years["years"])
#%%
EU_raster=rio.open(
    results_path+"EU/expected_crop_share_entire_EU_2010.tif"
)
transform=EU_raster.transform
#%%
for file in os.listdir(grid_1km_path)[:2]:
    print(file[:2])
#%%
new_raster=np.zeros(EU_raster.shape)
#%%
year=2020
for file in os.listdir(grid_1km_path):
    country=file[:2]
    
    if country=="gr":
        continue


  
    print("---"+country+"--"+str(year)+"---")

    posterior_probas=pd.read_parquet(Posterior_probability_path+country+"/"
                                    +country+str(year)+"entire_country")
    print("successfully imported posterior probabilities...")
    #some cells appear more than beta*n_crops (280) times because they are at the border of different nuts region
    #sort DF so that for each cell, crop and beta the one with the largest weight is on top (i.e., the largest cell)
    posterior_probas.sort_values(["CELLCODE","crop","beta","weight"],ascending=[True,True,True,False],inplace=True)
    #now drop all duplicate cells so that only the probabilities for each crop and beta with the largest weight (i.e., the largest) remain
    posterior_probas.drop_duplicates(["CELLCODE","crop","beta"],keep="first",inplace=True)
    #in the resulting df each cell appears 280 times:
    #posterior_probas.CELLCODE.value_counts()


    grouped=posterior_probas[["CELLCODE","crop","posterior_probability"]].groupby(["CELLCODE","crop"])

    minmax_crop_probas=pd.merge(grouped.min().reset_index().rename(columns={"posterior_probability":"min_posterior_probability"}),
                                grouped.max().reset_index().rename(columns={"posterior_probability":"max_posterior_probability"}),how="left",on=["CELLCODE","crop"])

    minmax_crop_probas["difference"]=minmax_crop_probas.max_posterior_probability-minmax_crop_probas.min_posterior_probability

    minmax_crop_probas=minmax_crop_probas[["CELLCODE","difference"]].groupby("CELLCODE").mean().reset_index()

    print("get cell coordinates...")
    corner_coordinates=np.concatenate(minmax_crop_probas.CELLCODE.str.split("E").str[1].str.split("N").values).reshape(-1,2)

    minmax_crop_probas["EOFORIGIN"]=corner_coordinates.T[0].astype(int)
    minmax_crop_probas["NOFORIGIN"]=corner_coordinates.T[1].astype(int)

    minmax_crop_probas=minmax_crop_probas[(minmax_crop_probas["EOFORIGIN"]>EU_raster.bounds[0]/1000)&
            (minmax_crop_probas["EOFORIGIN"]<EU_raster.bounds[2]/1000)&
            (minmax_crop_probas["NOFORIGIN"]>EU_raster.bounds[1]/1000)&
            (minmax_crop_probas["NOFORIGIN"]<EU_raster.bounds[3]/1000)]

    east=((np.array(minmax_crop_probas.EOFORIGIN)-EU_raster.bounds[0]/1000)).astype(int)
    north=(np.abs((np.array(minmax_crop_probas.NOFORIGIN)-EU_raster.bounds[3]/1000))).astype(int)

    new_raster[north,east]=minmax_crop_probas.difference
    
    
    del posterior_probas
    gc.collect()


new_raster_int=(new_raster*1000).astype(int)


Path(output_path).mkdir(parents=True, exist_ok=True)
with rio.open(output_path+"EU_posterior_range_all_crops_"+str(year)+".tif", 'w',
            width=int(new_raster.shape[1]),height=int(new_raster.shape[0]),transform=transform,count=1,dtype=rio.int16,crs="EPSG:3035") as dst:
    dst.write(np.expand_dims(new_raster_int,0).astype(rio.int16))
# %%
test=rio.open(output_path+"EU_posterior_range_all_crops_.tif")
# %%
show(test.read())
# %%
test_read=test.read()
# %%
plt.hist(test_read.flatten(),bins=50)
# %%
show(test_read/1000,cmap="YlOrRd",vmin=0,vmax=0.03)
# %%
test_read.flatten
# %%

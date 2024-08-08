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
"""descriptive statistics for crops in the EU"""
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
results_path=data_main_path+"Results/Simulated_consistent_crop_shares_v1/"
raw_data_path = data_main_path+"Raw_Data/"
output_path=data_main_path+"Results/Validations_and_Visualizations/selected_region_change/"
Simulated_cropshares_path=(data_main_path+"Results/Simulated_consistent_crop_shares/")
cellsize_path=data_main_path+"Intermediary_Data/Zonal_Stats/"
grid_path=raw_data_path+"Grid/"
# %%
#import NUTS_data
NUTS=gpd.read_file(data_main_path+"Raw_Data/NUTS/NUTS_RG_01M_2016_3035.shp.zip")
#%%
"""
EU_cropshares_2010=rio.open(results_path+"EU/expected_crop_share_entire_EU_2010.tif")
EU_cropshares_2010_read=EU_cropshares_2010.read()

# %%

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
    out_shape=EU_cropshares_2010.shape,
    transform=EU_cropshares_2010.transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS1.geometry, NUTS1.country_index))
NUTS1_raster=features.rasterize(
    geom_value,
    out_shape=EU_cropshares_2010.shape,
    transform=EU_cropshares_2010.transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS2.geometry, NUTS2.country_index))
NUTS2_raster=features.rasterize(
    geom_value,
    out_shape=EU_cropshares_2010.shape,
    transform=EU_cropshares_2010.transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS3.geometry, NUTS3.country_index))
NUTS3_raster=features.rasterize(
    geom_value,
    out_shape=EU_cropshares_2010.shape,
    transform=EU_cropshares_2010.transform,
)
# %%
bands=pd.read_csv(results_path+"AT/AT2010simulated_cropshare_10reps_bands.csv")
#get crop names
crops=[]
for band in np.array(bands.name).astype(str)[2:30]:
    crops.append(band[15:])


relevant_start_years=[2010,2012,2014,2016,2018]
for year in relevant_start_years[:1]:
    year0=year
    year1=year+2
    data_year0=rio.open(results_path+"EU/expected_crop_share_entire_EU_"+str(year0)+".tif").read()
    data_year1=rio.open(results_path+"EU/expected_crop_share_entire_EU_"+str(year1)+".tif").read()


# %%
weighted_data_year0=data_year0[1:]*data_year0[0]
#%%
show(weighted_data_year0[1])
"""
# %%
selected_NUTS="RO2"
relevant_cells=pd.read_csv(cellsize_path+selected_NUTS[:2]+"/cell_size/1kmgrid_"+selected_NUTS+"_all_years.csv")
grid=gpd.read_file(grid_path+selected_NUTS[:2]+"_1km.zip!/"+selected_NUTS[:2].lower()+"_1km.shp")
relevant_grid=grid[grid["CELLCODE"].isin(relevant_cells["CELLCODE"])][["CELLCODE","geometry"]]



#%%
crop="SOYA"
relevant_start_years=[2010,2012,2014,2016,2018]
data0=pd.read_csv("/home/baumert/fdiexchange/baumert/DGPCM_expected_crop_shares_EU28_20102020/"+selected_NUTS[:2]+"/expected_crop_shares_"+str(relevant_start_years[0])+".csv")
for year in relevant_start_years[:2]:
    print(year,year+2)
    year0=year
    year1=year+2
    data1=pd.read_csv("/home/baumert/fdiexchange/baumert/DGPCM_expected_crop_shares_EU28_20102020/"+selected_NUTS[:2]+"/expected_crop_shares_"+str(year1)+".csv")

    selected_cells_year0=pd.merge(relevant_grid,
                                    data0[["CELLCODE","crop","weight","expected_cropshare"]],
                                    how="left",on="CELLCODE")
    selected_cells_year1=pd.merge(relevant_grid,
                                 data1[["CELLCODE","crop","weight","expected_cropshare"]],
                                how="left",on="CELLCODE")


    selected_cells_crop_year0=selected_cells_year0[selected_cells_year0["crop"]==crop]
    selected_cells_crop_year1=selected_cells_year1[selected_cells_year1["crop"]==crop]

    regional_crop_share_year1=(np.nansum(np.array(selected_cells_crop_year1.weight)*np.array(selected_cells_crop_year1.expected_cropshare))/
    np.nansum(np.array(selected_cells_year1.weight)*np.array(selected_cells_year1.expected_cropshare)))

    regional_crop_share_year0=(np.nansum(np.array(selected_cells_crop_year0.weight)*np.array(selected_cells_crop_year0.expected_cropshare))/
    np.nansum(np.array(selected_cells_year0.weight)*np.array(selected_cells_year0.expected_cropshare)))

    relative_change_of_crop=(np.nansum(np.array(selected_cells_crop_year1.weight)*np.array(selected_cells_crop_year1.expected_cropshare))/
    np.nansum(np.array(selected_cells_crop_year0.weight)*np.array(selected_cells_crop_year0.expected_cropshare)))-1

    crop_acreage_year0=np.nansum(np.array(selected_cells_crop_year0.weight)*np.array(selected_cells_crop_year0.expected_cropshare))
    crop_acreage_year1=np.nansum(np.array(selected_cells_crop_year1.weight)*np.array(selected_cells_crop_year1.expected_cropshare))

    print("start exporting images...")
    """plot year0"""
    plt.figure(figsize=(12, 12))
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    selected_cmap="Blues"
    max_val=0.1
    NUTS[NUTS["NUTS_ID"]==selected_NUTS].plot(ax=ax,facecolor="lightgrey")
    selected_cells_crop_year0.plot(
        ax=ax,
        legend=True,
        column="expected_cropshare",
        cmap=selected_cmap,
        vmin=0,
        vmax=max_val
        )
    NUTS[NUTS["NUTS_ID"]==selected_NUTS].boundary.plot(ax=ax,edgecolor="darkgrey",linewidth=0.5)
    plt.axis("off")
    plt.title(crop+" "+str(year0)+" "+str(np.round(regional_crop_share_year0,4))+" in ha: "+str(round(crop_acreage_year0)))
    #Path(output_path).mkdir(parents=True, exist_ok=True)
    #plt.savefig(output_path+selected_NUTS+"_"+crop+"_"+str(year0)+"_cropshare.png")
    #plt.close(fig)
    plt.show()

    
    """plot year1"""
    plt.figure(figsize=(12, 12))
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    selected_cmap="Blues"
    max_val=0.1
    NUTS[NUTS["NUTS_ID"]==selected_NUTS].plot(ax=ax,facecolor="lightgrey")
    selected_cells_crop_year1.plot(
        ax=ax,
        legend=True,
        column="expected_cropshare",
        cmap=selected_cmap,
        vmin=0,
        vmax=max_val
        )
    NUTS[NUTS["NUTS_ID"]==selected_NUTS].boundary.plot(ax=ax,edgecolor="darkgrey",linewidth=0.5)
    plt.axis("off")
    plt.title(crop+" "+str(year1)+" "+str(np.round(regional_crop_share_year1,4))+" in ha: "+str(round(crop_acreage_year1)))
    #Path(output_path).mkdir(parents=True, exist_ok=True)
    #plt.savefig(output_path+selected_NUTS+"_"+crop+"_"+str(year1)+"_cropshare.png")
    #plt.close(fig)
    plt.show()

    #change
    change=pd.merge(selected_cells_crop_year0.rename(columns={"expected_cropshare":"year0_exp"})[["CELLCODE","year0_exp","geometry"]],
                    selected_cells_crop_year1.rename(columns={"expected_cropshare":"year1_exp"})[["CELLCODE","year1_exp"]],
                    how="left",on="CELLCODE")
    
    change["change"]=change["year1_exp"]-change["year0_exp"]
   
    """plot change"""

    plt.figure(figsize=(12, 12))
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    selected_cmap="bwr"
    NUTS[NUTS["NUTS_ID"]==selected_NUTS].plot(ax=ax,facecolor="lightgrey")
    change.plot(
        ax=ax,
        column="change",
        legend=True,
        cmap=selected_cmap,
        vmin=-0.03,
        vmax=0.03,

        )
    NUTS[NUTS["NUTS_ID"]==selected_NUTS].boundary.plot(ax=ax,edgecolor="darkgrey",linewidth=0.5)
    plt.axis("off")
    plt.title(crop+" "+str(year0)+"-"+str(year1)+" "+str(np.round(relative_change_of_crop,4)))
    #Path(output_path).mkdir(parents=True, exist_ok=True)
    #plt.savefig(output_path+selected_NUTS+"_"+crop+"_"+str(year0)+"-"+str(year1)+"_change.png")
    #plt.close(fig)
    #
    plt.show()

    #set data0 to data1
    data0=data1.copy()
    # %%
change
# %%
data0
# %%

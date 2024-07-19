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
aggregated_data_all_years=[]
year_list=[[2016,2020],[2016,2020]]
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
 

Simulated_cropshares_path=(data_main_path+"Results/Simulated_consistent_crop_shares/")

for y in range(len(year_list)):
    print(y)
    LAU=gpd.read_file("/home/baumert/fdiexchange/baumert/project2/Local Administrative Units/LAU_RG_01M_"+str(year_list[1][y])+
                      "_3035.shp.zip!/LAU_RG_01M_"+str(year_list[1][y])+"_3035.shp")

    selected_LAU=LAU[LAU["CNTR_CODE"]=="DE"]

    crop_production_data=pd.read_excel(data_main_path+"/Raw_Data/LAU_crop_production_Germany_"+str(year_list[0][y])+".xlsx")

    LAU_codes_crop_data=np.array(crop_production_data["1_Auspraegung_Code"])


    selected_LAU_codes=np.array(selected_LAU["LAU_ID"]).astype(str)


    NUTS3_LAU_codes_available=np.unique(LAU_codes_crop_data[np.where(np.char.str_len(LAU_codes_crop_data.astype(str))>=4)[0]])

    relevant_data=pd.DataFrame()
    for code in NUTS3_LAU_codes_available.astype(str):
        if len(code)==4:
            code="0"+code
        relevant_data=pd.concat((
            relevant_data,
            selected_LAU.iloc[np.where(np.char.find(selected_LAU_codes,str(code))==0)[0]]
        ))

    relevant_data["NUTS3_LAU"]=np.array(relevant_data["LAU_ID"]).astype("U5")

    LAU_codes_crop_data

    add_0_to_beginning=LAU_codes_crop_data[np.where(np.char.str_len(LAU_codes_crop_data.astype(str))==4)[0]]
    add_0_to_beginning=np.char.add(np.zeros_like(add_0_to_beginning).astype(str),add_0_to_beginning.astype(str))
    LAU_codes_crop_data[np.where(np.char.str_len(LAU_codes_crop_data.astype(str))==4)[0]]=add_0_to_beginning
    
    crop_production_data["NUTS3_LAU"]=LAU_codes_crop_data.astype(str)
   
    merged_data=pd.merge(relevant_data,crop_production_data,how="left",on="NUTS3_LAU")

    merged_data.dropna(inplace=True)
    
    merged_data=merged_data.iloc[np.where(np.char.str_len(np.array(merged_data["1_Auspraegung_Code"]).astype(str))<=5)[0]]
   
    

    aggregated_data=merged_data[["NUTS3_LAU","2_Auspraegung_Label","FLC004__Flaeche__ha"]].groupby(["NUTS3_LAU","2_Auspraegung_Label"]).sum().reset_index()
    aggregated_data_all_years.append(aggregated_data)
#%%
aggregated_data_year1=aggregated_data_all_years[0]
aggregated_data_year2=aggregated_data_all_years[1]
# %%
aggregated_data_year1.rename(columns={"FLC004__Flaeche__ha":"ha_year1"},inplace=True)
aggregated_data_year2.rename(columns={"FLC004__Flaeche__ha":"ha_year2"},inplace=True)
#%%
NUTS3_regs=merged_data[["NUTS3_LAU","geometry"]]
NUTS3_regs["NUTS3_index"]=NUTS3_regs["NUTS3_LAU"].astype("category").cat.codes+1
#%%
aggregated_data_both_years=pd.merge(aggregated_data_year1,aggregated_data_year2,how="left",on=["NUTS3_LAU","2_Auspraegung_Label"])

# %%
aggregated_data_both_years.sort_values(by=["NUTS3_LAU","2_Auspraegung_Label"],inplace=True)
# %%
crops=np.unique(aggregated_data_both_years["2_Auspraegung_Label"])
#%%
matrix_year1=np.array(aggregated_data_both_years["ha_year1"]).astype(str).reshape(-1,len(crops))
matrix_year1=np.where(np.char.isnumeric(matrix_year1),matrix_year1,np.nan).astype(float)
matrix_year2=np.array(aggregated_data_both_years["ha_year2"]).astype(str).reshape(-1,len(crops))
matrix_year2=np.where(np.char.isnumeric(matrix_year2),matrix_year2,np.nan).astype(float)
# %%
matrix_year1=np.append(matrix_year1.T,matrix_year1.T[np.where(crops=='Körnermais/Corn-Cob-Mix')[0]]+matrix_year1.T[np.where(crops=="Silomais/Grünmais")[0]],axis=0).T
matrix_year2=np.append(matrix_year2.T,matrix_year2.T[np.where(crops=='Körnermais/Corn-Cob-Mix')[0]]+matrix_year2.T[np.where(crops=="Silomais/Grünmais")[0]],axis=0).T
#%%
crops=np.append(crops,"LMAIZ_sum")
#%%
matrix_year1_cropshare=(matrix_year1.T*(1/matrix_year1.T[0])).T
matrix_year2_cropshare=(matrix_year2.T*(1/matrix_year2.T[0])).T
# %%
relative_change=matrix_year2_cropshare-matrix_year1_cropshare
#%%
average_share=np.nanmean(np.array((matrix_year1_cropshare,matrix_year2_cropshare)),axis=0)
average_share_df=pd.DataFrame(average_share,columns=crops)
average_share_df["NUTS3_LAU"]=np.array(aggregated_data_both_years["NUTS3_LAU"].drop_duplicates())
#%%
relative_change_df=pd.DataFrame(relative_change,columns=crops)
relative_change_df["NUTS3_LAU"]=np.array(aggregated_data_both_years["NUTS3_LAU"].drop_duplicates())
# %%
#DGPCM_crop_shares_year1=rio.open(Simulated_cropshares_path+"/EU/expected_crop_share_entire_EU_"+str(year_list[0][0])+".tif")
DGPCM_crop_shares_year1_read=DGPCM_crop_shares_year1.read()
#%%



#%%
geom_value = ((geom,value) for geom, value in zip(NUTS3_regs.geometry, NUTS3_regs.NUTS3_index))
NUTS3_raster=features.rasterize(
    geom_value,
    out_shape=DGPCM_crop_shares_year1.shape,
    transform=DGPCM_crop_shares_year1.transform,
)
# %%
DGPCM_crop_shares_year1_read=np.where(NUTS3_raster>0,DGPCM_crop_shares_year1_read,np.nan)
weight_year1=DGPCM_crop_shares_year1_read[0]
#delete weight band
DGPCM_crop_shares_year1_read=np.delete(DGPCM_crop_shares_year1_read,0,axis=0)


weighted_DGPCM_crop_shares_year1=weight_year1*DGPCM_crop_shares_year1_read


#%%
regionalized_DGPCM_crop_shares_year1=np.zeros((NUTS3_regs["NUTS3_index"].max(),28))
region_size=np.zeros((NUTS3_regs["NUTS3_index"].max(),1))
for r in range(NUTS3_regs["NUTS3_index"].max()):
    selected_region_info=weighted_DGPCM_crop_shares_year1.transpose(1,2,0)[np.where(NUTS3_raster==r+1)]
    region_size[r]=selected_region_info.shape[0]
    regional_cropshare=selected_region_info.sum(axis=0)
    regional_cropshare=regional_cropshare/regional_cropshare.sum()
    regionalized_DGPCM_crop_shares_year1[r]=regional_cropshare



#%%
#load year2
DGPCM_crop_shares_year2=rio.open(Simulated_cropshares_path+"/EU/expected_crop_share_entire_EU_"+str(year_list[0][1])+".tif")
DGPCM_crop_shares_year2_read=DGPCM_crop_shares_year2.read()
#%%
DGPCM_crop_shares_year2_read=np.where(NUTS3_raster>0,DGPCM_crop_shares_year2_read,np.nan)
weight_year2=DGPCM_crop_shares_year2_read[0]
#delete weight band
DGPCM_crop_shares_year2_read=np.delete(DGPCM_crop_shares_year2_read,0,axis=0)
weighted_DGPCM_crop_shares_year2=weight_year1*DGPCM_crop_shares_year2_read

# %%
regionalized_DGPCM_crop_shares_year2=np.zeros((NUTS3_regs["NUTS3_index"].max(),28))
for r in range(NUTS3_regs["NUTS3_index"].max()):
    regional_cropshare=weighted_DGPCM_crop_shares_year2.transpose(1,2,0)[np.where(NUTS3_raster==r+1)].sum(axis=0)
    regional_cropshare=regional_cropshare/regional_cropshare.sum()
    regionalized_DGPCM_crop_shares_year2[r]=regional_cropshare
# %%
regionalized_DGPCM_crop_shares_change=regionalized_DGPCM_crop_shares_year2-regionalized_DGPCM_crop_shares_year1
# %%
regionalized_DGPCM_crop_shares_change
# %%
#bands are for every year the same
bands=pd.read_csv(Simulated_cropshares_path+"DE/DE2010simulated_cropshare_10reps_bands.csv")
all_crops_DGPCM=[]
for i in np.char.split(np.array(bands["name"].iloc[2:30]).astype(str)):
    #crop name starts after 15th character
    all_crops_DGPCM.append(i[0][15:])
# %%
regionalized_DGPCM_crop_shares_change_df=pd.DataFrame(regionalized_DGPCM_crop_shares_change,columns=all_crops_DGPCM)
# %%
regionalized_DGPCM_crop_shares_change_df["NUTS3_LAU"]=np.array(NUTS3_regs[["NUTS3_LAU","NUTS3_index"]].drop_duplicates().sort_values(by="NUTS3_index")["NUTS3_LAU"])
# %%
regionalized_DGPCM_crop_shares_change_df["region_size"]=region_size
# %%
regionalized_DGPCM_crop_shares_change_df
# %%
#get NUTS data
NUTS=gpd.read_file(data_main_path+"Raw_Data/NUTS/NUTS_RG_01M_2016_3035.shp.zip!/NUTS_RG_01M_2016_3035.shp")
# %%
NUTS1=gpd.GeoDataFrame(NUTS[(NUTS["LEVL_CODE"]==1)&(NUTS["CNTR_CODE"]=="DE")])

NUTS1_regs=np.unique(NUTS1["NUTS_ID"])
NUTS1_index=np.arange(len(NUTS1_regs))+1
NUTS1.sort_values(by="NUTS_ID",inplace=True)
NUTS1["country_index"]=NUTS1_index
# %%
geom_value = ((geom,value) for geom, value in zip(NUTS1.geometry, NUTS1.country_index))
NUTS1_raster=features.rasterize(
    geom_value,
    out_shape=DGPCM_crop_shares_year1.shape,
    transform=DGPCM_crop_shares_year1.transform,
)
# %%

regionalized_DGPCM_crop_shares_change_df
# %%
NUTS1_DGPCM_crop_shares_year1=np.zeros((len(NUTS1_regs),28))
for r in range(len(NUTS1_regs)):
    NUTS1_cropshare=np.nansum(weighted_DGPCM_crop_shares_year1.transpose(1,2,0)[np.where(NUTS1_raster==r+1)],axis=0)
    NUTS1_cropshare=NUTS1_cropshare/NUTS1_cropshare.sum()
    NUTS1_DGPCM_crop_shares_year1[r]=NUTS1_cropshare

NUTS1_DGPCM_crop_shares_year2=np.zeros((len(NUTS1_regs),28))
for r in range(len(NUTS1_regs)):
    NUTS1_cropshare=np.nansum(weighted_DGPCM_crop_shares_year2.transpose(1,2,0)[np.where(NUTS1_raster==r+1)],axis=0)
    NUTS1_cropshare=NUTS1_cropshare/NUTS1_cropshare.sum()
    NUTS1_DGPCM_crop_shares_year2[r]=NUTS1_cropshare
# %%
NUTS1_DGPCM_crop_shares_change=NUTS1_DGPCM_crop_shares_year2-NUTS1_DGPCM_crop_shares_year1
#%%
NUTS1_DGPCM_crop_shares_change_df=pd.DataFrame(NUTS1_DGPCM_crop_shares_change,columns=all_crops_DGPCM)
NUTS1_DGPCM_crop_shares_change_df["NUTS1_Index"]=NUTS1_index
# %%

# %%
NUTS1_NUTS3_translation=NUTS3_regs[["NUTS3_LAU","NUTS3_index"]].drop_duplicates()

# %%
NUTS1_list=[]
for r in np.array([NUTS1_NUTS3_translation["NUTS3_index"]])[0]:
    nuts1,freq=np.unique(NUTS1_raster[np.where(NUTS3_raster==int(r))],return_counts=True)
    try:
        NUTS1_list.append(nuts1[np.argmax(freq)])
    except:
        NUTS1_list.append(np.nan)
    
# %%
NUTS1_NUTS3_translation["NUTS1_Index"]=NUTS1_list
# %%
selected_crop_comparison
#%%
crop_name_DGPCM="OFAR"
crop_name_stats="Pflanzen zur Grünernte"
selected_crop_comparison=pd.merge(regionalized_DGPCM_crop_shares_change_df[["NUTS3_LAU",crop_name_DGPCM,"region_size"]],
         relative_change_df[["NUTS3_LAU",crop_name_stats]],
         how="left",on="NUTS3_LAU")

# %%
plt.scatter(x=selected_crop_comparison[crop_name_DGPCM],y=selected_crop_comparison[crop_name_stats],s=selected_crop_comparison["region_size"]/100)
plt.xlim(-0.14,0.05)
plt.ylim(-0.14,0.05)
plt.hlines(0,-0.14,0.05,color="black")
plt.vlines(0,-0.14,0.05,color="black")
#%%
selected_crop_comparison=pd.merge(selected_crop_comparison,NUTS1_NUTS3_translation[["NUTS3_LAU","NUTS1_Index"]],how="left",on="NUTS3_LAU")


selected_crop_comparison=pd.merge(selected_crop_comparison,NUTS1_DGPCM_crop_shares_change_df[["NUTS1_Index",crop_name_DGPCM]],how="left",on="NUTS1_Index")
# %%
selected_crop_comparison["relative_change_DGPCM"]=selected_crop_comparison[f"{crop_name_DGPCM}_x"]-selected_crop_comparison[f"{crop_name_DGPCM}_y"]
selected_crop_comparison["relative_change_truth"]=selected_crop_comparison[f"{crop_name_stats}"]-selected_crop_comparison[f"{crop_name_DGPCM}_y"]
# %%

#%%
plt.scatter(x=selected_crop_comparison["relative_change_DGPCM"],y=selected_crop_comparison["relative_change_truth"],s=selected_crop_comparison["region_size"]/100)
plt.xlim(-0.14,0.05)
plt.ylim(-0.14,0.05)
plt.hlines(0,-0.14,0.05,color="black")
plt.vlines(0,-0.14,0.05,color="black")

#%%
valid_weights=np.array(selected_crop_comparison["region_size"])[np.where(np.invert(np.isnan(selected_crop_comparison["relative_change_truth"])))[0]].sum()

np.array(selected_crop_comparison["region_size"])[np.where(((selected_crop_comparison["relative_change_DGPCM"]>=0)&(selected_crop_comparison["relative_change_truth"]>=0)) |
         ((selected_crop_comparison["relative_change_DGPCM"]<=0)&(selected_crop_comparison["relative_change_truth"]<=0)))[0]].sum()/valid_weights
# %%
np.where(np.abs(selected_crop_comparison["relative_change_truth"])<0.0001)[0].shape[0]/n_valid_obs
# %%
crops
# %%

# %%
regionalized_DGPCM_crop_shares_year1.shape
# %%
aggregated_data_year1[aggregated_data_year1["2_Auspraegung_Label"]=="Weizen"].sort_values(by="NUTS3_LAU")
# %%
regionalized_DGPCM_crop_shares_year1.T[np.where(np.array(all_crops_DGPCM)=="SWHE")[0]].shape
# %%
matrix_year1_cropshare.T[np.where(np.array(crops)=="Weizen")[0]].shape
# %%
for i in range(16):
    i+=1
    a=selected_crop_comparison[selected_crop_comparison["NUTS1_Index"]==i]

    plt.scatter(a[f"{crop_name_DGPCM}_x"],a[crop_name_stats])
    plt.title(NUTS1_regs[i-1])
    plt.show()
# %%
selected_crop_comparison[[f"{crop_name_DGPCM}_x",f"{crop_name_DGPCM}_y",crop_name_stats,"NUTS1_Index"]].groupby("NUTS1_Index").mean()
# %%
NUTS1_regs
# %%
for i in range(16):
    i+=1
    a=selected_crop_comparison[selected_crop_comparison["NUTS1_Index"]==i]

    plt.scatter(a["relative_change_DGPCM"],a["relative_change_truth"],s=a["region_size"]/100)
    plt.title(NUTS1_regs[i-1])
    try:
        min_value=min(np.nanmin(a["relative_change_DGPCM"]),np.nanmin(a["relative_change_truth"]))
        max_value=max(np.nanmax(a["relative_change_DGPCM"]),np.nanmax(a["relative_change_truth"]))
    except:
        continue
    plt.xlim(min_value,max_value)
    plt.xlim(min_value,max_value)
    plt.hlines(0,min_value,max_value,color="black")
    plt.vlines(0,min_value,max_value,color="black")
    plt.show()

# %%
NUTS1_regs[11]
# %%
crops
# %%
nonnan=np.where(np.invert(np.isnan(DGPCM_crop_shares_year1_read)))
# %%
nonnan[1].min()
# %%
DGPCM_crop_shares_year1_read[np.where(np.array(all_crops_DGPCM)=="SWHE")[0]][0][np.arange(nonnan[1].min(),nonnan[1].max()),np.arange(nonnan[2].min(),nonnan[2].max())]
# %%
np.arange(nonnan[1].min(),nonnan[1].max())
# %%
DGPCM_crop_shares_year1_read[np.where(np.array(all_crops_DGPCM)=="SWHE")[0]].shape
# %%

# %%
DGPCM_crop_shares_year1_read[np.where(np.array(all_crops_DGPCM)=="SWHE")[0]][0][[1,2,3],[3,4,5]]
# %%
#%%
import rasterio as rio
# %%
file=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Raw_Data/CORINE/clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif")
# %%
file_read=file.read()
# %%
show(file_read)
# %%
np.unique(file_read)
# %%
file_read.shape
# %%
file.bounds
# %%
file.transform
# %%

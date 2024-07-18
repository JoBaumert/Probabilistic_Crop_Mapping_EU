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
# %%
country="FR"
year=2018
Posterior_probability_path=(data_main_path+"Results/Posterior_crop_probability_estimates/")
parameter_path = (
    data_main_path+"delineation_and_parameters/DGPCM_user_parameters.xlsx"
)
raw_data_path = data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
grid_1km_path=raw_data_path+"Grid/"
n_of_fields_path=intermediary_data_path+"Zonal_Stats/"
#%%
Simulated_cropshares_path=(data_main_path+"Results/Simulated_consistent_crop_shares/")
#%%
generated_raster=rio.open(Simulated_cropshares_path+"/FR/FR2018simulated_cropshare_"+str(postsampling_reps)+"reps_int.tif").read()
#%%
comparison=pd.read_parquet(Simulated_cropshares_path+"FR/2018/FR2018_FRB")

#%%

grid_1km_path_country = (
# "zip+file://"
    grid_1km_path
+ country
+"_1km.zip"     
)
zip_obj=zipfile.ZipFile(grid_1km_path_country)
for file in zip_obj.namelist():
    if (file[-3:]=="shp")&(file[3:7]=="1km"):
        break

grid_1km_country = gpd.read_file(grid_1km_path_country+"!/"+file)
#%%
comparison=pd.merge(comparison,grid_1km_country,how="left",on="CELLCODE")
#%%
comparison=gpd.GeoDataFrame(comparison)

# %%
emax=comparison["EOFORIGIN"].max()
emin=comparison["EOFORIGIN"].min()
nmax=comparison["NOFORIGIN"].max()
nmin=comparison["NOFORIGIN"].min()
width=(emax-emin)/1000+1
height=(nmax-nmin)/1000+1

ul = (emin, nmax+1000)  # in lon, lat / x, y order
ll = (emin,nmin)
ur = (emax+1000,nmax+1000)
lr = (emax+1000,nmin)


gcps = [
    GCP(0, 0, *ul),
    GCP(0, width, *ur),
    GCP(height, 0, *ll),
    GCP(height, width, *lr)
]

transform_grid_selected_region = from_gcps(gcps)
# %%
grid_1km_country
# %%
selection=comparison.drop_duplicates("CELLCODE")
grid_raster=np.ndarray((2,int(height),int(width)))
geom_value = ((geom,value) for geom, value in zip(selection.geometry, selection.EOFORIGIN))
rasterized=features.rasterize(
    geom_value,
    out_shape=(int(height),int(width)),
    transform=transform_grid_selected_region,
    default_value=1 
)
grid_raster[0]=rasterized

geom_value = ((geom,value) for geom, value in zip(selection.geometry, selection.NOFORIGIN))
rasterized=features.rasterize(
    geom_value,
    out_shape=(int(height),int(width)),
    transform=transform_grid_selected_region,
    default_value=1 
)
grid_raster[1]=rasterized
# %%
rasterized[np.where(rasterized>0)]
# %%
with rio.open(Simulated_cropshares_path+"/"+country+"/"+country+str(2018)+"simulated_cropshare_"+str(postsampling_reps)+"reps_int.tif") as data:
    rst = data.read(window=from_bounds(
        ll[0], ll[1], ur[0], ur[1], 
        data.transform)
    )

# %%
component0=np.repeat("1km",(int(width)*int(height)))
component1=np.repeat("E",(int(width)*int(height)))
component2=grid_raster[0].astype("U4").flatten()
component3=np.repeat("N",(int(width)*int(height)))
component4=grid_raster[1].astype("U4").flatten()
# %%
cellcode_raster=np.char.add(np.char.add(np.char.add(
                    np.char.add(component0,component1),
                    component2),component3),component4)
# %%
cellcode_raster[np.where(grid_raster[0].flatten()==0)]="0"
# %%
relevant_cell_indices=np.where(cellcode_raster!="0")[0]
# %%
#load bands
bands=pd.read_csv(Simulated_cropshares_path+"/"+country+"/"+country+str(2018)+"simulated_cropshare_"+str(postsampling_reps)+"reps_bands.csv")
# %%
crop="SWHE"

selection=rst[np.where(np.char.find(np.array(bands["name"]).astype(str),crop)==0)[0]].reshape(1,100,-1)
selection=selection.T[relevant_cell_indices].T
# %%
hdi_100samples_90perc=az.hdi(selection,hdi_prob=0.9)
# %%
order=np.lexsort((hdi_100samples_90perc.T[1],hdi_100samples_90perc.T[0]))
plt.plot(hdi_100samples_90perc.T[0][order])
plt.plot(hdi_100samples_90perc.T[1][order])
# %%
selection_500maps=comparison[["CELLCODE",crop]]
# %%
selection_500maps["CELLCODE"].value_counts()
# %%
selection_500maps=comparison[["NUTS_ID","CELLCODE","beta",crop]].sort_values(by=["NUTS_ID","CELLCODE"])
# %%
selection_500maps=selection_500maps.groupby("CELLCODE").head(500)

selection_500maps.sort_values(by="CELLCODE",inplace=True)

selection_500maps["counter"]=np.tile(np.arange(500),relevant_cell_indices.shape[0])
# %%
selection_500maps=selection_500maps.pivot(index="CELLCODE",columns="counter",values=crop).reset_index()
# %%
hdi_500samples_90perc=az.hdi(np.array(selection_500maps.iloc[:,1:]).T.reshape(1,500,-1),hdi_prob=0.9)

# %%
sampling_frequency_comparison=pd.DataFrame(selection_500maps["CELLCODE"])
sampling_frequency_comparison["lower_bound_500samples"]=hdi_500samples_90perc.T[0]
sampling_frequency_comparison["upper_bound_500samples"]=hdi_500samples_90perc.T[1]
# %%
sampling_frequency_comparison

# %%
sampling_frequency_comparison=pd.merge(sampling_frequency_comparison,pd.DataFrame({"CELLCODE":cellcode_raster[relevant_cell_indices],
              "lower_bound_100samples":hdi_100samples_90perc.T[0]/1000,
              "upper_bound_100samples":hdi_100samples_90perc.T[1]/1000,}),how="left",on="CELLCODE")
# %%
plt.scatter(x=sampling_frequency_comparison.upper_bound_500samples,y=sampling_frequency_comparison.upper_bound_100samples,s=0.1)
# %%
order=np.argsort(np.array(sampling_frequency_comparison.lower_bound_100samples))
plt.plot(np.array(sampling_frequency_comparison.upper_bound_500samples)[order])
plt.plot(np.array(sampling_frequency_comparison.upper_bound_100samples)[order])
# %%
np.argsort(np.array(sampling_frequency_comparison.lower_bound_500samples))
# %%
size_quantile=100
len_comparison_array=(order.shape[0]//size_quantile)*size_quantile
order=np.argsort(np.array(sampling_frequency_comparison.lower_bound_100samples)[:len_comparison_array])
lower_bound_100samples_quantile_mean=np.array(sampling_frequency_comparison.lower_bound_100samples)[order].reshape((size_quantile,-1)).mean(axis=1)
lower_bound_500samples_quantile_mean=np.array(sampling_frequency_comparison.lower_bound_500samples)[order].reshape((size_quantile,-1)).mean(axis=1)
upper_bound_100samples_quantile_mean=np.array(sampling_frequency_comparison.upper_bound_100samples)[order].reshape((size_quantile,-1)).mean(axis=1)
upper_bound_500samples_quantile_mean=np.array(sampling_frequency_comparison.upper_bound_500samples)[order].reshape((size_quantile,-1)).mean(axis=1)
# %%
plt.plot(lower_bound_500samples_quantile_mean,color="orange")
plt.plot(lower_bound_100samples_quantile_mean,color="red")
plt.plot(upper_bound_500samples_quantile_mean,color="lightblue")
plt.plot(upper_bound_100samples_quantile_mean)

#%%

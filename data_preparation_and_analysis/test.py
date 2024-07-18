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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# %%
aggregated_data=pd.read_csv("/home/baumert/fdiexchange/baumert/project1/Intermediary Data_copied_from_server/Eurostat/optimization_constraints/cropdata_20102020_final.csv")
# %%
aggregated_data_countrylevel=aggregated_data[aggregated_data["NUTS_LEVL"]==0]
# %%
crop="LRAPE"
aggregated_data_countrylevel[aggregated_data_countrylevel["crop"]==crop].pivot(index="country",columns="year",values="area").reset_index()
# %%
country="EL5"
croparea_df=aggregated_data_countrylevel[aggregated_data_countrylevel["country"]==country].pivot(index="year",columns="crop",values="area").reset_index()
croparea=np.array(croparea_df.iloc[:,1:])
# %%
share=(croparea.T*(1/np.nansum(croparea,axis=1)))
# %%
plt.plot(share.T)
# %%
croparea_df.sort_values(by=0,axis=1,ascending=False)
# %%
croplevel_EU=aggregated_data_countrylevel[["year","crop","area"]].groupby(["year","crop"]).sum().reset_index()
# %%
croplevel_EU.pivot(index="year",columns="crop",values="area").reset_index().sort_values(by=0,axis=1,ascending=False)
# %%
croparea_df
# %%
aggregated_data[(aggregated_data["country"]=="EL")&(aggregated_data["crop"]=="OFAR")&(aggregated_data["NUTS_LEVL"]==1)][["year","NUTS_ID","area"]].pivot(
    index="year",columns="NUTS_ID",values="area"
).reset_index()
# %%
#in Germany: DE9 (lower saxony) look at changes for maize and wheat
#in Italy: ITC4 (Lombardia) OFAR
#%%

data10=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/DE/DE2010simulated_cropshare_10reps_int.tif").read()
data12=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/DE/DE2012simulated_cropshare_10reps_int.tif").read()
data14=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/DE/DE2014simulated_cropshare_10reps_int.tif").read()
data16=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/DE/DE2016simulated_cropshare_10reps_int.tif").read()
data18=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/DE/DE2018simulated_cropshare_10reps_int.tif").read()
data20=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/DE/DE2020simulated_cropshare_10reps_int.tif").read()


#%%
bands=pd.read_csv("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/DE/DE2013simulated_cropshare_10reps_bands.csv")
# %%

#%%
LMAIZ10=data10[8]/1000
LMAIZ12=data12[8]/1000
LMAIZ14=data14[8]/1000
LMAIZ16=data16[8]/1000
LMAIZ18=data18[8]/1000
LMAIZ20=data20[8]/1000
SWHE10=data10[25]/1000
SWHE12=data12[25]/1000
SWHE14=data14[25]/1000
SWHE16=data16[25]/1000
SWHE18=data18[25]/1000
SWHE20=data20[25]/1000
# %%
LMAIZdif_1210=LMAIZ12-LMAIZ10
LMAIZdif_1412=LMAIZ14-LMAIZ12
LMAIZdif_1614=LMAIZ16-LMAIZ14
LMAIZdif_1816=LMAIZ18-LMAIZ16
LMAIZdif_2018=LMAIZ20-LMAIZ18
SWHEdif_1210=SWHE12-SWHE10
SWHEdif_1412=SWHE14-SWHE12
SWHEdif_1614=SWHE16-SWHE14
SWHEdif_1816=SWHE18-SWHE16
SWHEdif_2018=SWHE20-SWHE18
# %%

# %%
#plt.scatter(SWHEdif_1210.flatten(),LMAIZdif_1210.flatten(),s=0.01,c="red")
plt.scatter(SWHEdif_1412.flatten(),LMAIZdif_1412.flatten(),s=0.01)
#plt.scatter(SWHEdif_1614.flatten(),LMAIZdif_1614.flatten(),s=0.01)
# %%
plt.contourf(SWHE_dif_1110.flatten(),LMAIZ_dif_1110.flatten())
# %%
plt.scatter(x=SWHE_dif_1110.flatten(),y=LMAIZ_dif_1110.flatten(),s=0.02)
# %%
edges=np.linspace(-0.3,0.3,30)
# %%
a,_,_=np.histogram2d(x=SWHEdif_1412.flatten(),y=LMAIZdif_1412.flatten(),bins=edges)
# %%
a[np.where(a>0)]
# %%
a[0]
#%%
plt.contour(a)
# %%
a[np.where(a>0)].shape
# %%
plt.hist(SWHE_dif_1110.flatten())
# %%
edges[:-1]
# %%
show(a,cmap="viridis",norm="linear")
# %%
import seaborn as sns
# %%
sns.kdeplot(x=SWHE_dif_1110.flatten(),y=LMAIZ_dif_1110.flatten(),levels=2)
# %%
sns.kdeplot(x=np.random.normal(0,1,10),y=np.random.normal(0,1,10))
# %%
LMAIZ_dif_1110.flatten()[:100]
# %%
np.where(SWHEdif_1210.flatten()>0,1,0)
# %%
sns.kdeplot(x=np.where(SWHEdif_1210.flatten()>0,1,0),y=np.where(LMAIZdif_1210.flatten()>0,1,0))
# %%
expected_crop_matrix=data10[2:30]
#%%
expected_crop_matrix*data10[0]
#%%
fig, ax = plt.subplots(1, 5, figsize=(13, 3))
ax[0].scatter(SWHEdif_1210.flatten(),LMAIZdif_1210.flatten(),s=0.01)
ax[0].scatter(SWHEdif_1210.flatten().mean(),LMAIZdif_1210.flatten().mean(),s=40,c="orange")
ax[1].scatter(SWHEdif_1412.flatten(),LMAIZdif_1412.flatten(),s=0.01)
ax[1].scatter(SWHEdif_1412.flatten().mean(),LMAIZdif_1412.flatten().mean(),s=40,c="orange")
ax[2].scatter(SWHEdif_1614.flatten(),LMAIZdif_1614.flatten(),s=0.01)
ax[3].scatter(SWHEdif_1816.flatten(),LMAIZdif_1816.flatten(),s=0.01)
ax[4].scatter(SWHEdif_2018.flatten(),LMAIZdif_2018.flatten(),s=0.01)
ax[0].set_xlim(-0.2,0.15)
ax[0].set_ylim(-0.15,0.15)
ax[0].axhline(0,c="black")
ax[0].axvline(0,c="black")

ax[1].set_xlim(-0.2,0.15)
ax[1].set_ylim(-0.15,0.15)
ax[1].axhline(0,c="black")
ax[1].axvline(0,c="black")
ax[2].set_xlim(-0.2,0.15)
ax[2].set_ylim(-0.15,0.15)
ax[2].axhline(0,c="black")
ax[2].axvline(0,c="black")
ax[3].set_xlim(-0.2,0.15)
ax[3].set_ylim(-0.15,0.15)
ax[3].axhline(0,c="black")
ax[3].axvline(0,c="black")
ax[4].set_xlim(-0.2,0.15)
ax[4].set_ylim(-0.15,0.15)
ax[4].axhline(0,c="black")
ax[4].axvline(0,c="black")
ax[4].set_yticks([])
# %%

#%%
order=np.argsort(SWHEdif_1210.flatten())
# %%
n_of_quantiles=1000
order=np.argsort(SWHEdif_1210.flatten())
order=order[:(order.shape[0]//n_of_quantiles)*n_of_quantiles]
plt.scatter(x=SWHEdif_1210.flatten()[order].reshape(n_of_quantiles,-1).mean(axis=1),
            y=LMAIZdif_1210.flatten()[order].reshape(n_of_quantiles,-1).mean(axis=1))
order=np.argsort(SWHEdif_1412.flatten())
order=order[:(order.shape[0]//n_of_quantiles)*n_of_quantiles]
plt.scatter(x=SWHEdif_1412.flatten()[order].reshape(n_of_quantiles,-1).mean(axis=1),
            y=LMAIZdif_1412.flatten()[order].reshape(n_of_quantiles,-1).mean(axis=1))
order=np.argsort(SWHEdif_1614.flatten())
order=order[:(order.shape[0]//n_of_quantiles)*n_of_quantiles]
plt.scatter(x=SWHEdif_1614.flatten()[order].reshape(n_of_quantiles,-1).mean(axis=1),
            y=LMAIZdif_1614.flatten()[order].reshape(n_of_quantiles,-1).mean(axis=1))
plt.hlines(0,-0.25,0.07)
# %%

# %%
show(SWHEdif_1210)
# %%

np.std([10,1,1,1,1])
# %%
dif=np.array([SWHEdif_1210.flatten(),
          SWHEdif_1412.flatten(),
          SWHEdif_1614.flatten(),
          SWHEdif_1816.flatten(),
          SWHEdif_2018.flatten()])
# %%
np.unique(np.argmax(abs(dif),axis=0),return_counts=True)
# %%
biggest_annual_change=np.argmax(abs(dif),axis=0).reshape(SWHEdif_1210.shape)
show(biggest_annual_change)
# %%
colors=[
    np.array([256/256, 256/256, 256/256, 1]),
    np.array([166/256, 206/256, 227/256, 1]),
    np.array([31/256, 120/256, 180/256, 1]),
    np.array([178/256, 223/256, 138/256, 1]),
    np.array([51/256, 160/256, 44/256, 1]),
    np.array([251/256, 154/256, 153/256, 1]),
    
    #np.array([256/256, 256/256, 256/256, 1])


]
newcmp = ListedColormap(colors)
# %%
biggest_annual_change[np.where(data10[0]==0)]=-1
# %%
show(biggest_annual_change,cmap=newcmp)
# %%
biggest_annual_change.shape
#%%
expected_crop_matrix10=data10[2:30]/1000
expected_crop_matrix12=data12[2:30]/1000
expected_crop_matrix14=data14[2:30]/1000
expected_crop_matrix16=data16[2:30]/1000
expected_crop_matrix18=data18[2:30]/1000
expected_crop_matrix20=data20[2:30]/1000
#%%
dif1=abs(expected_crop_matrix12-expected_crop_matrix10)
dif2=abs(expected_crop_matrix14-expected_crop_matrix12)
dif3=abs(expected_crop_matrix16-expected_crop_matrix14)
dif4=abs(expected_crop_matrix18-expected_crop_matrix16)
dif5=abs(expected_crop_matrix20-expected_crop_matrix18)
#%%
np.max(np.concatenate((dif1,dif2,dif3,dif4,dif5)))
#%%
plt.hist(np.max(np.concatenate((dif1,dif2,dif3,dif4,dif5)),axis=0).flatten(),bins=40)
# %%
SWHEdif_1210[np.where(data10[0]==0)]=np.nan
LMAIZdif_1210[np.where(data10[0]==0)]=np.nan
SWHEdif_1412[np.where(data10[0]==0)]=np.nan
LMAIZdif_1412[np.where(data10[0]==0)]=np.nan
# %%
plt.scatter(x=LMAIZdif_1210,y=SWHEdif_1210,s=0.05,)
# %%
expected_crop_matrix20=data20[2:30]/1000
# %%
expected_crop_matrix20.transpose(1,2,0)[np.where(data10[0]==0)]=np.nan
# %%
shannon_index20=((expected_crop_matrix20*np.log(expected_crop_matrix20+0.00001)).sum(axis=0))*(-1)
# %%
show(np.where(shannon_index20==0,np.nan,shannon_index20))
# %%
shannon_index20
# %%
n_of_crops=np.where(expected_crop_matrix20>0.03,1,0).sum(axis=0)
# %%
n_of_crops_index=np.zeros_like(n_of_crops)
n_of_crops_index[np.where(n_of_crops>0)]=1
n_of_crops_index[np.where(n_of_crops>5)]=2
n_of_crops_index[np.where(n_of_crops>10)]=3
# %%
show(n_of_crops_index)
# %%
np.unique(n_of_crops_index,return_counts=True)
# %%

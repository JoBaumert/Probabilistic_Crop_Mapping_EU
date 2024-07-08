#%%
import rasterio as rio
from rasterio import features
from rasterio.plot import show
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
import geopandas as gpd
import pandas as pd
from pathlib import Path
import os
import numpy as np
import zipfile
import xarray
# %%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
# %%
Posterior_probability_path=(data_main_path+"Results/Posterior_crop_probability_estimates/")
parameter_path = (
    data_main_path+"delineation_and_parameters/DGPCM_user_parameters.xlsx"
)
raw_data_path = data_main_path+"Raw_Data/"
grid_1km_path=raw_data_path+"Grid/"
# %%
#import parameters
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])
nuts_info = pd.read_excel(parameter_path, sheet_name="NUTS")
all_years = np.array(nuts_info["crop_map_year"])
# %%
for country in country_codes_relevant:
    for year in all_years:
        pass
# %%
posterior_probas=pd.read_parquet(Posterior_probability_path+country+"/"
                                 +country+str(year)+"entire_country")

# %%
posterior_probas_test=posterior_probas[posterior_probas["NUTS1"]=="FR1"]
# %%
grid_1km_path_country = (
# "zip+file://"
    grid_1km_path
+ country
+"_1km.zip"     
)
zip_obj=zipfile.ZipFile(grid_1km_path_country)
for file in zip_obj.namelist():
    if file[-3:]=="shp":
        break

grid_1km_country = gpd.read_file(grid_1km_path_country+"!/"+file)
# %%
grid_1km_country
# %%
#some cells appear more than beta*n_crops (280) times because they are at the border of different nuts region
#sort DF so that for each cell, crop and beta the one with the largest weight is on top (i.e., the largest cell)
posterior_probas.sort_values(["CELLCODE","crop","beta","weight"],ascending=[True,True,True,False],inplace=True)
#now drop all duplicate cells so that only the probabilities for each crop and beta with the largest weight (i.e., the largest) remain
posterior_probas.drop_duplicates(["CELLCODE","crop","beta"],keep="first",inplace=True)
#in the resulting df each cell appears 280 times:
#posterior_probas.CELLCODE.value_counts()
#%%
posterior_probas=pd.merge(posterior_probas,grid_1km_country,how="left",on="CELLCODE")

posterior_probas=gpd.GeoDataFrame(posterior_probas)
# %%
posterior_probas[posterior_probas["crop"]=="SWHE"].plot(column="posterior_probability")
# %%
posterior_probas.transform
# %%
selection= posterior_probas[posterior_probas["crop"]=="SWHE"]
# %%

emax=posterior_probas["EOFORIGIN"].max()
emin=posterior_probas["EOFORIGIN"].min()
nmax=posterior_probas["NOFORIGIN"].max()
nmin=posterior_probas["NOFORIGIN"].min()
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

transform = from_gcps(gcps)
# %%
height
#%%
crops=np.unique(posterior_probas["crop"])
betas=np.unique(posterior_probas["beta"])
#%%
posterior_probas.sort_values(by=["NOFORIGIN","EOFORIGIN"],ascending=[False,True],inplace=True)
resulting_matrix=np.ndarray((crops.shape[0]*betas.shape[0]+1,int(height),int(width)))
print("rasterizing cell weight...")
selection=posterior_probas.drop_duplicates(["CELLCODE"])
geom_value = ((geom,value) for geom, value in zip(selection.geometry, selection.weight))
rasterized=features.rasterize(
    geom_value,
    out_shape=(int(height),int(width)),
    transform=transform,
    default_value=1 
)
#first band is the cellweight
resulting_matrix[0]=rasterized

band=1
for crop in crops:
    print("rasterizing "+crop)
    for beta in betas:
        print("beta "+str(beta))
        selection=posterior_probas_test[(posterior_probas_test["crop"]==crop)&(posterior_probas_test["beta"]==beta)]
        result_array=np.zeros(int(height*width),dtype=np.float16)
        result_array[np.where(resulting_matrix[0].flatten()>0)[0]]=np.array(selection.posterior_probability,dtype=np.float16)
        resulting_matrix[band]=result_array.reshape(resulting_matrix[0].shape)
        band+=1
#%%
"""
band=1
for crop in crops[:1]:
    crop="SWHE"
    print("rasterizing "+crop)
    for beta in betas[:1]:
        print("beta "+str(beta))
        selection=posterior_probas[(posterior_probas["crop"]==crop)&(posterior_probas["beta"]==beta)]
        geom_value = ((geom,value) for geom, value in zip(selection.geometry, selection.posterior_probability))
        rasterized=features.rasterize(
            geom_value,
            out_shape=(int(height),int(width)),
            transform=transform,
            default_value=1 
        )
        resulting_matrix[band]=rasterized
        band+=1
"""
# %%
band_indices=pd.DataFrame({"crop":np.repeat(crops,len(betas)),"beta":np.tile(betas,len(crops))})
band_indices=pd.concat((pd.DataFrame({"crop":["weight"],"beta":[np.nan]}),band_indices))
band_indices["band"]=np.arange(len(band_indices))
#%%
band_indices[band_indices["crop"]=="GRAS"]
#%%
show(resulting_matrix[151])
#%%
resulting_matrix_float16=np.array(resulting_matrix,dtype=np.float16)
#%%

# %%
transform
# %%
with rio.open(Posterior_probability_path+country+"/"+country+str(year)+"entire_country_tif.tif", 'w',
              width=int(width),height=int(height),transform=transform,count=281,dtype=rio.float32,crs="EPSG:3035") as dst:
    dst.write(resulting_matrix.astype(rio.float32))
#%%
band_indices.to_csv(Posterior_probability_path+country+"/tif_files_bands.tif")
# %%
test=rio.open("example.tif")
# %%
show(test.read()[0])
# %%
testfile=rio.open(Posterior_probability_path+country+"/"+country+str(year)+"entire_country_tif.tif")
# %%
testfile=testfile.read()
# %%
show(testfile[53])
# %%
testfile.transform
# %%
band_indices
# %%

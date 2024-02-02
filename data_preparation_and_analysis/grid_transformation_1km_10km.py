#%%
from bdb import effective
from gettext import find
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
import os
from pathlib import Path
import zipfile
# %%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]

countries=["FR"] #France is default as for France validation at 10km level is performed

raw_data_path=data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
delineation_and_parameter_path = (
    data_main_path+"delineation_and_parameters/"
)
parameter_path = delineation_and_parameter_path+"DGPCM_user_parameters.xlsx"
grid_path=raw_data_path+"Grid/"
output_path=data_main_path+"Intermediary_Data/Preprocessed_Inputs/Grid//Grid_conversion_1km_10km_"

#%%

for country in countries:

    grid_1km_path_country = (
        grid_path
    + country
    +"_1km.zip"     
    )
    zip=zipfile.ZipFile(grid_1km_path_country)
    for file in zip.namelist():
        if file[-3:]=="shp":
            break
    grid_1km = gpd.read_file(grid_1km_path_country+"!/"+file)

    grid_10km_path_country = (
        grid_path
    + country
    +"_10km.zip"     
    )   
    zip=zipfile.ZipFile(grid_10km_path_country)
    for file in zip.namelist():
        if file[-3:]=="shp":
            break
    grid_10km=gpd.read_file(grid_10km_path_country+"!/"+file)
    
# %%
EOFORIGIN_1km=np.array(grid_1km['EOFORIGIN'])
NOFORIGIN_1km=np.array(grid_1km['NOFORIGIN'])
#%%
grid_1km_list,grid_10km_list=[],[]
for i in range(len(grid_10km)):
    indices=np.where((EOFORIGIN_1km>=grid_10km.iloc[i].EOFORIGIN)&(EOFORIGIN_1km<grid_10km.iloc[i].EOFORIGIN+10000)&
                    (NOFORIGIN_1km>=grid_10km.iloc[i].NOFORIGIN)&(NOFORIGIN_1km<grid_10km.iloc[i].NOFORIGIN+10000))
    grid_1km_list.append(list(grid_1km.loc[indices].CELLCODE))
    grid_10km_list.append(list(np.repeat(grid_10km.iloc[i].CELLCODE,len(indices[0]))))
# %%
grid_1km_array=np.array([cell for cellsample in grid_1km_list for cell in cellsample])
grid_10km_array=np.array([cell for cellsample in grid_10km_list for cell in cellsample])
#%%
grid_conversion_df=pd.DataFrame({'grid_1km':grid_1km_array,'grid_10km':grid_10km_array})
#%%
Path(output_path).mkdir(parents=True, exist_ok=True)
grid_conversion_df.to_csv(output_path+country+".csv")


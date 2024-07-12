#%%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# %%










cellweights=pd.read_csv("/home/baumert/fdiexchange/baumert/project1/Intermediary Data_copied_from_server/Zonal Stats/HR/cell_weight/cell_weights_20102020_new.csv")
# %%
cellweights
# %%
cellweights[cellweights["inferred_UAA"]==0]["nuts3"].value_counts()
# %%
grid=gpd.read_file("/home/baumert/fdiexchange/baumert/project1/Raw Data_copied_from_server/grid/Croatia_shapefile.zip!/hr_1km.shp")
# %%
grid
# %%
cellweights=pd.merge(cellweights,grid[["CELLCODE","geometry"]],how="left",on="CELLCODE")
# %%

cellweights=gpd.GeoDataFrame(cellweights)

# %%
cellweights[(cellweights["year"]==2011)&(cellweights["weight"]>0)].plot()
# %%
cellweights[(cellweights["year"]==2011)&(cellweights["weight"]>0)]["nuts3"].value_counts()
# %%

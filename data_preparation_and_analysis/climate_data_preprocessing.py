#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import modules.functions_for_data_preparation as ffd
from pathlib import Path
import os
# %%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
#%%

temperature_path=data_main_path+"Raw_Data/Temperature/"
precipitation_path=data_main_path+"Raw_Data/Precipitation/"
output_path=data_main_path+"Intermediary_Data/Preprocessed_Inputs/Climate/"

precipitation_files=listdir(precipitation_path)
temperature_files=listdir(temperature_path)
#%%
if __name__ == "__main__":
    print("import data...")
    precipitation_all_dfs=[]
    for file in precipitation_files:
        df_country=pd.read_csv(precipitation_path+file, sep=";")
        df_country["COUNTRY"]=np.repeat(file.split("_")[0],len(df_country))
        precipitation_all_dfs.append(df_country)

    precipitation_all_dfs=pd.concat(precipitation_all_dfs)

    temperature_all_dfs=[]
    for file in temperature_files:
        df_country=pd.read_csv(temperature_path+file, sep=";")
        df_country["COUNTRY"]=np.repeat(file.split("_")[0],len(df_country))
        temperature_all_dfs.append(df_country)

    temperature_all_dfs=pd.concat(temperature_all_dfs)
    
    temperature_all_dfs=temperature_all_dfs[["COUNTRY","GRID_NO","DAY","TEMPERATURE_AVG"]]
    precipitation_all_dfs=precipitation_all_dfs[["COUNTRY","GRID_NO","DAY","PRECIPITATION"]]
    
    print("export data...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    precipitation_all_dfs.to_parquet(output_path+"all_precipitation_data.parquet")
    temperature_all_dfs.to_parquet(output_path+"all_temperature_data.parquet")
    print("task successfully completed")
#%%
"""
#"HOW TO ADD DATA LATER
df_all_files=pd.read_parquet("/home/baumert/serverExchange/Temperature/all_temperature_data.parquet")

uk_data1=pd.read_csv("/home/baumert/serverExchange/Temperature/Temperature_raw_data/united_kingdom_mean_air_temp_19792000.csv",sep=";")
uk_data2=pd.read_csv("/home/baumert/serverExchange/Temperature/Temperature_raw_data/united_kingdom_mean_air_temp_20012022.csv",sep=";")

df_uk=pd.concat((uk_data1,uk_data2))

plt.scatter(x=np.arange(len(df_uk[df_uk["GRID_NO"]==102065])),y=df_uk[df_uk["GRID_NO"]==102065]["TEMPERATURE_AVG"],s=0.1)

df_uk["COUNTRY"]=np.repeat("united_kingdom",len(df_uk))

df_all_files=pd.concat((df_all_files,df_uk[["COUNTRY","GRID_NO","DAY","TEMPERATURE_AVG"]]))

df_all_files.to_parquet("/home/baumert/research/Project-1/data/all_temperature_data.parquet")

df_all_files

df_all_files["COUNTRY"].value_counts()
#%%
"""

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
import seaborn as sns

#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
results_path=data_main_path+"Results/Simulated_consistent_crop_shares/"
raw_data_path = data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
IACS_path=intermediary_data_path+"Preprocessed_Inputs/IACS/true_shares/"
RSCM_path=intermediary_data_path+"Preprocessed_Inputs/RSCM/"
posterior_probability_path=data_main_path+"Results/Posterior_crop_probability_estimates/"
crop_delineation_path=data_main_path+"delineation_and_parameters/DGPCM_crop_delineation.xlsx"
output_path=data_main_path+"Results/Validations_and_Visualizations/Quantile_means/"
# %%
cropname_conversion_file=pd.read_excel(crop_delineation_path)
cropname_conversion=cropname_conversion_file[["DGPCM_code","RSCM","DGPCM_RSCM_common"]].drop_duplicates()


#%%
#import all true shares for France
true_shares_df=pd.DataFrame()
for file in os.listdir(IACS_path):
    if file[12:14]=="FR":
        true_shares=pd.read_csv(IACS_path+file)
        true_shares_df=pd.concat((true_shares_df,true_shares[["CAPRI_code","cropshare_true"]]))
# %%
true_shares_df.dropna(inplace=True)
#%%

posterior_probabilities=pd.read_parquet(posterior_probability_path+"FR/FR2018entire_country")
posterior_probabilities_relevant=posterior_probabilities[posterior_probabilities["beta"]==0]

# %%
selected_crops=["GRAS","SWHE","OFAR","LMAIZ","BARL","LRAPE","SUNF","OCER","DWHE","POTA","OATS","SOYA","ROOF","RYEM","PARI"]
"""
density_all_crops=np.ndarray((len(selected_crops),len(bins)-1))
for c,crop in enumerate(selected_crops[:3]):
    density,_=np.histogram(np.array(posterior_probabilities_relevant[posterior_probabilities_relevant["crop"]==crop].posterior_probability),bins=bins,density=True)
    density_all_crops[c]=density

"""

# %%
step=0.05
bins=np.arange(0,1+step,step)
for crop in selected_crops:
    plt.hist(true_shares_df[true_shares_df["CAPRI_code"]==crop].cropshare_true,density=True,bins=bins)
    plt.hist(posterior_probabilities_relevant[posterior_probabilities_relevant["crop"]==crop].posterior_probability,density=True,bins=bins,alpha=0.6,color="orange")
    plt.title(crop)
    plt.show()
# %%

# %%
for crop in selected_crops:
    true_quantiles=np.quantile(
        true_shares_df[true_shares_df["CAPRI_code"]==crop].cropshare_true,
        q=np.linspace(0,1,101)    
        
    )

    estimated_quantiles=np.quantile(
        posterior_probabilities_relevant[posterior_probabilities_relevant["crop"]==crop].posterior_probability,
       # q=np.linspace(0,1,101)
        q=np.linspace(0,1,11) 
    
    )
    plt.scatter(estimated_quantiles,true_quantiles)
    plt.plot([0,1],[0,1])
    plt.title(crop)
    plt.show()
# %%
np.array(true_shares_df[true_shares_df["CAPRI_code"]==crop].cropshare_true)[np.where((np.array(true_shares_df[true_shares_df["CAPRI_code"]==crop].cropshare_true))>0.9)[0]]
# %%
selected_crops=["GRAS","SWHE","OFAR","LMAIZ","BARL","LRAPE","SUNF","OCER","DWHE"]
selected_crop_names=["Grass","Soft wheat","Forage plants","Maize","Barley","Rapeseed","Sunflower","Other cereals","Durum wheat"]
n_quantiles=100
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(8,6))
fig.tight_layout() 
for c,crop in enumerate(selected_crops):

    true_crop_shares_array=np.array(true_shares_df[true_shares_df["CAPRI_code"]==crop].cropshare_true)
    true_crop_shares_array=np.sort(true_crop_shares_array[:len(true_crop_shares_array)//n_quantiles*n_quantiles])
    true_crop_shares_matrix=true_crop_shares_array.reshape((n_quantiles,len(true_crop_shares_array)//n_quantiles))
    true_quantile_mean=true_crop_shares_matrix.mean(axis=1)

    estimated_crop_shares_array=np.array(posterior_probabilities_relevant[posterior_probabilities_relevant["crop"]==crop].posterior_probability)
    estimated_crop_shares_array=np.sort(estimated_crop_shares_array[:len(estimated_crop_shares_array)//n_quantiles*n_quantiles])
    estimated_crop_shares_matrix=estimated_crop_shares_array.reshape((n_quantiles,len(estimated_crop_shares_array)//n_quantiles))
    estimated_quantile_mean=estimated_crop_shares_matrix.mean(axis=1)

    ax[c//3,c%3].scatter(estimated_quantile_mean,true_quantile_mean,s=15)
    limit=min(max(max(estimated_quantile_mean),max(true_quantile_mean))*1.1,1)
    ax[c//3,c%3].plot([0,limit],[0,limit],color="black")
    ax[c//3,c%3].set_title(selected_crop_names[c])
Path(output_path).mkdir(parents=True, exist_ok=True)
plt.savefig(output_path+"quantile_means_DGPCM_GT.png")
plt.close() 
# %%
RSCM_all_regs=pd.DataFrame()
for file in os.listdir(RSCM_path+"FR/"):
    RSCM_selected=pd.read_csv(RSCM_path+"FR/"+file)

    RSCM_selected.drop(columns="Unnamed: 0",inplace=True)

    common_code_list=[]
    for code in np.array(RSCM_selected.columns):
        if code=="CELLCODE":
            common_code_list.append(code)
            continue
        common_code_list.append(str(cropname_conversion["DGPCM_RSCM_common"].iloc[np.where(np.array(cropname_conversion["RSCM"]).astype(str)==code)[0]].iloc[0]))
    RSCM_selected.columns=common_code_list
    RSCM_all_regs=pd.concat((RSCM_all_regs,RSCM_selected))

#%%
RSCM_all_regs["OCER_new"]=np.array(RSCM_all_regs["OCER"]).sum(axis=1)
RSCM_all_regs.drop("OCER",inplace=True,axis=1)

RSCM_all_regs.rename(columns={"OCER_new":"OCER"},inplace=True)
#%%
selected_crops=["GRAS","SWHE","OFAR","LMAIZ","BARL","LRAPE","SUNF","OCER","DWHE"]
selected_crop_names=["Grass","Soft wheat","Forage plants","Maize","Barley","Rapeseed","Sunflower","Other cereals","Durum wheat"]
n_quantiles=100
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(8,6))
fig.tight_layout() 
for c,crop in enumerate(selected_crops):

    true_crop_shares_array=np.array(true_shares_df[true_shares_df["CAPRI_code"]==crop].cropshare_true)
    true_crop_shares_array=np.sort(true_crop_shares_array[:len(true_crop_shares_array)//n_quantiles*n_quantiles])
    true_crop_shares_matrix=true_crop_shares_array.reshape((n_quantiles,len(true_crop_shares_array)//n_quantiles))
    true_quantile_mean=true_crop_shares_matrix.mean(axis=1)

    estimated_crop_shares_array=np.array(RSCM_all_regs[crop])
    estimated_crop_shares_array=np.sort(estimated_crop_shares_array[:len(estimated_crop_shares_array)//n_quantiles*n_quantiles])
    estimated_crop_shares_matrix=estimated_crop_shares_array.reshape((n_quantiles,len(estimated_crop_shares_array)//n_quantiles))
    estimated_quantile_mean=estimated_crop_shares_matrix.mean(axis=1)

    ax[c//3,c%3].scatter(estimated_quantile_mean,true_quantile_mean,s=15)
    limit=min(max(max(estimated_quantile_mean),max(true_quantile_mean))*1.1,1)
    ax[c//3,c%3].plot([0,limit],[0,limit],color="black")
    ax[c//3,c%3].set_title(selected_crop_names[c])
Path(output_path).mkdir(parents=True, exist_ok=True)
plt.savefig(output_path+"quantile_means_RSCM_GT.png")
plt.close() 
# %%
estimated_crop_shares_array.shape
# %%
estimated_crop_shares_array.reshape((n_quantiles,len(estimated_crop_shares_array)//n_quantiles))
# %%
crop
# %%

# %%

# %%

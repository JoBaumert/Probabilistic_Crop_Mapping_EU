# %%
from statistics import median_high
import arviz as az
import pandas as pd
import numpy as np
from bdb import effective
from gettext import find

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy import stats
import seaborn as sns
from shapely import wkt
from pathlib import Path


#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
beta = 0

country = "FR"
year = 2018



year=2018 # the only year considered by RSCM and for which IACS data is provided
country="FR" #default, as France is the only country for which IACS data in 2018 is available

raw_data_path = data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
parameter_path=data_main_path+"delineation_and_parameters/DGPCM_user_parameters.xlsx"
nuts_path=intermediary_data_path+"Preprocessed_Inputs/NUTS/NUTS_all_regions_all_years"
excluded_NUTS_regions_path = data_main_path+"delineation_and_parameters/excluded_NUTS_regions.xlsx"
crop_delineation_path=data_main_path+"delineation_and_parameters/DGPCM_crop_delineation.xlsx"
IACS_path=intermediary_data_path+"Preprocessed_Inputs/IACS/true_shares/true_shares_"
RSCM_path=intermediary_data_path+"Preprocessed_Inputs/RSCM/"
grid_path=raw_data_path+"Grid/"
grid_conversion_path=intermediary_data_path+"Preprocessed_Inputs/Grid/Grid_conversion_1km_10km_"
posterior_probability_path=data_main_path+"Results/Posterior_crop_probability_estimates/"
#output path
output_path=data_main_path+"Results/Validations_and_Visualizations/Comparison_metrics/"


#%%


""" import parameters"""
nuts_years=pd.read_excel(parameter_path,sheet_name="NUTS")
relevant_nuts_year=nuts_years[nuts_years["crop_map_year"]==year]["nuts_year"].iloc[0]
excluded_NUTS_regions = pd.read_excel(excluded_NUTS_regions_path)
excluded_NUTS_regions = np.array(
    excluded_NUTS_regions["excluded NUTS1 regions"]
)


cropname_conversion_file=pd.read_excel(crop_delineation_path)
cropname_conversion=cropname_conversion_file[["DGPCM_code","RSCM","DGPCM_RSCM_common"]].drop_duplicates()



#%%
"""import data"""

nuts_input = pd.read_csv(nuts_path+".csv")
#import DGPCM estimates
posterior_probabilities=pd.read_parquet(posterior_probability_path+country+"/"+country+str(year)+"entire_country")
posterior_probabilities_relevant=posterior_probabilities[posterior_probabilities["beta"]==beta]
#%%
nuts_regions_relevant = nuts_input[
    (nuts_input["CNTR_CODE"] == country) & (nuts_input["year"] == year)
]

nuts_regions_relevant = nuts_regions_relevant.iloc[
    np.where(
        np.isin(
            np.array(nuts_regions_relevant["NUTS_ID"]).astype("U3"),
            excluded_NUTS_regions,
        ).astype(int)
        == 0
    )[0]
]
#%%
selected_nuts2_regs = np.sort(nuts_regions_relevant[nuts_regions_relevant["LEVL_CODE"]==2]["NUTS_ID"])
selected_nuts1_regs=selected_nuts2_regs.astype("U3")



#%%

#%%

all_pearsonr_df_comparison=pd.DataFrame()

prev_nuts1=""
for nuts2 in selected_nuts2_regs:
    
    if nuts2[:3]!=prev_nuts1:
        eucm=pd.read_csv(RSCM_path+country+"/"+nuts2[:3]+"_1km_reference_grid.csv")
        prev_nuts1=nuts2[:3]

    print(f"calculating region {nuts2}")
    true_shares = pd.read_csv(IACS_path+nuts2+"_"+str(year)+".csv")

    relevant_cells_observed = true_shares["CELLCODE"].value_counts().keys()

    posterior_probabilities_region=posterior_probabilities_relevant[posterior_probabilities_relevant["CELLCODE"].isin(relevant_cells_observed)]
    #for some cells more than one value is available, because it is at the border of several NUTS3 regions...
    #to get one value for each crop per cell, we group the df by crops and cells and take the mean.
    #If a cell only appears once per crop in the df this doesn't change anything, 
    posterior_probabilities_region=posterior_probabilities_region[["CELLCODE","crop","posterior_probability"]].groupby(["CELLCODE","crop"]).mean().reset_index()
    if "CAPRI_code" in true_shares.columns:
        true_shares.rename(columns={"CAPRI_code":"crop"},inplace=True)
    elif "DGPCm_code" in true_shares.columns: 
        true_shares.rename(columns={"DGPCM_code":"crop"},inplace=True)
  
    posterior_probabilities_region_pivot=posterior_probabilities_region.pivot(index="CELLCODE",columns="crop",values="posterior_probability").reset_index()
    true_shares_pivot=true_shares.pivot(index="CELLCODE",columns="crop",values="cropshare_true").reset_index()

    relevant_cells_prediction=np.array(posterior_probabilities_region_pivot["CELLCODE"])
    true_shares_pivot=true_shares_pivot[true_shares_pivot["CELLCODE"].isin(relevant_cells_prediction)]
    RSCM_selected=eucm[eucm["CELLCODE"].isin(relevant_cells_prediction)]

    posterior_probabilities_region_pivot.sort_values(by="CELLCODE",inplace=True)
    true_shares_pivot.sort_values(by="CELLCODE",inplace=True)
    RSCM_selected.sort_values(by="CELLCODE",inplace=True)

    common_code_list=[]
    for code in np.array(posterior_probabilities_region_pivot.columns):
        if code=="CELLCODE":
            common_code_list.append(code)
            continue
        common_code_list.append(str(cropname_conversion["DGPCM_RSCM_common"].iloc[np.where(np.array(cropname_conversion["DGPCM_code"])==code)[0]].iloc[0]))
    posterior_probabilities_region_pivot.columns=common_code_list

    new_dict={}
    I=len(posterior_probabilities_region_pivot)
    for code in np.unique(np.array(posterior_probabilities_region_pivot.columns)):
        if (code=="CELLCODE")|(code=="nan"):
            continue
        new_dict[code]=np.array(posterior_probabilities_region_pivot[code]).reshape(I,-1).sum(axis=1)
    DGPCM_prediction_df=pd.DataFrame(new_dict)

    #normalize shares so that they 
    DGPCM_prediction_df=pd.DataFrame(np.multiply(np.array(DGPCM_prediction_df).T,1/np.array(DGPCM_prediction_df).sum(axis=1)).T,columns=DGPCM_prediction_df.columns)

    DGPCM_prediction_df.insert(loc=0,column="CELLCODE",value=np.array(posterior_probabilities_region_pivot["CELLCODE"]).astype(str))

    RSCM_selected.drop(columns="Unnamed: 0",inplace=True)
    common_code_list=[]
    for code in np.array(RSCM_selected.columns):
        if code=="CELLCODE":
            common_code_list.append(code)
            continue
        common_code_list.append(str(cropname_conversion["DGPCM_RSCM_common"].iloc[np.where(np.array(cropname_conversion["RSCM"]).astype(str)==code)[0]].iloc[0]))
    RSCM_selected.columns=common_code_list

    new_dict={}
    I=len(RSCM_selected)
    for code in np.unique(np.array(RSCM_selected.columns)):
        if (code=="CELLCODE")|(code=="nan"):
            continue
        new_dict[code]=np.array(RSCM_selected[code]).reshape(I,-1).sum(axis=1)
    RSCM_prediction_df=pd.DataFrame(new_dict)
    RSCM_prediction_df.insert(loc=0,column="CELLCODE",value=np.array(RSCM_selected["CELLCODE"]).astype(str))

    common_code_list=[]
    for code in np.array(true_shares_pivot.columns):
        if code=="CELLCODE":
            common_code_list.append(code)
            continue
        common_code_list.append(str(cropname_conversion["DGPCM_RSCM_common"].iloc[np.where(np.array(cropname_conversion["DGPCM_code"])==code)[0]].iloc[0]))
    true_shares_pivot.columns=common_code_list

    new_dict={}
    I=len(true_shares_pivot)
    for code in np.unique(np.array(true_shares_pivot.columns)):
        if (code=="CELLCODE")|(code=="nan"):
            continue
        new_dict[code]=np.array(true_shares_pivot[code]).reshape(I,-1).sum(axis=1)
    true_shares_df=pd.DataFrame(new_dict)

    #normalize shares so that they 
    true_shares_df=pd.DataFrame(np.multiply(np.array(true_shares_df).T,1/np.array(true_shares_df).sum(axis=1)).T,columns=true_shares_df.columns)
    true_shares_df.insert(loc=0,column="CELLCODE",value=np.array(true_shares_pivot["CELLCODE"]).astype(str))
    true_shares_df=pd.merge(true_shares_df,
                            posterior_probabilities_relevant[posterior_probabilities_relevant["lowest_relevant_NUTS_level"]==nuts2][["CELLCODE","weight"]].drop_duplicates("CELLCODE"),
                            how="left",on="CELLCODE")

    pearson_dict={"crop":[],"r_DGPCM_truth":[],"r_RSCM_truth":[],"r_DGPCM_RSCM":[],"wMAE_DGPCM_truth":[],"wMAE_RSCM_truth":[],"wRMSE_DGPCM_truth":[],"wRMSE_RSCM_truth":[]}
    for crop in np.unique(np.array(true_shares_df.columns)):
        if (crop=="CELLCODE")|(crop=="weight"):
            continue
        keep_cells=np.where((~np.isnan(np.array(true_shares_df[crop])))&
                            (~np.isnan(np.array(DGPCM_prediction_df[crop])))&
                            (~np.isnan(np.array(RSCM_prediction_df[crop]))))[0]
        true_shares_crop=np.array(true_shares_df[crop])[keep_cells]
        DGPCM_shares_crop=np.array(DGPCM_prediction_df[crop])[keep_cells]
        RSCM_shares_crop=np.array(RSCM_prediction_df[crop])[keep_cells]
        cell_weight=np.array(true_shares_df["weight"])[keep_cells]
        pearson_dict["crop"].append(crop)
        pearson_dict["r_DGPCM_truth"].append(stats.pearsonr(true_shares_crop,DGPCM_shares_crop)[0])
        pearson_dict["r_RSCM_truth"].append(stats.pearsonr(true_shares_crop,RSCM_shares_crop)[0])
        pearson_dict["r_DGPCM_RSCM"].append(stats.pearsonr(RSCM_shares_crop,DGPCM_shares_crop)[0])
        pearson_dict["wMAE_DGPCM_truth"].append(mean_absolute_error(DGPCM_shares_crop,true_shares_crop,sample_weight=cell_weight))
        pearson_dict["wMAE_RSCM_truth"].append(mean_absolute_error(RSCM_shares_crop,true_shares_crop,sample_weight=cell_weight))
        pearson_dict["wRMSE_DGPCM_truth"].append(mean_squared_error(DGPCM_shares_crop,true_shares_crop,sample_weight=cell_weight,squared=False))
        pearson_dict["wRMSE_RSCM_truth"].append(mean_squared_error(RSCM_shares_crop,true_shares_crop,sample_weight=cell_weight,squared=False))
    
    pearson_df=pd.DataFrame(pearson_dict)
    pearson_df.insert(0,"NUTS_ID",np.repeat(nuts2,len(pearson_df)))
    all_pearsonr_df_comparison=pd.concat((all_pearsonr_df_comparison,pearson_df))


#%%
all_pearsonr_df_comparison_1km=all_pearsonr_df_comparison
Path(output_path).mkdir(parents=True, exist_ok=True)
all_pearsonr_df_comparison_1km.to_csv(output_path+country+str(year)+"_pearsonr_and_wMAE_comparison_DGPCM_RSCM_1km.csv")

#%%

""""""""""""
"""Aggregation at 10x10km"""

all_pearsonr_df_comparison_10km=pd.DataFrame()
prev_nuts1=""
for nuts2 in selected_nuts2_regs:
    
    if nuts2[:3]!=prev_nuts1:
        eucm=pd.read_csv(RSCM_path+country+"/"+nuts2[:3]+"_1km_reference_grid.csv")
        prev_nuts1=nuts2[:3]

    print(f"calculating region {nuts2}")
    true_shares = pd.read_csv(IACS_path+nuts2+"_"+str(year)+".csv")

    relevant_cells_observed = true_shares["CELLCODE"].value_counts().keys()

    posterior_probabilities_region=posterior_probabilities_relevant[posterior_probabilities_relevant["CELLCODE"].isin(relevant_cells_observed)]
    #for some cells more than one value is available, because it is at the border of several NUTS3 regions...
    #to get one value for each crop per cell, we group the df by crops and cells and take the mean.
    #If a cell only appears once per crop in the df this doesn't change anything, 
    posterior_probabilities_region=posterior_probabilities_region[["CELLCODE","crop","posterior_probability","weight"]].groupby(["CELLCODE","crop"]).mean().reset_index()
    true_shares.rename(columns={"CAPRI_code":"crop"},inplace=True)
    weight_df=posterior_probabilities_region[["CELLCODE","weight"]]
    weight_df.drop_duplicates(inplace=True)
    weight_df.sort_values(by="CELLCODE",inplace=True)

    posterior_probabilities_region_pivot=posterior_probabilities_region.pivot(index="CELLCODE",columns="crop",values="posterior_probability").reset_index()
    true_shares_pivot=true_shares.pivot(index="CELLCODE",columns="crop",values="cropshare_true").reset_index()

    relevant_cells_prediction=np.array(posterior_probabilities_region_pivot["CELLCODE"])
    true_shares_pivot=true_shares_pivot[true_shares_pivot["CELLCODE"].isin(relevant_cells_prediction)]
    RSCM_selected=eucm[eucm["CELLCODE"].isin(relevant_cells_prediction)]

    posterior_probabilities_region_pivot.sort_values(by="CELLCODE",inplace=True)
    true_shares_pivot.sort_values(by="CELLCODE",inplace=True)
    RSCM_selected.sort_values(by="CELLCODE",inplace=True)

    common_code_list=[]
    for code in np.array(posterior_probabilities_region_pivot.columns):
        if code=="CELLCODE":
            common_code_list.append(code)
            continue
        common_code_list.append(str(cropname_conversion["DGPCM_RSCM_common"].iloc[np.where(np.array(cropname_conversion["DGPCM_code"])==code)[0]].iloc[0]))
    posterior_probabilities_region_pivot.columns=common_code_list

    new_dict={}
    I=len(posterior_probabilities_region_pivot)
    for code in np.unique(np.array(posterior_probabilities_region_pivot.columns)):
        if (code=="CELLCODE")|(code=="nan"):
            continue
        new_dict[code]=np.array(posterior_probabilities_region_pivot[code]).reshape(I,-1).sum(axis=1)
    DGPCM_prediction_df=pd.DataFrame(new_dict)

    #normalize shares so that they 
    DGPCM_prediction_df=pd.DataFrame(np.multiply(np.array(DGPCM_prediction_df).T,1/np.array(DGPCM_prediction_df).sum(axis=1)).T,columns=DGPCM_prediction_df.columns)

    DGPCM_prediction_df.insert(loc=0,column="CELLCODE",value=np.array(posterior_probabilities_region_pivot["CELLCODE"]).astype(str))

    RSCM_selected.drop(columns="Unnamed: 0",inplace=True)
    common_code_list=[]
    for code in np.array(RSCM_selected.columns):
        if code=="CELLCODE":
            common_code_list.append(code)
            continue
        common_code_list.append(str(cropname_conversion["DGPCM_RSCM_common"].iloc[np.where(np.array(cropname_conversion["RSCM"]).astype(str)==code)[0]].iloc[0]))
    RSCM_selected.columns=common_code_list

    new_dict={}
    I=len(RSCM_selected)
    for code in np.unique(np.array(RSCM_selected.columns)):
        if (code=="CELLCODE")|(code=="nan"):
            continue
        new_dict[code]=np.array(RSCM_selected[code]).reshape(I,-1).sum(axis=1)
    RSCM_prediction_df=pd.DataFrame(new_dict)
    RSCM_prediction_df.insert(loc=0,column="CELLCODE",value=np.array(RSCM_selected["CELLCODE"]).astype(str))

    common_code_list=[]
    for code in np.array(true_shares_pivot.columns):
        if code=="CELLCODE":
            common_code_list.append(code)
            continue
        common_code_list.append(str(cropname_conversion["DGPCM_RSCM_common"].iloc[np.where(np.array(cropname_conversion["DGPCM_code"])==code)[0]].iloc[0]))
    true_shares_pivot.columns=common_code_list

    new_dict={}
    I=len(true_shares_pivot)
    for code in np.unique(np.array(true_shares_pivot.columns)):
        if (code=="CELLCODE")|(code=="nan"):
            continue
        new_dict[code]=np.array(true_shares_pivot[code]).reshape(I,-1).sum(axis=1)
    true_shares_df=pd.DataFrame(new_dict)

    #normalize shares so that they 
    true_shares_df=pd.DataFrame(np.multiply(np.array(true_shares_df).T,1/np.array(true_shares_df).sum(axis=1)).T,columns=true_shares_df.columns)
    true_shares_df.insert(loc=0,column="CELLCODE",value=np.array(true_shares_pivot["CELLCODE"]).astype(str))



    relevant_cells=np.array(true_shares_df["CELLCODE"]).astype(str)
    intermediate_str=np.array([i[1] for i in np.array(np.char.split(relevant_cells,"E"))]).astype(str)
    east=np.array([i[0] for i in np.char.split(intermediate_str,"N")]).astype(int)
    north=np.array([i[1] for i in np.char.split(intermediate_str,"N")]).astype(int)
    cellcode_10km=np.char.add(np.char.add(np.char.add("1kmE",east.astype("U3")),"N"),north.astype("U3"))
    true_shares_df.insert(1,"CELLCODE_10km",cellcode_10km)
    RSCM_prediction_df.insert(1,"CELLCODE_10km",cellcode_10km)
    DGPCM_prediction_df.insert(1,"CELLCODE_10km",cellcode_10km)


    DGPCM_prediction_df_weighted=pd.DataFrame(np.multiply(np.array(DGPCM_prediction_df.iloc[:,2:]).T,np.array(weight_df["weight"])).T,columns=np.array(DGPCM_prediction_df.columns)[2:])
    DGPCM_prediction_df_weighted.insert(0,"CELLCODE_10km",DGPCM_prediction_df["CELLCODE_10km"])
    true_shares_df_weighted=pd.DataFrame(np.multiply(np.array(true_shares_df.iloc[:,2:]).T,np.array(weight_df["weight"])).T,columns=np.array(true_shares_df.columns)[2:])
    true_shares_df_weighted.insert(0,"CELLCODE_10km",true_shares_df["CELLCODE_10km"])
    RSCM_prediction_df_weighted=pd.DataFrame(np.multiply(np.array(RSCM_prediction_df.iloc[:,2:]).T,np.array(weight_df["weight"])).T,columns=np.array(RSCM_prediction_df.columns)[2:])
    RSCM_prediction_df_weighted.insert(0,"CELLCODE_10km",RSCM_prediction_df["CELLCODE_10km"])

    RSCM_prediction_df_10km=RSCM_prediction_df_weighted.groupby("CELLCODE_10km").sum().reset_index()
    DGPCM_prediction_df_10km=DGPCM_prediction_df_weighted.groupby("CELLCODE_10km").sum().reset_index()
    true_shares_df_10km=true_shares_df_weighted.groupby("CELLCODE_10km").sum().reset_index()

    true_shares_df_10km_normalized=pd.DataFrame(
        np.multiply(
            np.array(true_shares_df_10km.iloc[:,1:]).T,1/np.array(true_shares_df_10km.iloc[:,1:]).sum(axis=1)
            ).T,
            columns=np.array(true_shares_df_10km.columns)[1:]
        )
    true_shares_df_10km_normalized.insert(0,"CELLCODE_10km",np.array(true_shares_df_10km["CELLCODE_10km"]).astype(str))

    DGPCM_prediction_df_10km_normalized=pd.DataFrame(
        np.multiply(
            np.array(DGPCM_prediction_df_10km.iloc[:,1:]).T,1/np.array(DGPCM_prediction_df_10km.iloc[:,1:]).sum(axis=1)
            ).T,
            columns=np.array(DGPCM_prediction_df_10km.columns)[1:]
        )
    DGPCM_prediction_df_10km_normalized.insert(0,"CELLCODE_10km",np.array(DGPCM_prediction_df_10km["CELLCODE_10km"]).astype(str))

    RSCM_prediction_df_10km_normalized=pd.DataFrame(
        np.multiply(
            np.array(RSCM_prediction_df_10km.iloc[:,1:]).T,1/np.array(RSCM_prediction_df_10km.iloc[:,1:]).sum(axis=1)
            ).T,
            columns=np.array(RSCM_prediction_df_10km.columns)[1:]
        )
    RSCM_prediction_df_10km_normalized.insert(0,"CELLCODE_10km",np.array(RSCM_prediction_df_10km["CELLCODE_10km"]).astype(str))


    weights=np.array(true_shares_df_10km.iloc[:,1:]).sum(axis=1)
    pearson_dict_10km={"crop":[],"r_DGPCM_truth":[],"r_RSCM_truth":[],"r_DGPCM_RSCM":[],"wMAE_DGPCM_truth":[],"wMAE_RSCM_truth":[],"wRMSE_DGPCM_truth":[],"wRMSE_RSCM_truth":[]}
    for crop in np.unique(np.array(true_shares_df_10km_normalized.columns)):
        if crop=="CELLCODE_10km":
            continue
        keep_cells=np.where((~np.isnan(np.array(true_shares_df_10km_normalized[crop])))&
                            (~np.isnan(np.array(DGPCM_prediction_df_10km_normalized[crop])))&
                            (~np.isnan(np.array(RSCM_prediction_df_10km_normalized[crop]))))[0]
        true_shares_crop_10km_normalized=np.array(true_shares_df_10km_normalized[crop])[keep_cells]
        DGPCM_shares_crop_10km_normalized=np.array(DGPCM_prediction_df_10km_normalized[crop])[keep_cells]
        RSCM_shares_crop_10km_normalized=np.array(RSCM_prediction_df_10km_normalized[crop])[keep_cells]
        pearson_dict_10km["crop"].append(crop)
        pearson_dict_10km["r_DGPCM_truth"].append(stats.pearsonr(true_shares_crop_10km_normalized,DGPCM_shares_crop_10km_normalized)[0])
        pearson_dict_10km["r_RSCM_truth"].append(stats.pearsonr(true_shares_crop_10km_normalized,RSCM_shares_crop_10km_normalized)[0])
        pearson_dict_10km["r_DGPCM_RSCM"].append(stats.pearsonr(RSCM_shares_crop_10km_normalized,DGPCM_shares_crop_10km_normalized)[0])
        pearson_dict_10km["wMAE_DGPCM_truth"].append(mean_absolute_error(true_shares_crop_10km_normalized,DGPCM_shares_crop_10km_normalized,sample_weight=weights[keep_cells]))
        pearson_dict_10km["wMAE_RSCM_truth"].append(mean_absolute_error(true_shares_crop_10km_normalized,RSCM_shares_crop_10km_normalized,sample_weight=weights[keep_cells]))
        pearson_dict_10km["wRMSE_DGPCM_truth"].append(mean_squared_error(true_shares_crop_10km_normalized,DGPCM_shares_crop_10km_normalized,sample_weight=weights[keep_cells],squared=False))
        pearson_dict_10km["wRMSE_RSCM_truth"].append(mean_squared_error(true_shares_crop_10km_normalized,RSCM_shares_crop_10km_normalized,sample_weight=weights[keep_cells],squared=False))
 

    pearson_df_10km=pd.DataFrame(pearson_dict_10km)
    pearson_df_10km.insert(0,"NUTS_ID",np.repeat(nuts2,len(pearson_df_10km)))
    all_pearsonr_df_comparison_10km=pd.concat((all_pearsonr_df_comparison_10km,pearson_df_10km))
#%%
all_pearsonr_df_comparison_10km.to_csv(output_path+country+str(year)+"_pearsonr_and_wMAE_comparison_DGPCM_RSCM_10km.csv")


#%%



selected_crops=["GRAS","SWHE","OFAR","LMAIZ","BARL","LRAPE","SUNF","OCER","DWHE","POTA","OATS","SOYA","ROOF","RYEM","PARI"]

#%%
all_pearsonr_and_wMAE_df_comparison_1km=pd.read_csv(output_path+country+str(year)+"_pearsonr_and_wMAE_comparison_DGPCM_RSCM_1km.csv")
all_pearsonr_and_wMAE_df_comparison_10km=pd.read_csv(output_path+country+str(year)+"_pearsonr_and_wMAE_comparison_DGPCM_RSCM_10km.csv")

all_pearsonr_df_comparison_1km=all_pearsonr_and_wMAE_df_comparison_1km[["crop","r_DGPCM_truth","r_RSCM_truth"]]
all_wMAE_df_comparison_1km=all_pearsonr_and_wMAE_df_comparison_1km[["crop","wMAE_DGPCM_truth","wMAE_RSCM_truth"]]
all_wRMSE_df_comparison_1km=all_pearsonr_and_wMAE_df_comparison_1km[["crop","wRMSE_DGPCM_truth","wRMSE_RSCM_truth"]]

all_pearsonr_df_comparison_10km=all_pearsonr_and_wMAE_df_comparison_10km[["crop","r_DGPCM_truth","r_RSCM_truth"]]
all_wMAE_df_comparison_10km=all_pearsonr_and_wMAE_df_comparison_10km[["crop","wMAE_DGPCM_truth","wMAE_RSCM_truth"]]
all_wRMSE_df_comparison_10km=all_pearsonr_and_wMAE_df_comparison_10km[["crop","wRMSE_DGPCM_truth","wRMSE_RSCM_truth"]]

all_pearsonr_df_melted_1km=pd.melt(all_pearsonr_df_comparison_1km,id_vars=["crop"],value_vars=["r_DGPCM_truth","r_RSCM_truth"],value_name="r")
all_wMAE_df_melted_1km=pd.melt(all_wMAE_df_comparison_1km,id_vars=["crop"],value_vars=["wMAE_DGPCM_truth","wMAE_RSCM_truth"],value_name="wMAE")
all_wRMSE_df_melted_1km=pd.melt(all_wRMSE_df_comparison_1km,id_vars=["crop"],value_vars=["wRMSE_DGPCM_truth","wRMSE_RSCM_truth"],value_name="wRMSE")

all_pearsonr_df_melted_10km=pd.melt(all_pearsonr_df_comparison_10km,id_vars=["crop"],value_vars=["r_DGPCM_truth","r_RSCM_truth"],value_name="r")
all_wMAE_df_melted_10km=pd.melt(all_wMAE_df_comparison_10km,id_vars=["crop"],value_vars=["wMAE_DGPCM_truth","wMAE_RSCM_truth"],value_name="wMAE")
all_wRMSE_df_melted_10km=pd.melt(all_wRMSE_df_comparison_10km,id_vars=["crop"],value_vars=["wRMSE_DGPCM_truth","wRMSE_RSCM_truth"],value_name="wRMSE")
# %%
all_pearsonr_df_melted_1km["crop map"]=np.where(all_pearsonr_df_melted_1km["variable"]=="r_DGPCM_truth","DGP model","remote sensing-based")
all_wMAE_df_melted_1km["crop map"]=np.where(all_wMAE_df_melted_1km["variable"]=="wMAE_DGPCM_truth","DGP model","remote sensing-based")
all_wRMSE_df_melted_1km["crop map"]=np.where(all_wRMSE_df_melted_1km["variable"]=="wRMSE_DGPCM_truth","DGP model","remote sensing-based")

all_pearsonr_df_melted_10km["crop map"]=np.where(all_pearsonr_df_melted_10km["variable"]=="r_DGPCM_truth","DGP model","remote sensing-based")
all_wMAE_df_melted_10km["crop map"]=np.where(all_wMAE_df_melted_10km["variable"]=="wMAE_DGPCM_truth","DGP model","remote sensing-based")
all_wRMSE_df_melted_10km["crop map"]=np.where(all_wRMSE_df_melted_10km["variable"]=="wRMSE_DGPCM_truth","DGP model","remote sensing-based")
# %%
all_pearsonr_df_melted_1km=all_pearsonr_df_melted_1km[all_pearsonr_df_melted_1km["crop"].isin(selected_crops)]
all_wMAE_df_melted_1km=all_wMAE_df_melted_1km[all_wMAE_df_melted_1km["crop"].isin(selected_crops)]
all_wRMSE_df_melted_1km=all_wRMSE_df_melted_1km[all_wRMSE_df_melted_1km["crop"].isin(selected_crops)]

all_pearsonr_df_melted_10km=all_pearsonr_df_melted_10km[all_pearsonr_df_melted_10km["crop"].isin(selected_crops)]
all_wMAE_df_melted_10km=all_wMAE_df_melted_10km[all_wMAE_df_melted_10km["crop"].isin(selected_crops)]
all_wRMSE_df_melted_10km=all_wRMSE_df_melted_10km[all_wRMSE_df_melted_10km["crop"].isin(selected_crops)]



#%%
ax=sns.boxplot(data=all_wMAE_df_melted_1km,x="crop",y="wMAE",hue="crop map")
#plt.hlines(xmin=-0.5,xmax=len(selected_crops)-0.5,y=0,linestyles="dotted",color="black")
plt.ylabel("weighted Mean Absolute Error")
plt.title("weighted Mean Absolute Error - 1km resolution - France 2018")
plt.ylim(0,0.23)
plt.xticks(rotation=90)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5,-0.3), ncol=2, title=None, frameon=False)
plt.savefig(output_path+"wMAE_1km_boxplot.png")
plt.close()
#%%
ax=sns.boxplot(data=all_wMAE_df_melted_10km,x="crop",y="wMAE",hue="crop map")
#plt.hlines(xmin=-0.5,xmax=len(selected_crops)-0.5,y=0,linestyles="dotted",color="black")
plt.ylabel("weighted Mean Absolute Error")
plt.title("weighted Mean Absolute Error - 10km resolution - France 2018")
plt.ylim(0,0.23)
plt.xticks(rotation=90)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5,-0.3), ncol=2, title=None, frameon=False)
plt.savefig(output_path+"wMAE_10km_boxplot.png")
plt.close()
# %%
ax=sns.boxplot(data=all_wRMSE_df_melted_10km,x="crop",y="wRMSE",hue="crop map")
#plt.hlines(xmin=-0.5,xmax=len(selected_crops)-0.5,y=0,linestyles="dotted",color="black")
plt.ylabel("weighted Root Mean Squared Error")
plt.title("weighted Root Mean Squared Error - 10km resolution - France 2018")
plt.ylim(0,0.3)
plt.xticks(rotation=90)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5,-0.3), ncol=2, title=None, frameon=False)
#%%
ax=sns.boxplot(data=all_pearsonr_df_melted_10km,x="crop",y="r",hue="crop map")
plt.hlines(xmin=-0.5,xmax=len(selected_crops)-0.5,y=0,linestyles="dotted",color="black")
plt.ylabel("Pearson correlation coefficient")
plt.title("Pearson r - 10km resolution - France 2018")
plt.ylim(-0.35,1)
plt.xticks(rotation=90)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5,-0.3), ncol=2, title=None, frameon=False)

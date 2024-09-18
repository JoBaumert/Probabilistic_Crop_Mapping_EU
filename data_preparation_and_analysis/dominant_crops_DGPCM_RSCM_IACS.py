#%%
from bdb import effective
import argparse
# from gettext import find
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib.colors import ListedColormap
from pathlib import Path
import os, zipfile
# %%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
beta = 0

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
posterior_probability_path=data_main_path+"Results/Posterior_crop_probability_estimates/"
#output path
output_path=data_main_path+"Results/Validations_and_Visualizations/Comparison_dominant_crops/"
#%%
# import parameters
selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])
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
""" import nuts data"""
nuts_input = pd.read_csv(nuts_path+".csv")

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
excluded_NUTS_regions_info = pd.read_excel(excluded_NUTS_regions_path)


NUTS_boundaries = gpd.read_file(nuts_path+".shp")
if country in excluded_NUTS_regions_info["country"].value_counts().keys():
    country_boundaries = NUTS_boundaries[
        (NUTS_boundaries["CNTR_CODE"] == country)
        & (NUTS_boundaries["LEVL_CODE"] == 0)
    ]
    country_boundaries = gpd.GeoDataFrame(country_boundaries)
    relevant_nuts1_boundaries = NUTS_boundaries[
        (NUTS_boundaries["CNTR_CODE"] == country)
        & (NUTS_boundaries["LEVL_CODE"] == 1)
        & (
            ~NUTS_boundaries["NUTS_ID"].isin(
                excluded_NUTS_regions_info["excluded NUTS1 regions"]
            )
        )
    ]
    relevant_nuts1_boundaries = gpd.GeoDataFrame(relevant_nuts1_boundaries)
    relevant_boundaries_country = relevant_nuts1_boundaries.geometry.unary_union
    relevant_boundaries_country = gpd.GeoDataFrame(
        geometry=[relevant_boundaries_country], crs="epsg:3035"
    )
    relevant_boundaries_country.insert(0, "country", country)
else:
    relevant_boundaries_country = NUTS_boundaries[
        (NUTS_boundaries["CNTR_CODE"] == country)
        & (NUTS_boundaries["LEVL_CODE"] == 0)
    ]
    relevant_boundaries_country = relevant_boundaries_country[
        ["CNTR_CODE", "geometry"]
    ]
#%%
""" import estimated crop probabilities"""

grid_1km_path_country = (
            # "zip+file://"
                grid_path
                + country
                +"_1km.zip"     
                )

zip=zipfile.ZipFile(grid_1km_path_country)
for file in zip.namelist():
    if file[-3:]=="shp":
        break


grid_1km_country = gpd.read_file(grid_1km_path_country+"!/"+file)

posterior_probabilities_country = pd.read_parquet(
    posterior_probability_path
    + country
    + "/"
    + country
    + str(year)
    + "entire_country"
)
posterior_probabilities_country = posterior_probabilities_country[
    posterior_probabilities_country["beta"] == beta
]

#posterior_probabilities_country=posterior_probabilities_country[posterior_probabilities_country["crop"]==selected_crop]


#%%
posterior_probabilities_country.insert(
    0, "country", np.repeat(country, len(posterior_probabilities_country))
)
#%%
posterior_probabilities_country_gdf=gpd.GeoDataFrame(posterior_probabilities_country)

#%%
""" import IACS data"""
selected_nuts2_regs = np.sort(nuts_regions_relevant[nuts_regions_relevant["LEVL_CODE"]==2]["NUTS_ID"])
selected_nuts1_regs=selected_nuts2_regs.astype("U3")
#%%


all_true_shares=pd.DataFrame()

for nuts2 in selected_nuts2_regs:
    print(f"concatenating region {nuts2}")
    true_shares = pd.read_csv(IACS_path+nuts2+"_"+str(year)+".csv")
    true_shares.rename(columns={"CAPRI_code":"crop"},inplace=True)
    true_shares=true_shares[["CELLCODE","crop","cropshare_true"]]
    all_true_shares=pd.concat((all_true_shares,true_shares))

#some cells appear morethan once (they are in more than one nuts2 region). Take the mean crop share for those cells as the value
all_true_shares=all_true_shares.groupby(["CELLCODE","crop"]).mean().reset_index()
#%%
"""import RSCM data"""
all_rscm_data=pd.DataFrame()
for reg in selected_nuts1_regs:
    rscm_reg=pd.read_csv(RSCM_path+country+"/"+reg+"_1km_reference_grid.csv")
    all_rscm_data=pd.concat((all_rscm_data,rscm_reg))

#%%
rscm_code_array=np.array(all_rscm_data.columns[2:])
all_rscm_data=all_rscm_data.melt("CELLCODE",rscm_code_array,var_name="eucm_cropcode")

DGPCM_code_array=np.ndarray(rscm_code_array.shape).astype(str)
for i,rscm_cropcode in enumerate(rscm_code_array):
    DGPCM_code_array[i]=np.array(cropname_conversion["DGPCM_RSCM_common"])[np.where(cropname_conversion["RSCM"]==float(rscm_cropcode))[0]][0]

crops_in_DGPCM_code=np.ndarray(len(all_rscm_data)).astype(str)
for i,code in enumerate(rscm_code_array.astype(float).astype(int)):
    crops_in_DGPCM_code[np.where(np.array(all_rscm_data["eucm_cropcode"]).astype(float).astype(int)==code)[0]]=DGPCM_code_array[i]

all_rscm_data["crop"]=crops_in_DGPCM_code


all_rscm_data.rename(columns={"value":"rscm_estimated_cropshare"},inplace=True)
all_rscm_data=all_rscm_data[["CELLCODE","crop","rscm_estimated_cropshare"]].sort_values(by=["CELLCODE","crop"])
#%%

#%%
"""merge DGPCM predictions and true shares"""
estimated_and_true_shares_gdf=pd.merge(posterior_probabilities_country_gdf,all_true_shares,how="left",on=["CELLCODE","crop"])
estimated_and_true_shares_gdf.dropna(inplace=True)
#%%

#%%
estimated_crop_shares_gdf=estimated_and_true_shares_gdf[["CELLCODE","crop","posterior_probability"]]
true_crop_shares_gdf=estimated_and_true_shares_gdf[["CELLCODE","crop","cropshare_true"]]
# %%
"""merge s1cm predictions and true shares to receive a gdf"""
all_rscm_data_gdf=pd.merge(true_crop_shares_gdf[["CELLCODE","crop",]],all_rscm_data,how="left",on=["CELLCODE","crop"])
#%%
all_rscm_data_gdf.dropna(inplace=True)
#%%
all_rscm_data_gdf=pd.merge(all_rscm_data_gdf,grid_1km_country[["CELLCODE","geometry"]],how="left",on="CELLCODE")
estimated_crop_shares_gdf=pd.merge(estimated_crop_shares_gdf,grid_1km_country[["CELLCODE","geometry"]],how="left",on="CELLCODE")
true_crop_shares_gdf=pd.merge(true_crop_shares_gdf,grid_1km_country[["CELLCODE","geometry"]],how="left",on="CELLCODE")
#%%
all_rscm_data_gdf=gpd.GeoDataFrame(all_rscm_data_gdf)
estimated_crop_shares_gdf=gpd.GeoDataFrame(estimated_crop_shares_gdf)
true_crop_shares_gdf=gpd.GeoDataFrame(true_crop_shares_gdf)
#%%

#%%
"""
selected_crop="GRAS"
if selected_crop=="GRAS":
    selected_cmap="YlGn"
    max_val=1
elif selected_crop=="SWHE":
    selected_cmap="YlOrRd"
    max_val=0.6
elif selected_crop=="LMAIZ":
    selected_cmap="Blues"
    max_val=0.6

df_selected=all_rscm_data_gdf[all_rscm_data_gdf["crop"]==selected_crop]

plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gpd.GeoDataFrame(relevant_boundaries_country).plot(ax=ax, facecolor="lightgrey")
df_selected.plot(
    ax=ax,
    column="rscm_estimated_cropshare",
    legend=True,
    cmap=selected_cmap,
    linewidth=0.15,
)
relevant_boundaries_country.plot(ax=ax, facecolor="None", edgecolor="black", linewidth=0.1)

plt.title(f"RSCM grass share ({year})")
plt.axis("off")
"""

#%%
"""DOMINANT CROPS"""

true_dominant_crops=true_crop_shares_gdf.sort_values(by=["CELLCODE","cropshare_true"],ascending=False).drop_duplicates("CELLCODE")
estimated_dominant_crops=estimated_crop_shares_gdf.sort_values(by=["CELLCODE","posterior_probability"],ascending=False).drop_duplicates("CELLCODE")
s1cm_dominant_crops=all_rscm_data_gdf.sort_values(by=["CELLCODE","rscm_estimated_cropshare"],ascending=False).drop_duplicates("CELLCODE")
# %%
true_dominant_crops["dominant_crop_id"] = pd.Categorical(
            true_dominant_crops["crop"]
        ).codes
true_dominant_crops.sort_values(by="dominant_crop_id", inplace=True)

true_dominant_crop_array = np.array(
    true_dominant_crops.drop_duplicates(["crop"])["crop"]
)
true_dominant_crop_id_array = np.array(
    true_dominant_crops.drop_duplicates(["crop"])["dominant_crop_id"]
)

estimated_dominant_crops["dominant_crop_id"] = pd.Categorical(
            estimated_dominant_crops["crop"]
        ).codes
estimated_dominant_crops.sort_values(by="dominant_crop_id", inplace=True)

estimated_dominant_crop_array = np.array(
    estimated_dominant_crops.drop_duplicates(["crop"])["crop"]
)
estimated_dominant_crop_id_array = np.array(
    estimated_dominant_crops.drop_duplicates(["crop"])["dominant_crop_id"]
)

s1cm_dominant_crops["dominant_crop_id"] = pd.Categorical(
            s1cm_dominant_crops["crop"]
        ).codes
s1cm_dominant_crops.sort_values(by="dominant_crop_id", inplace=True)

s1cm_dominant_crop_array = np.array(
    s1cm_dominant_crops.drop_duplicates(["crop"])["crop"]
)
s1cm_dominant_crop_id_array = np.array(
    s1cm_dominant_crops.drop_duplicates(["crop"])["dominant_crop_id"]
)


#%%
"""for interpretation of the output figures, use the color codes 
to understand which color symbolizes which crop (suggestion: copy image to powerpoint which allows you to get the RGB color code)
"""
crop_colors={
            "APPL+OFRU":[193, 18, 196],# horticulture(appl+ofru)
            "BARL":[209, 162, 42],  # barley
            "CITR":[218, 245, 66],  # citr
            "DWHE":[245, 164, 66],  # wheat (durum and soft)
            "DWHE":[23, 23, 23],
            "FLOW":[193, 18, 196],  # horticulture (flowers)
            "GRAS":[30, 133, 83],  # gras and ofar (grass)
            "LMAIZ":[235, 89, 16],  # mais
            "LRAPE":[245, 217, 2],  # rape
            "NURS":[193, 18, 196],# horticulture(nurs)
            "OATS":[199, 173, 103],  # cereals (oats)
            "OCER":[199, 173, 103],  # cereals (ocer)
            "OCRO":[156, 156, 161],  # other permanent and ind crops
            "OFAR":[30, 133, 83],  # gras and ofar (ofar)
            "OIND":[156, 156, 161],  # other permanent and ind crop
            "OLIVGR":[2, 36, 9],  # olives
            "PARI":[93, 185, 227],  # rice
            "POTA":[147, 156, 34],  # rootcrops (pota),
            "PULS":[136, 179, 147],  # pulses
            "ROOF":[147, 156, 34],  # rootcrops (roof),
            "RYEM":[163, 126, 29],  # ryem
            "SOYA":[89, 76, 207],  # soya
            "SUGB":[147, 156, 34],  # rootcrops (sugb)
            "SUNF":[247, 223, 129],  # sunfl
            "SWHE":[245, 164, 66],  # wheat (durum and soft)
            "TEXT":[194, 8, 45],  # text
            "TOBA":[193, 18, 196],  # horticulture(tobacco)
            "TOMA+OVEG":[193, 18, 196],  # horticulture(toma+oveg)
            "VINY":[128, 4, 95]  # viny
        }
        


#%%
custom_cmap_true_dominant_crops=np.ndarray((len(true_dominant_crop_array),3))
for c,crop in enumerate(true_dominant_crop_array):
    custom_cmap_true_dominant_crops[c]=crop_colors[crop]
custom_cmap_true_dominant_crops = custom_cmap_true_dominant_crops / 256
custom_cmap_true_dominant_crops = np.insert(custom_cmap_true_dominant_crops, 3, np.ones(len(custom_cmap_true_dominant_crops)), axis=1)
custom_cmap_true_dominant_crops = ListedColormap(custom_cmap_true_dominant_crops)

custom_cmap_estimated_dominant_crops=np.ndarray((len(estimated_dominant_crop_array),3))
for c,crop in enumerate(estimated_dominant_crop_array):
    custom_cmap_estimated_dominant_crops[c]=crop_colors[crop]
custom_cmap_estimated_dominant_crops = custom_cmap_estimated_dominant_crops / 256
custom_cmap_estimated_dominant_crops = np.insert(custom_cmap_estimated_dominant_crops, 3, np.ones(len(custom_cmap_estimated_dominant_crops)), axis=1)
custom_cmap_estimated_dominant_crops = ListedColormap(custom_cmap_estimated_dominant_crops)

custom_cmap_s1cm_dominant_crops=np.ndarray((len(s1cm_dominant_crop_array),3))
for c,crop in enumerate(s1cm_dominant_crop_array):
    custom_cmap_s1cm_dominant_crops[c]=crop_colors[crop]
custom_cmap_s1cm_dominant_crops = custom_cmap_s1cm_dominant_crops / 256
custom_cmap_s1cm_dominant_crops = np.insert(custom_cmap_s1cm_dominant_crops, 3, np.ones(len(custom_cmap_s1cm_dominant_crops)), axis=1)
custom_cmap_s1cm_dominant_crops = ListedColormap(custom_cmap_s1cm_dominant_crops)
# %%

# %%

plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gpd.GeoDataFrame(relevant_boundaries_country).plot(ax=ax, facecolor="lightgrey")
estimated_dominant_crops.plot(
    ax=ax,
    column="dominant_crop_id",
    legend=True,
    cmap=custom_cmap_estimated_dominant_crops,
    linewidth=0.15,
)
relevant_boundaries_country.plot(ax=ax, facecolor="None", edgecolor="black", linewidth=0.1)

plt.title(f"Estimated dominant crops ({year})")
plt.axis("off")
Path(output_path).mkdir(parents=True, exist_ok=True)
plt.savefig(output_path+"_DGPCM_dominant_crops_"+country+"_"+str(year)+".png")
plt.close(fig)
#%%
plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gpd.GeoDataFrame(relevant_boundaries_country).plot(ax=ax, facecolor="lightgrey")
true_dominant_crops.plot(
    ax=ax,
    column="dominant_crop_id",
    legend=True,
    cmap=custom_cmap_true_dominant_crops,
    linewidth=0.15,
)
relevant_boundaries_country.plot(ax=ax, facecolor="None", edgecolor="black", linewidth=0.1)

plt.title(f"True dominant crops ({year})")
plt.axis("off")
Path(output_path).mkdir(parents=True, exist_ok=True)
plt.savefig(output_path+"_IACS_dominant_crops_"+country+"_"+str(year)+".png")
plt.close(fig)

# %%


plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gpd.GeoDataFrame(relevant_boundaries_country).plot(ax=ax, facecolor="lightgrey")
s1cm_dominant_crops.plot(
    ax=ax,
    column="dominant_crop_id",
    legend=True,
    cmap=custom_cmap_s1cm_dominant_crops,
    linewidth=0.15,
)
relevant_boundaries_country.plot(ax=ax, facecolor="None", edgecolor="black", linewidth=0.1)

plt.title(f"RSCM dominant crops ({year})")
plt.axis("off")

Path(output_path).mkdir(parents=True, exist_ok=True)
plt.savefig(output_path+"_RSCM_dominant_crops_"+country+"_"+str(year)+".png")
plt.close(fig)
#%%

# %%

# %%
from statistics import median_high
import arviz as az
import pandas as pd
import numpy as np#
from bdb import effective
from gettext import find
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os, zipfile
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from pathlib import Path
import rasterio as rio
from rasterio import features
from rasterio.plot import show
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
from rasterio.windows import from_bounds

"""
this script was used to generate figure 6 of the paper

we assume that 90% (95%) of the observations lie withing the 90% (95%) CI, if the CI is correctly estimated, etc. 

"""
#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
beta = 0
c=0.9 #HDI interval width

year=2018 # the only year considered by RSCM and for which IACS data is provided
country="FR" #default, as France is the only country for which IACS data in 2018 is available

raw_data_path = data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
parameter_path=data_main_path+"delineation_and_parameters/DGPCM_user_parameters.xlsx"
nuts_path=intermediary_data_path+"Preprocessed_Inputs/NUTS/NUTS_all_regions_all_years"
excluded_NUTS_regions_path = data_main_path+"delineation_and_parameters/excluded_NUTS_regions.xlsx"
crop_delineation_path=data_main_path+"delineation_and_parameters/DGPCM_crop_delineation.xlsx"
DGPCM_simulated_consistent_shares_path=data_main_path+"Results/Simulated_consistent_crop_shares/"
IACS_path=intermediary_data_path+"Preprocessed_Inputs/IACS/"
grid_path=raw_data_path+"Grid/"
grid_conversion_path=intermediary_data_path+"Preprocessed_Inputs/Grid/Grid_conversion_1km_10km_"
posterior_probability_path=data_main_path+"Results/Posterior_crop_probability_estimates/"
#output path
output_path=data_main_path+"Results/Credible_Intervals/"
output_CI_visualization_path=data_main_path+"Results/Validations_and_Visualizations/HDI_width/"
# %%

""" import parameters"""
nuts_years=pd.read_excel(parameter_path,sheet_name="NUTS")
relevant_nuts_year=nuts_years[nuts_years["crop_map_year"]==year]["nuts_year"].iloc[0]
excluded_NUTS_regions = pd.read_excel(excluded_NUTS_regions_path)
excluded_NUTS_regions = np.array(
    excluded_NUTS_regions["excluded NUTS1 regions"]
)


cropname_conversion_file=pd.read_excel(crop_delineation_path)
cropname_conversion=cropname_conversion_file[["DGPCM_code","RSCM","DGPCM_RSCM_common"]].drop_duplicates()

# %%
"""import data"""


nuts_input = pd.read_csv(nuts_path+".csv")

# %%
try:
    nuts_regions_relevant = nuts_input[
        (nuts_input["CNTR_CODE"] == country) & (nuts_input["year"] == year)
    ]
except:
    nuts_regions_relevant = nuts_input[
        (nuts_input["CNTR_CODE"] == country)
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


selected_nuts2_regs = np.sort(nuts_regions_relevant[nuts_regions_relevant["LEVL_CODE"]==2]["NUTS_ID"])
selected_nuts2_regs=np.unique(selected_nuts2_regs)
selected_nuts1_regs=selected_nuts2_regs.astype("U3")




"""ANALYZE CI"""
grid_1km_path_country = (
            # "zip+file://"
                grid_path
                + country
                +"_1km.zip"     
                )

zip=zipfile.ZipFile(grid_1km_path_country)
for file in zip.namelist():
    if (file[-3:]=="shp")&(file[3:6]=="1km"):
        break


grid_1km_country = gpd.read_file(grid_1km_path_country+"!/"+file)
#%%



#%%

selected_crops={
    "GRAS":"Grass",
    "SWHE":"Soft Wheat",
    "LMAIZ": "Maize",
    "BARL":"Barley",
    "LRAPE":"Rapeseed",
    "OFAR":"Other Forage Plants",
    "SUNF":"Sunflowers",
    "VINY":"Vinyeards",
    "OCER":"Other Cereals"
}
#%%
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

posterior_probabilities_country = pd.merge(
    posterior_probabilities_country[["CELLCODE", "crop", "posterior_probability","weight"]],
    grid_1km_country[["CELLCODE", "geometry"]],
    how="left",
    on="CELLCODE",
)
posterior_probabilities_country.insert(
    0, "country", np.repeat(country, len(posterior_probabilities_country))
)
posterior_probabilities_country_all_betas = pd.read_parquet(
    posterior_probability_path
    + country
    + "/"
    + country
    + str(year)
    + "entire_country"
)

posterior_probability_lower_boundary=posterior_probabilities_country_all_betas[["CELLCODE","crop","posterior_probability"]].groupby(["CELLCODE","crop"]).min().reset_index()
posterior_probability_upper_boundary=posterior_probabilities_country_all_betas[["CELLCODE","crop","posterior_probability"]].groupby(["CELLCODE","crop"]).max().reset_index()
posterior_probability_lower_boundary.rename(columns={"posterior_probability":"posterior_probability_lower_boundary"},inplace=True)
posterior_probability_upper_boundary.rename(columns={"posterior_probability":"posterior_probability_upper_boundary"},inplace=True)

posterior_probabilities_country=pd.merge(posterior_probabilities_country,posterior_probability_lower_boundary,how="left",on=["CELLCODE","crop"])
posterior_probabilities_country=pd.merge(posterior_probabilities_country,posterior_probability_upper_boundary,how="left",on=["CELLCODE","crop"])

#%%
grid_1km_country=pd.merge(posterior_probabilities_country[["CELLCODE"]].drop_duplicates(),grid_1km_country,how="left",on="CELLCODE")
#%%
raster=rio.open(DGPCM_simulated_consistent_shares_path+"HDIs/SWHE_"+str(int(c*100))+"%HDIs_entire_EU_"+str(year)+".tif")
bounds=raster.bounds
crop_grid=grid_1km_country.copy()
east=((np.array(crop_grid.EOFORIGIN)-bounds[0])/1000).astype(int)
north=(np.abs((np.array(crop_grid.NOFORIGIN)-bounds[3])/1000)).astype(int)-1


#%%
""" alternative: import raster CIs, if available"""
all_crops_CIs_raster=pd.DataFrame()
for crop in list(selected_crops.keys()):
    print(crop)
    CI_raster=rio.open(DGPCM_simulated_consistent_shares_path+"HDIs/"+crop+"_"+str(int(c*100))+"%HDIs_entire_EU_"+str(year)+".tif")
    CI_raster_read=CI_raster.read()
    selected_crop_crid=crop_grid.copy()
    selected_crop_crid["lower_boundary_C0.9"]=CI_raster_read[0][north,east]/1000
    selected_crop_crid["upper_boundary_C0.9"]=CI_raster_read[1][north,east]/1000
    selected_crop_crid["crop"]=np.repeat(crop,len(selected_crop_crid))#
    all_crops_CIs_raster=pd.concat((all_crops_CIs_raster,selected_crop_crid))
#%%
#import true crop share information
true_crop_shares=pd.DataFrame()
for file in os.listdir(IACS_path):
    true_crop_shares=pd.concat((true_crop_shares,pd.read_csv(IACS_path+file)))
#%%
true_crop_shares.rename(columns={"DGPCM_code":"crop","cropshare_true":"true_crop_share"},inplace=True)


all_crops_CIs_raster=pd.merge(all_crops_CIs_raster,true_crop_shares[["CELLCODE","crop","true_crop_share"]],how="left",on=["CELLCODE","crop"])



#difference_true_share_expected_share=pd.merge(posterior_probabilities_country,CI_all_regions,
#                                              how="left",on=["CELLCODE","crop"])

difference_true_share_expected_share=pd.merge(posterior_probabilities_country,all_crops_CIs_raster,
                                              how="left",on=["CELLCODE","crop"])

difference_true_share_expected_share["abs_difference_true_expected"]=np.abs(
    np.array(difference_true_share_expected_share["posterior_probability"])-
    np.array(difference_true_share_expected_share["true_crop_share"])
)

difference_true_share_expected_share["interval_width"]=(
    np.array(difference_true_share_expected_share["upper_boundary_C0.9"])-
    np.array(difference_true_share_expected_share["lower_boundary_C0.9"])
    )

difference_true_share_expected_share["relative_difference_true_expected"]=np.abs(
    np.array(difference_true_share_expected_share["posterior_probability"])-
    np.array(difference_true_share_expected_share["true_crop_share"])
)/(np.array(difference_true_share_expected_share["true_crop_share"])+0.001)

#%%
"""plot width of total and posterior interval vs error"""
for crop in selected_crops.keys():
    quantiles=100

    data=difference_true_share_expected_share[difference_true_share_expected_share["crop"]==crop]
    try:
        data.drop("geometry",axis=1).dropna(inplace=True)
    except:
        pass

    data=data.iloc[:(len(data)//quantiles)*quantiles,:]
    data.sort_values(by="interval_width",inplace=True)

    data["lower_boundary_minus_expected"]=data["lower_boundary_C0.9"]-data["posterior_probability"]
    data["upper_boundary_minus_expected"]=data["upper_boundary_C0.9"]-data["posterior_probability"]
    data["true_minus_exp"]=data["true_crop_share"]-data["posterior_probability"]
    data["lower_boundary_posterior_minus_expected"]=data["posterior_probability_lower_boundary"]-data["posterior_probability"]
    data["upper_boundary_posterior_minus_expected"]=data["posterior_probability_upper_boundary"]-data["posterior_probability"]

    upper_boundary_minus_expected_matrix=np.array(data["upper_boundary_minus_expected"]).reshape(quantiles,-1)
    lower_boundary_minus_expected_matrix=np.array(data["lower_boundary_minus_expected"]).reshape(quantiles,-1)
    upper_boundary_posterior_minus_expected_matrix=np.array(data["upper_boundary_posterior_minus_expected"]).reshape(quantiles,-1)
    lower_boundary_posterior_minus_expected_matrix=np.array(data["lower_boundary_posterior_minus_expected"]).reshape(quantiles,-1)
    upper_boundary_minus_expected_mean=np.nanmean(upper_boundary_minus_expected_matrix,axis=1)
    lower_boundary_minus_expected_mean=np.nanmean(lower_boundary_minus_expected_matrix,axis=1)
    upper_boundary_posterior_minus_expected_mean=np.nanmean(upper_boundary_posterior_minus_expected_matrix,axis=1)
    lower_boundary_posterior_minus_expected_mean=np.nanmean(lower_boundary_posterior_minus_expected_matrix,axis=1)

    CR_posterior_interval=np.where((data["posterior_probability_lower_boundary"]<=data["true_crop_share"])&(data["posterior_probability_upper_boundary"]>=data["true_crop_share"]),1,0).mean()
    CR_sampled_interval=np.where((data["lower_boundary_C0.9"]<=data["true_crop_share"])&(data["upper_boundary_C0.9"]>=data["true_crop_share"]),1,0).mean()

    width_y_axis_cell=0.02

    upper_boundary_grid=((data["true_minus_exp"].max()//width_y_axis_cell)+width_y_axis_cell)*width_y_axis_cell
    lower_boundary_grid=(data["true_minus_exp"].min()//width_y_axis_cell)*width_y_axis_cell

    grid_y_range=np.arange(lower_boundary_grid,upper_boundary_grid+width_y_axis_cell,width_y_axis_cell)
    grid_x_range=np.arange(quantiles)

    density_matrix=np.ndarray((quantiles,len(grid_y_range)-1))
    true_minus_exp_matrix=np.array(data["true_minus_exp"]).reshape(quantiles,-1)

    for q in range(quantiles):
        density_matrix[q]=np.array(
            [len(np.where((true_minus_exp_matrix[q]>=grid_y_range[i])&(true_minus_exp_matrix[q]<grid_y_range[i+1]))[0])/len(true_minus_exp_matrix[q]) for i in np.arange(len(grid_y_range)-2,-1,-1)]
            )

    plt.rcParams.update({'font.size': 16})
    y, x = np.meshgrid(np.flip(grid_y_range)[1:], grid_x_range)
    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, density_matrix, cmap='Blues',vmin=0.0,vmax=0.03)

    plt.plot(lower_boundary_minus_expected_mean,c="red")
    plt.plot(upper_boundary_minus_expected_mean,c="red")
    plt.plot(lower_boundary_posterior_minus_expected_mean,c="black")
    plt.plot(upper_boundary_posterior_minus_expected_mean,c="black")
    plt.fill_between(np.arange(quantiles),upper_boundary_minus_expected_mean,lower_boundary_minus_expected_mean,zorder=3,alpha=0.2,color="red")
    plt.fill_between(np.arange(quantiles),upper_boundary_posterior_minus_expected_mean,lower_boundary_posterior_minus_expected_mean,zorder=3,alpha=0.6,color="black")
    plt.xlim(0,quantiles)
    #plt.legend()
    plt.title(selected_crops[crop]+" CR HDI:"+str(np.round((CR_sampled_interval),3))+" CR posterior: "+str(np.round((CR_posterior_interval),3)))
    plt.ylabel(r"error in $ km^2 $")
    plt.xlabel("90%-HDI width quantile")
    #plt.show()
    Path(output_CI_visualization_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_CI_visualization_path+country+str(year)+"_"+crop+".png")
    plt.close()
# %%

# %%

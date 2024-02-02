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


"""
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
IACS_path=intermediary_data_path+"Preprocessed_Inputs/IACS/true_shares/true_shares_"
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


selected_nuts2_regs = np.sort(nuts_regions_relevant[nuts_regions_relevant["LEVL_CODE"]==2]["NUTS_ID"])
selected_nuts1_regs=selected_nuts2_regs.astype("U3")

#%%

#%%
c_interval_statistics_dict = {}
UA_and_PA_dict = {}
for nuts1 in np.unique(selected_nuts1_regs):
    #consistent_samples = pd.read_parquet(consistent_samples_path+nuts1)
    consistent_samples=pd.read_parquet(DGPCM_simulated_consistent_shares_path+country+"/"+str(year)+"/"+country+str(year)+"_"+nuts1)
    for nuts2 in selected_nuts2_regs[np.where(selected_nuts1_regs==nuts1)[0]]:
        print(f"start calculation for region {nuts2}...")


        true_shares = pd.read_csv(IACS_path+nuts2+"_"+str(year)+".csv")

        relevant_cells_observed = true_shares["CELLCODE"].value_counts().keys()
        relevant_cells_available = (
            consistent_samples[
                consistent_samples["CELLCODE"].isin(relevant_cells_observed)
            ]["CELLCODE"]
            .value_counts()
            .keys()
        )
        consistent_samples_relevant=consistent_samples.iloc[np.where(np.array(consistent_samples["NUTS_ID"]).astype("U4")==nuts2)[0]]
        consistent_samples_relevant = consistent_samples_relevant[
            consistent_samples_relevant["CELLCODE"].isin(relevant_cells_available)
        ]
      
        true_shares = true_shares[true_shares["CELLCODE"].isin(relevant_cells_available)]

        K=consistent_samples_relevant["CELLCODE"].value_counts().values.min()

        """
        for some cells we have more than K*n_of_param_samples share samples, as these cells appear in more than 1 NUTS3 region 
        (because they are at the corner of a NUTS3 region). We drop duplicates and keep only K*n_of_param_samples of the samples from these cells


        """
        grouped_df=consistent_samples_relevant[["NUTS_ID","CELLCODE","beta"]].groupby(["CELLCODE","NUTS_ID","beta"]).sum().reset_index()
        duplicates_dropped_df=grouped_df.drop_duplicates(["CELLCODE","beta"])
        consistent_samples_relevant=pd.merge(duplicates_dropped_df,consistent_samples_relevant,how="left",on=["CELLCODE","NUTS_ID","beta"])

        relevant_crops = np.array(consistent_samples_relevant.columns)[3:]
        C = len(relevant_crops)
        consistent_samples_relevant.sort_values(by="CELLCODE", inplace=True)

        

        interval_statistics = {
            "crop": [],
            "share_in_c_interval": [],
            "mean_width_c_interval": [],
            "max_width_c_interval": [],
        }
        #due to some naming difference between older and newer versions it cannot be guaranteed that all
        #input files have the new column name "DGPCM_code"
        if "CAPRI_code" in true_shares.columns:
            correct_column_name="CAPRI_code"
        elif "DGPCM_code" in true_shares.columns:
            correct_column_name="DGPCM_code"
        true_shares_pivot=true_shares.pivot(index="CELLCODE",columns=correct_column_name,values="cropshare_true").reset_index().sort_values(by="CELLCODE")
        
        all_interval_results=pd.DataFrame()
        for crop in relevant_crops:
            if crop not in true_shares_pivot.columns:
                continue
            true_shares_selected_crop=true_shares_pivot[crop]
            predicted_shares_selected_crop=np.array(consistent_samples_relevant[crop])
            I=len(true_shares_selected_crop)
            predicted_shares_selected_crop_matrix=predicted_shares_selected_crop.reshape(I,K)
            c_hdi = np.ndarray((I, 2))

            for i in range(I):
                c_hdi[i] = az.hdi(predicted_shares_selected_crop_matrix[i], c)


            interval_results=pd.DataFrame(
                {"CELLCODE":true_shares_pivot["CELLCODE"],
                "crop":np.repeat(crop,I),
                "true_crop_share":true_shares_selected_crop,
                f"lower_boundary_C{c}":c_hdi.transpose()[0],
                f"upper_boundary_C{c}":c_hdi.transpose()[1]
                }
            )
            all_interval_results=pd.concat((all_interval_results,interval_results))

        in_interval_array=np.zeros(len(all_interval_results))
        in_interval_array[np.where((all_interval_results[f"lower_boundary_C{c}"]<=all_interval_results["true_crop_share"])&(all_interval_results[f"upper_boundary_C{c}"]>=all_interval_results["true_crop_share"]))[0]]=1 
        all_interval_results["true_crop_share_in_interval"]=in_interval_array
    
        
        
        print("export evaluation results...")
        Path(output_path+country+"/").mkdir(parents=True, exist_ok=True)
        all_interval_results.to_csv(output_path+country+"/"+nuts2+str(year)+"_"+str(c)+"%_HDI.csv")
#%%
"""ANALYZE CI"""
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

# %%
"""import CIs"""
CI_all_regions=pd.DataFrame()
for nuts2 in selected_nuts2_regs:
    CI_region=pd.read_csv(output_path+country+"/"+nuts2+str(year)+"_"+str(c)+"%_HDI.csv")
    CI_all_regions=pd.concat((CI_all_regions,CI_region))




#%%
difference_true_share_expected_share=pd.merge(posterior_probabilities_country,CI_all_regions,
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
selected_crops={
    "GRAS":"Grass",
    "SWHE":"Soft Wheat",
    "LMAIZ": "Maize",
    "BARL":"Barley",
    "LRAPE":"Rapeseed",
    "OFAR":"Other Forage Plants",
    "SUNF":"Sunflowers",
    "VINY":"Vinyeards"
}
#%%
"""plot width of total and posterior interval vs error"""
for crop in selected_crops.keys():
    quantiles=100

    data=difference_true_share_expected_share[difference_true_share_expected_share["crop"]==crop]
    data.dropna(inplace=True)

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
    plt.legend()
    plt.title(selected_crops[crop])
    plt.ylabel(r"error in $ km^2 $")
    plt.xlabel("90%-HDI width quantile")
    Path(output_CI_visualization_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_CI_visualization_path+country+str(year)+"_"+crop+".png")
    plt.close()
# %%

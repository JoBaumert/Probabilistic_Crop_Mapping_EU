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
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
from pathlib import Path
import os, zipfile
# %%
#default crops:
map_for_entire_EU=False #when this is true you also need the shapefiles for counties not in the EU but on the map (e.g., Albania) for visualization
selected_crops=["GRAS","SWHE","LMAIZ"]
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]

raw_data_path = data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
parameter_path=data_main_path+"delineation_and_parameters/DGPCM_user_parameters.xlsx"
nuts_path=intermediary_data_path+"Preprocessed_Inputs/NUTS/NUTS_all_regions_all_years"
excluded_NUTS_regions_path = data_main_path+"delineation_and_parameters/excluded_NUTS_regions.xlsx"
crop_delineation_path=data_main_path+"delineation_and_parameters/DGPCM_crop_delineation.xlsx"
grid_path=raw_data_path+"Grid/"
posterior_probability_path=data_main_path+"Results/Posterior_crop_probability_estimates/"
output_path=data_main_path+"Results/Validations_and_Visualizations/Expected_crop_shares/"

#only needed when making a map for the entire EU to visually fill the gaps for non-EU countries on the map
albania_shapefile_path=raw_data_path+"NUTS/albania_shapefile.zip!/ALB_adm0.shp"
bosnia_shapefile_path=raw_data_path+"NUTS/bosnia_shapefile.zip!/BIH_adm0.shp"
kosovo_shapefile_path=raw_data_path+"NUTS/kosovo_shapefile.zip!/XKO_adm0.shp"
serbia_shapefile_path=raw_data_path+"NUTS/serbia_shapefile.zip!/SRB_adm0.shp"



# import parameters
selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])


year_country_crop_list=[np.tile(selected_years,len(country_codes_relevant)*len(selected_crops)),
        np.repeat(country_codes_relevant,len(selected_years)*len(selected_crops)),
     np.tile(np.repeat(selected_crops,len(selected_years)),len(country_codes_relevant))]

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--crop", type=str, required=False)
parser.add_argument("-cc","--countrycode", type=str, required=False)
parser.add_argument("-y", "--year", type=int, required=False)
parser.add_argument("--from_xlsx", type=str, required=False)

args = parser.parse_args()
if args.from_xlsx == "True":
    year_country_crop_list=[np.tile(selected_years,len(country_codes_relevant)*len(selected_crops)),
        np.repeat(country_codes_relevant,len(selected_years)*len(selected_crops)),
     np.tile(np.repeat(selected_crops,len(selected_years)),len(country_codes_relevant))]
elif args.year is not None:
    year_country_crop_list = [[args.year], [args.countrycode],[args.crop]]

#year=args.year

beta = 0
#%%
year_crop_list=[np.repeat(np.unique(year_country_crop_list[0]),len(np.unique(year_country_crop_list[2]))),
            np.tile(np.unique(year_country_crop_list[2]),len(np.unique(year_country_crop_list[0])))]
#%%
all_NUTS_boundaries=gpd.read_file(nuts_path+".shp")

#%%
if __name__ == "__main__":
    print("generate visualization of crop map for selected crops, country and year")
    for c in range(len(year_crop_list[0])):
        year = year_crop_list[0][c]
        selected_crop = year_crop_list[1][c]
        print(selected_crop+" in "+str(year))
    
        nuts_years=pd.read_excel(parameter_path,sheet_name="NUTS")
        relevant_nuts_year=nuts_years[nuts_years["crop_map_year"]==year]["nuts_year"].iloc[0]
        excluded_NUTS_regions = pd.read_excel(excluded_NUTS_regions_path)
        excluded_NUTS_regions = np.array(
            excluded_NUTS_regions["excluded NUTS1 regions"]
            )
        excluded_NUTS_regions_countries=np.unique(excluded_NUTS_regions.astype("U2")) #countries that contain some excluded nuts regions
        nuts_info = pd.read_excel(parameter_path, sheet_name="NUTS")
        all_years = np.array(nuts_info["crop_map_year"])
        nuts_years = np.sort(nuts_info["nuts_year"].value_counts().keys())
        relevant_nuts_years = np.array(nuts_info["nuts_year"])
        
        NUTS_boundaries = all_NUTS_boundaries[(all_NUTS_boundaries["year"]==year)]
        
       
        #%%
        all_country_boundaries = pd.DataFrame()
        all_country_codes=np.unique(np.array(year_country_crop_list[1]))
        # for country in all_country_codes:
        for country in all_country_codes:
            if country in excluded_NUTS_regions_countries:
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
                            excluded_NUTS_regions
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
                    ["CNTR_CODE","geometry"]
                ]
                relevant_boundaries_country.insert(0, "country", country)
            all_country_boundaries = pd.concat(
                (all_country_boundaries, relevant_boundaries_country)
            )

       
        #%%
        #insert countries for which boundaries so far are not in the shapefile
        if map_for_entire_EU:
            all_country_boundaries.drop(columns=["country"],inplace=True)

            albania_boundary=gpd.read_file(albania_shapefile_path)
            bosnia_boundary=gpd.read_file(bosnia_shapefile_path)
            kosovo_boundary=gpd.read_file(kosovo_shapefile_path)
            serbia_boundary=gpd.read_file(serbia_shapefile_path)

            albania_boundary_epsg3035=albania_boundary.to_crs("epsg:3035")
            bosnia_boundary_epsg3035=bosnia_boundary.to_crs("epsg:3035")
            kosovo_boundary_epsg3035=kosovo_boundary.to_crs("epsg:3035")
            serbia_boundary_epsg3035=serbia_boundary.to_crs("epsg:3035")

            albania_boundary_epsg3035=albania_boundary_epsg3035["geometry"]
            bosnia_boundary_epsg3035=bosnia_boundary_epsg3035["geometry"]
            kosovo_boundary_epsg3035=kosovo_boundary_epsg3035["geometry"]
            serbia_boundary_epsg3035=serbia_boundary_epsg3035["geometry"]

            albania_boundary_epsg3035_df=pd.DataFrame(albania_boundary_epsg3035)
            albania_boundary_epsg3035_df.insert(0,"CNTR_CODE","AL")
            bosnia_boundary_epsg3035_df=pd.DataFrame(bosnia_boundary_epsg3035)
            bosnia_boundary_epsg3035_df.insert(0,"CNTR_CODE","BA")
            kosovo_boundary_epsg3035_df=pd.DataFrame(kosovo_boundary_epsg3035)
            kosovo_boundary_epsg3035_df.insert(0,"CNTR_CODE","XK")
            serbia_boundary_epsg3035_df=pd.DataFrame(serbia_boundary_epsg3035)
            serbia_boundary_epsg3035_df.insert(0,"CNTR_CODE","RS")

            all_country_boundaries_df=pd.concat((all_country_boundaries,albania_boundary_epsg3035_df))
            all_country_boundaries_df=pd.concat((all_country_boundaries_df,bosnia_boundary_epsg3035_df))
            all_country_boundaries_df=pd.concat((all_country_boundaries_df,kosovo_boundary_epsg3035_df))
            all_country_boundaries_df=pd.concat((all_country_boundaries_df,serbia_boundary_epsg3035_df))
        #%%
        #selected_crop="GRAS"
        all_countries_selected_crop_share_df=pd.DataFrame()
        for country in all_country_codes:

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
            
            posterior_probabilities_country=posterior_probabilities_country[posterior_probabilities_country["crop"]==selected_crop]
            
            posterior_probabilities_country = pd.merge(
                posterior_probabilities_country[["CELLCODE", "crop", "posterior_probability"]],
                grid_1km_country[["CELLCODE", "geometry"]],
                how="left",
                on="CELLCODE",
            )
            posterior_probabilities_country.insert(
                0, "country", np.repeat(country, len(posterior_probabilities_country))
            )
            all_countries_selected_crop_share_df = pd.concat(
                (all_countries_selected_crop_share_df, posterior_probabilities_country)
            )


        #%%
        all_countries_selected_crop_share_gdf=gpd.GeoDataFrame(all_countries_selected_crop_share_df)

        # %%
        if selected_crop=="GRAS":
            selected_cmap="YlGn"
            max_val=1
        elif selected_crop=="SWHE":
            selected_cmap="YlOrRd"
            max_val=0.6
        elif selected_crop=="LMAIZ":
            selected_cmap="Blues"
            max_val=0.6

        
        plt.figure(figsize=(12, 12))
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        gpd.GeoDataFrame(all_country_boundaries).plot(ax=ax, facecolor="lightgrey")
        all_countries_selected_crop_share_gdf.plot(
            ax=ax,
            column="posterior_probability",
            legend=True,
            cmap=selected_cmap,  # YlGn "YlOrRd"
            vmin=0,
            vmax=max_val,
        )

        
        """
        gpd.GeoDataFrame(all_country_boundaries).plot(ax=ax, facecolor="lightgrey")
        all_countries_selected_crop_share_gdf.plot(
            ax=ax,
            column="posterior_probability",
            legend=True,
            cmap=selected_cmap,  # YlGn "YlOrRd"
            vmin=0,
            vmax=max_val,
        )
        """
        plt.title(f"Share of {selected_crop} in {year}")
        plt.axis("off")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path+"share_of_"+selected_crop+"_"+country+"_"+str(year)+".png")
        plt.close(fig)
        #%%
        output_path
# %%

# %%

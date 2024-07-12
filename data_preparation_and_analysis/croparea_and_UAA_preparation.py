# %%
# from gettext import find

from timeit import repeat
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
from rasterio.plot import show
import rasterio as rio
import sys
from shapely import wkt
from pathlib import Path
import os

#%%

data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]



delineation_and_parameter_path = (
    data_main_path+"delineation_and_parameters/"
)
raw_data_path=data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/Preprocessed_Inputs/"
# input files
parameter_path = delineation_and_parameter_path + "DGPCM_user_parameters.xlsx"
crop_delineation_path = delineation_and_parameter_path + "DGPCM_crop_delineation.xlsx"
nuts_path = intermediary_data_path+"NUTS/NUTS_all_regions_all_years.csv"
preprocessed_eurostat_path=intermediary_data_path+"Eurostat/"
excluded_NUTS_regions_path=delineation_and_parameter_path+ "excluded_NUTS_regions.xlsx"
# output files
output_path=data_main_path+"Intermediary_Data/Regional_Aggregates/"



#%%
selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])

nuts_year_info=pd.read_excel(parameter_path,sheet_name="NUTS")
relevant_nuts_years=np.array(nuts_year_info[nuts_year_info["crop_map_year"].isin(selected_years)]["nuts_year"])

crop_info = pd.read_excel(parameter_path, sheet_name="crops")
all_crops = np.array(crop_info["crop"])
levl_info = pd.read_excel(parameter_path, sheet_name="lowest_agg_level")
country_levls = {}
for i in range(len(levl_info)):
    country_levls[levl_info.iloc[i].country_code] = levl_info.iloc[i].lowest_agg_level

excluded_NUTS_regions = pd.read_excel(excluded_NUTS_regions_path)
excluded_NUTS_regions = np.array(excluded_NUTS_regions["excluded NUTS1 regions"])

# load data on NUTS regions in a year
NUTS_data = pd.read_csv(nuts_path)

#%%
cropdata_raw = pd.read_csv(preprocessed_eurostat_path+"Eurostat_cropdata_compiled_"+str(selected_years.min())+str(selected_years.max())+"_DGPCMcodes.csv")
UAA_raw = pd.read_csv(preprocessed_eurostat_path+"Eurostat_UAA_compiled_"+str(selected_years.min())+str(selected_years.max())+".csv")
cropdata_raw.rename(columns={"DGPCM_crop_code": "crop"}, inplace=True)

#%%
NUTS_data=NUTS_data[(NUTS_data["year"].isin(selected_years))&(NUTS_data["CNTR_CODE"].isin(country_codes_relevant))]

#%%
"""
for some crops no data is available, which is why there is no entry in the table for this crop.
to ensure that for every crop there is an entry (even if it is a nan value), we first get all NUTS regions and
years and then construct a dataframe that contains an entry for every crop in every region and in every year
"""
# relevant_NUTS_and_years=cropdata_raw[["NUTS_ID","year","area"]].groupby(["NUTS_ID","year"]).sum().reset_index()[["NUTS_ID","year"]]
relevant_NUTS_and_years = NUTS_data[["NUTS_ID", "year"]].sort_values(
    by=["NUTS_ID", "year"]
)
NUTS_ID_array = np.repeat(np.array(relevant_NUTS_and_years["NUTS_ID"]), len(all_crops))
year_array = np.repeat(np.array(relevant_NUTS_and_years["year"]), len(all_crops))
all_crops_array = np.tile(all_crops, int(len(year_array) / len(all_crops)))
NUTS_levl_array = np.vectorize(len)(NUTS_ID_array) - 2
cropdata_construction_df = pd.DataFrame(
    {
        "country": NUTS_ID_array.astype("U2"),
        "NUTS_ID": NUTS_ID_array,
        "NUTS_LEVL": NUTS_levl_array,
        "year": year_array,
        "crop": all_crops_array,
    }
)

cropdata_raw = pd.merge(
    cropdata_raw,
    cropdata_construction_df,
    how="right",
    on=["country", "NUTS_ID", "year", "crop"],
)
cropdata_raw.fillna(0, inplace=True)
#%%
cropdata_raw[cropdata_raw["NUTS_ID"]=="AT112"]

#%%
cropdata_corrected_1 = cropdata_raw.copy()
excluded_NUTS_regions_countries = np.unique(excluded_NUTS_regions.astype("U2"))

# some NUTS_regions are excluded (e.g., French oversea area). Substract the known crop areas for those regions from the national crop area
for year in selected_years:
    for country in excluded_NUTS_regions_countries:
        relevant_NUTS1_regions = excluded_NUTS_regions[
            np.where(excluded_NUTS_regions.astype("U2") == country)[0]
        ]
        cropdata_relevant = cropdata_corrected_1[
            (cropdata_corrected_1["NUTS_ID"].isin(relevant_NUTS1_regions))
            & (cropdata_corrected_1["year"] == year)
        ]
        for crop in all_crops:
            try:
                croparea_national = cropdata_corrected_1.at[
                    np.where(
                        (cropdata_corrected_1["NUTS_ID"] == country)
                        & (cropdata_corrected_1["year"] == year)
                        & (cropdata_corrected_1["crop"] == crop)
                    )[0][0],
                    "area",
                ]
            except:
                continue
            croparea_excluded_nuts_regs = cropdata_relevant[
                cropdata_relevant["crop"] == crop
            ]["area"].sum()
            cropdata_corrected_1.at[
                np.where(
                    (cropdata_corrected_1["NUTS_ID"] == country)
                    & (cropdata_corrected_1["year"] == year)
                    & (cropdata_corrected_1["crop"] == crop)
                )[0][0],
                "area",
            ] = (
                croparea_national - croparea_excluded_nuts_regs
            )
# then, drop all data for the excluded NUTS regions
cropdata_corrected_1 = cropdata_corrected_1.iloc[
    np.where(
        np.isin(
            np.array(cropdata_corrected_1["NUTS_ID"]).astype("U3"),
            excluded_NUTS_regions,
        ).astype(int)
        == 0
    )[0]
]
#%%
cropdata_corrected_1[cropdata_corrected_1["NUTS_LEVL"]==3]
#%%
"""CORRECTION OF OBVIOUS DATA ERRORS"""
"""some data entries don't make sense at all (for example, no grass for UK in 2020)
some of the errors are hard to identify and hard to correct. While not all of them are too problematic, it would still be good 
if we could identify and correct those errors in the future.
here we only correct two very obvious (OCRO in Spain in 2011 and GRAS in 2020 in UK)
"""
cropdata_corrected_1.reset_index(inplace=True)

cropdata_corrected_2 = cropdata_corrected_1.copy()

invalid_countries, invalid_years, invalid_crops = (
    ["ES", "UK"],
    [2011, 2020],
    ["OCRO", "GRAS"],
)
# take the mean of the two values closest to the incorrect data as a replacement
for ic, invalid_country in enumerate(invalid_countries):
    if invalid_country in country_codes_relevant:
        invalid_year = invalid_years[ic]
        invalid_crop = invalid_crops[ic]
        if invalid_year in selected_years:
            closest_years = selected_years[np.where(np.abs(invalid_year - selected_years) == 1)[0]]
            if len(closest_years) == 1:  # if year is either 2010 or 2020
                closest_years = np.sort(
                    np.append(
                        closest_years,
                        selected_years[np.where(np.abs(invalid_year - selected_years) == 2)[0][0]],
                    )
                )
            corrected_value = cropdata_corrected_1[
                (cropdata_corrected_1["NUTS_ID"] == invalid_country)
                & (cropdata_corrected_1["year"].isin(closest_years))
                & (cropdata_corrected_1["crop"] == invalid_crop)
            ]["area"].mean()
            # correct value in the cropdata_corrected_2 df:
            cropdata_corrected_2.at[
                np.where(
                    (cropdata_corrected_2["NUTS_ID"] == invalid_country)
                    & (cropdata_corrected_2["year"] == invalid_year)
                    & (cropdata_corrected_2["crop"] == invalid_crop)
                )[0][0],
                "area",
            ] = corrected_value


#%%
"""
we want to ensure that the total acreage of a crop at lower administrative level never exceeds the acreage of the same crop
at higher administrative level (i.e., sum of crop c at NUTS2 <= sum of crop c at NUTS1 <= sum of crop c at NUTS0).
We therefore go through all relevant NUTS levels and check if this is the case, otherwise we update the crop acreage at the higher administrative level
by assigning it the sum of the acreage of crop c at the lower administrative levels. 
"""
cropdata_corrected_3 = pd.DataFrame()

for country in country_codes_relevant:
    for year in selected_years:
      
        df_relevant = cropdata_corrected_2[
            (cropdata_corrected_2["country"] == country)
            & (cropdata_corrected_2["year"] == year)
        ]

        highest_NUTS_level = df_relevant[df_relevant["area"] > 0]["NUTS_LEVL"].max()
        corrected_df = df_relevant[(df_relevant["NUTS_LEVL"] == highest_NUTS_level)][
            ["NUTS_ID", "crop", "area"]
        ]

        for level in np.arange(highest_NUTS_level, -1, -1):
            df_relevant_onelevel = df_relevant[
                df_relevant["NUTS_LEVL"] == level
            ].sort_values(by=["NUTS_ID", "crop"])
            if level < highest_NUTS_level:
                corrected_df_part = df_relevant_onelevel[["NUTS_ID", "crop"]].copy()
                corrected_df_part["area"] = np.vstack(
                    (
                        np.array(df_relevant_onelevel["area"]),
                        np.array(sum_of_lower_levels["area"]),
                    )
                ).max(axis=0)
                corrected_df = pd.concat((corrected_df, corrected_df_part))
            if level > 0:
                corrected_df_higher_level = corrected_df.iloc[
                    np.where(
                        np.vectorize(len)(np.array(corrected_df["NUTS_ID"]))
                        == level + 2
                    )[0]
                ].copy()
                corrected_df_higher_level["NUTS_ID_higher_level"] = np.array(
                    corrected_df_higher_level["NUTS_ID"]
                ).astype(f"U{int(level+1)}")
                sum_of_lower_levels = (
                    corrected_df_higher_level[["NUTS_ID_higher_level", "crop", "area"]]
                    .groupby(["NUTS_ID_higher_level", "crop"])
                    .sum()
                    .reset_index()
                )

        if (
            np.square(
                np.array(corrected_df.groupby("crop").max()["area"])
                - np.array(corrected_df[corrected_df["NUTS_ID"] == country]["area"])
            ).max()
            > 0.01
        ):
            raise Exception(
                "some lower level values are apparently larger than the national value"
            )

        corrected_df.insert(0, "country", np.repeat(country, len(corrected_df)))
        corrected_df.insert(1, "year", np.repeat(year, len(corrected_df)))
        cropdata_corrected_3 = pd.concat((cropdata_corrected_3, corrected_df))


#%%
"""UAA"""

relevant_nuts_year_combinations = (
    cropdata_corrected_3[["country", "NUTS_ID", "year", "area"]]
    .groupby(["country", "NUTS_ID", "year"])
    .sum()
    .reset_index()[["country", "NUTS_ID", "year"]]
)
UAA_relevant = pd.merge(
    relevant_nuts_year_combinations,
    UAA_raw,
    how="left",
    on=["country", "NUTS_ID", "year"],
)
# for some years there are two entries for a given NUTS regions: a "nan" value and a non-empty value.
UAA_relevant = UAA_relevant.groupby(["country", "NUTS_ID", "year"]).mean().reset_index()

#%%
# the UAA is not known for some years and regions.

UAA_corrected1 = pd.merge(
    cropdata_corrected_3[["NUTS_ID", "year", "area"]]
    .groupby(["NUTS_ID", "year"])
    .sum()
    .reset_index(),
    UAA_relevant,
    how="left",
    on=["NUTS_ID", "year"],
)

UAA_corrected1["NUTS_LEVL"] = np.vectorize(len)(UAA_corrected1["NUTS_ID"]) - 2

#%%
#replace Nan values by the mean of the respective UAA region in all years for which the value is not nan
UAA_corrected1_mean=UAA_corrected1[["NUTS_ID","UAA"]].groupby("NUTS_ID").mean().rename(columns={"UAA":"mean_UAA"})
UAA_corrected1=pd.merge(UAA_corrected1,UAA_corrected1_mean,how="left",on="NUTS_ID")
UAA_corrected1_array=np.array(UAA_corrected1["UAA"])
UAA_corrected1_array[np.where(np.isnan(UAA_corrected1_array))[0]]=np.array(UAA_corrected1["mean_UAA"])[np.where(np.isnan(UAA_corrected1_array))[0]]
UAA_corrected1["UAA"]=UAA_corrected1_array

UAA_corrected1.drop("mean_UAA",axis=1,inplace=True)


#%%
#only keept UAA information at the lowest administrative levle available (at higher levels, it is just the sum of those lower levels)
UAA_corrected2 = pd.DataFrame()
for country in country_codes_relevant:
    for year in selected_years:

        UAA_corrected1_selection = UAA_corrected1[
            (UAA_corrected1["year"] == year) & (UAA_corrected1["country"] == country)
        ]

        max_level = UAA_corrected1_selection["NUTS_LEVL"].max()
        UAA_corrected1_selection = UAA_corrected1_selection[
            UAA_corrected1_selection["NUTS_LEVL"] == max_level
        ]
        UAA_corrected2 = pd.concat((UAA_corrected2, UAA_corrected1_selection))

#%%


""" ensure that the UAA is for every region, year and country at least as large as the required area.
Besides, ensure that the UAA at one level is for every country, year and region equal to the sum of the lower level UAAs
"""
#first make sure that at the lowest level of administrative regions the UAA is at least as large as the crop area
UAA_corrected2["UAA_corrected"] = np.nanmax(
    np.array(UAA_corrected2[["area", "UAA"]]), axis=1
)

UAA_corrected3 = pd.DataFrame()
for country in country_codes_relevant:
    for year in selected_years:

        UAA_selected = UAA_corrected2[
            (UAA_corrected2["year"] == year) & (UAA_corrected2["country"] == country)
        ][["NUTS_ID", "NUTS_LEVL", "UAA_corrected"]].sort_values(by="NUTS_ID")
        crop_area_regions = (
            cropdata_corrected_3[
                (cropdata_corrected_3["country"] == country)
                & (cropdata_corrected_3["year"] == year)
            ][["NUTS_ID", "area"]]
            .groupby("NUTS_ID")
            .sum()
            .reset_index()
        )
        nuts_level_max = UAA_selected["NUTS_LEVL"].max()
        most_disaggregated_UAA = np.array(UAA_selected["UAA_corrected"])

        if nuts_level_max > 0:
            for level in np.arange(nuts_level_max, 0, -1):
                UAA_selected[f"NUTS{level-1}"] = np.array(
                    UAA_selected["NUTS_ID"]
                ).astype(f"U{level+1}")

        for level in np.arange(nuts_level_max, 0, -1):
            selected_level_UAA = (
                UAA_selected[[f"NUTS{level-1}", "UAA_corrected"]]
                .groupby(f"NUTS{level-1}")
                .sum()
                .reset_index()
            )
            selected_level_UAA.rename(
                columns={f"NUTS{level-1}": "NUTS_ID"}, inplace=True
            )
            comparison = pd.merge(
                selected_level_UAA, crop_area_regions, how="left", on="NUTS_ID"
            )
            comparison_UAA_corrected = np.array(comparison["UAA_corrected"])
            comparison_UAA_corrected = np.where(
                comparison_UAA_corrected == 0, -1, comparison_UAA_corrected
            )
            comparison.drop(columns="UAA_corrected", inplace=True)
            comparison["UAA_corrected"] = comparison_UAA_corrected

            relative_difference = (
                np.array(comparison["area"]) - np.array(comparison["UAA_corrected"])
            ) / np.array(comparison["UAA_corrected"]) + 1

            for i in np.where(relative_difference)[0]:
                if relative_difference[i] < 0:
                    n_of_lower_regions = len(
                        np.where(
                            UAA_selected[f"NUTS{level-1}"]
                            == comparison.iloc[i]["NUTS_ID"]
                        )[0]
                    )
                    most_disaggregated_UAA[
                        np.where(
                            UAA_selected[f"NUTS{level-1}"]
                            == comparison.iloc[i]["NUTS_ID"]
                        )[0]
                    ] = (-relative_difference[i] / n_of_lower_regions)
                elif relative_difference[i] > 1:
                    most_disaggregated_UAA[
                        np.where(
                            UAA_selected[f"NUTS{level-1}"]
                            == comparison.iloc[i]["NUTS_ID"]
                        )[0]
                    ] = (
                        most_disaggregated_UAA[
                            np.where(
                                UAA_selected[f"NUTS{level-1}"]
                                == comparison.iloc[i]["NUTS_ID"]
                            )[0]
                        ]
                        * relative_difference[i]
                    )
            UAA_selected.drop(columns="UAA_corrected", inplace=True)
            UAA_selected["UAA_corrected"] = most_disaggregated_UAA

        UAA_corrected_country = UAA_selected[["NUTS_ID", "UAA_corrected"]]
        for level in np.arange(nuts_level_max - 1, -1, -1):
            selection = (
                UAA_selected[[f"NUTS{level}", "UAA_corrected"]]
                .groupby(f"NUTS{level}")
                .sum()
                .reset_index()
            )
            selection.rename(columns={f"NUTS{level}": "NUTS_ID"}, inplace=True)
            UAA_corrected_country = pd.concat((UAA_corrected_country, selection))

        UAA_corrected_country.insert(
            0, "country", np.repeat(country, len(UAA_corrected_country))
        )
        UAA_corrected_country.insert(
            1, "year", np.repeat(year, len(UAA_corrected_country))
        )
        UAA_corrected3 = pd.concat((UAA_corrected3, UAA_corrected_country))

        excess = np.diff(
            np.array(
                pd.merge(
                    UAA_corrected_country, crop_area_regions, how="left", on="NUTS_ID"
                )[["UAA_corrected", "area"]]
            ),
            axis=1,
        ).max()
        if excess > 0:
            print(
                "needed crop area exceeds UAA by "
                + str(excess)
                + " in "
                + country
                + " in "
                + str(year)
            )


#%%
"""if the available UAA at national level is larger than the required UAA, multiply only the national area of each crop by a factor to ensure that
 the national crop area is equal to the national UAA
"""
national_UAAs = UAA_corrected3.iloc[
    np.where(UAA_corrected3["country"] == UAA_corrected3["NUTS_ID"])
][["country", "year", "UAA_corrected"]]
national_required_cropdata = (
    cropdata_corrected_3.iloc[
        np.where(cropdata_corrected_3["country"] == cropdata_corrected_3["NUTS_ID"])
    ][["country", "year", "area"]]
    .groupby(["country", "year"])
    .sum()
    .reset_index()
)

comparison_national_UAA_and_croparea = pd.merge(
    national_UAAs, national_required_cropdata, how="left", on=["country", "year"]
)

weightfactor = np.array(
    (
        comparison_national_UAA_and_croparea["UAA_corrected"]
        - comparison_national_UAA_and_croparea["area"]
    )
    / comparison_national_UAA_and_croparea["area"]
    + 1
)
weightfactor = np.where(weightfactor <= 1, 1, weightfactor)

comparison_national_UAA_and_croparea["weightfactor"] = weightfactor
# %%
cropdata_corrected_3_array = np.array(cropdata_corrected_3["area"])
for i in range(len(comparison_national_UAA_and_croparea)):
    country = comparison_national_UAA_and_croparea.iloc[i]["country"]
    year = comparison_national_UAA_and_croparea.iloc[i]["year"]
    factor = comparison_national_UAA_and_croparea.iloc[i]["weightfactor"]
    cropdata_corrected_3_array[
        np.where(
            (cropdata_corrected_3["NUTS_ID"] == country)
            & (cropdata_corrected_3["year"] == year)
        )[0]
    ] = (
        cropdata_corrected_3_array[
            np.where(
                (cropdata_corrected_3["NUTS_ID"] == country)
                & (cropdata_corrected_3["year"] == year)
            )[0]
        ]
        * factor
    )
# %%
cropdata_corrected_4 = cropdata_corrected_3.copy()
cropdata_corrected_4.drop(columns="area", inplace=True)
cropdata_corrected_4["area"] = cropdata_corrected_3_array


#%%
#Exort data
print("export data...")
Path(output_path).mkdir(parents=True, exist_ok=True)
cropdata_corrected_4.to_csv(output_path+"cropdata_"+str(selected_years.min())+str(selected_years.max())+".csv")
UAA_corrected3.to_csv(output_path+"coherent_UAA_"+str(selected_years.min())+str(selected_years.max())+".csv")


# %%

"""
3) FIND OUT AT WHICH LEVEL INFORMATION ON CROP AREA IS (IN)CONSISTENT
"""


df_cropdata_selected_years =cropdata_corrected_4

#%%
consistency_dict = {
    "country": [],
    "year": [],
    "crop": [],
    "0level_value": [],
    "1level_value": [],
    "2level_value": [],
    "3level_value": [],
    "01_consistency": [],
    "02_consistency": [],
    "03_consistency": []
}
for country in country_codes_relevant:
    for y, year in enumerate(selected_years):
        relevant_nuts3_regs = np.sort(np.array(NUTS_data[(NUTS_data["CNTR_CODE"]==country)&(NUTS_data["year"]==year)&(NUTS_data["LEVL_CODE"]==3)]["NUTS_ID"]))
        relevant_nuts3_regs = relevant_nuts3_regs[np.where(np.isin(relevant_nuts3_regs.astype("U3"),excluded_NUTS_regions,invert=True))[0]]
        relevant_nuts2_regs = relevant_nuts3_regs.astype("U4")
        relevant_nuts1_regs = relevant_nuts2_regs.astype("U3")

        nuts3_level_index = np.where(
            (df_cropdata_selected_years["NUTS_ID"].isin(relevant_nuts3_regs))
            & (df_cropdata_selected_years["year"] == year)
        )[0]
        nuts2_level_index = np.where(
            (df_cropdata_selected_years["NUTS_ID"].isin(relevant_nuts2_regs))
            & (df_cropdata_selected_years["year"] == year)
        )[0]
        nuts1_level_index = np.where(
            (df_cropdata_selected_years["NUTS_ID"].isin(relevant_nuts1_regs))
            & (df_cropdata_selected_years["year"] == year)
        )[0]
        nuts0_level_index = np.where(
            (df_cropdata_selected_years["NUTS_ID"] == country)
            & (df_cropdata_selected_years["year"] == year)
        )[0]

        nuts3_df = df_cropdata_selected_years.iloc[nuts3_level_index]
        nuts2_df = df_cropdata_selected_years.iloc[nuts2_level_index]
        nuts1_df = df_cropdata_selected_years.iloc[nuts1_level_index]
        nuts0_df = df_cropdata_selected_years.iloc[nuts0_level_index]
        for crop in all_crops:
            consistency_dict["country"].append(country)
            consistency_dict["year"].append(year)
            consistency_dict["crop"].append(crop)
            nuts0_crop_area = nuts0_df.iloc[
                np.where(nuts0_df["crop"] == crop)[0]
            ]["area"].sum()
            consistency_dict["0level_value"].append(nuts0_crop_area)
            nuts3_crop_area = nuts3_df.iloc[
                np.where(nuts3_df["crop"] == crop)[0]
            ]["area"].sum()
            if nuts3_crop_area > 0:
                consistency_dict["3level_value"].append(nuts3_crop_area)
                consistency_dict["03_consistency"].append(
                    float(nuts0_crop_area - nuts3_crop_area)
                )
            else:
                consistency_dict["3level_value"].append(np.nan)
                consistency_dict["03_consistency"].append(np.nan)
            nuts2_crop_area = nuts2_df.iloc[
                np.where(nuts2_df["crop"] == crop)[0]
            ]["area"].sum()
            if nuts2_crop_area > 0:
                consistency_dict["2level_value"].append(nuts2_crop_area)
                consistency_dict["02_consistency"].append(
                    float(nuts0_crop_area - nuts2_crop_area)
                )
            else:
                consistency_dict["2level_value"].append(np.nan)
                consistency_dict["02_consistency"].append(np.nan)
            nuts1_crop_area = nuts1_df.iloc[
                np.where(nuts1_df["crop"] == crop)[0]
            ]["area"].sum()
            if nuts1_crop_area > 0:
                consistency_dict["1level_value"].append(nuts1_crop_area)
                consistency_dict["01_consistency"].append(
                    float(nuts0_crop_area - nuts1_crop_area)
                )
            else:
                consistency_dict["1level_value"].append(np.nan)
                consistency_dict["01_consistency"].append(np.nan)

        

crop_consistency_df = pd.DataFrame(consistency_dict)

#write 0 or 1 depending on whether any level contains new information
#%%
crop_consistency_df.fillna(0,inplace=True)
nuts0_level_required=np.zeros(len(crop_consistency_df))
abs_difference_01=np.abs(crop_consistency_df["0level_value"]-crop_consistency_df["1level_value"])
rel_difference_01=abs_difference_01/crop_consistency_df["0level_value"]
#nuts0_level_required[np.where((rel_difference_01>0.01)|(abs_difference_01>0.005))[0]]=1
nuts0_level_required[np.where(abs_difference_01>0.0001)[0]]=1

nuts1_level_required=np.zeros(len(crop_consistency_df))
abs_difference_12=np.abs(crop_consistency_df["1level_value"]-crop_consistency_df["2level_value"])
rel_difference_12=abs_difference_12/crop_consistency_df["1level_value"]
#nuts1_level_required[np.where((rel_difference_12>0.01)|(abs_difference_12>0.005))[0]]=1
nuts1_level_required[np.where(abs_difference_12>0.0001)[0]]=1

nuts2_level_required=np.zeros(len(crop_consistency_df))
abs_difference_23=np.abs(crop_consistency_df["2level_value"]-crop_consistency_df["3level_value"])
rel_difference_23=abs_difference_23/crop_consistency_df["2level_value"]
#nuts2_level_required[np.where((rel_difference_23>0.01)|(abs_difference_23>0.005))[0]]=1
nuts2_level_required[np.where(abs_difference_23>0.0001)[0]]=1

nuts3_level_required=np.zeros(len(crop_consistency_df))
nuts3_level_required[np.where(crop_consistency_df["3level_value"]>0)[0]]=1

crop_consistency_df["NUTS0"]=nuts0_level_required
crop_consistency_df["NUTS1"]=nuts1_level_required
crop_consistency_df["NUTS2"]=nuts2_level_required
crop_consistency_df["NUTS3"]=nuts3_level_required
#%%
crop_consistency_df["3level_information_share"]=crop_consistency_df["3level_value"]/crop_consistency_df["0level_value"]
crop_consistency_df["2level_information_share"]=(crop_consistency_df["2level_value"]-crop_consistency_df["3level_value"])/crop_consistency_df["0level_value"]
crop_consistency_df["1level_information_share"]=(crop_consistency_df["1level_value"]-crop_consistency_df["2level_value"])/crop_consistency_df["0level_value"]
crop_consistency_df["0level_information_share"]=(crop_consistency_df["0level_value"]-crop_consistency_df["1level_value"])/crop_consistency_df["0level_value"]

#%%
crop_consistency_df

#%%
crop_consistency_df.to_csv(output_path+"crop_levels_selected_countries_"+str(selected_years.min())+str(selected_years.max())+".csv")
# %%

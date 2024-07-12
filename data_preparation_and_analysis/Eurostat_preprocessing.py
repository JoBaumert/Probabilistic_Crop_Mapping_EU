#%%


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
from pathlib import Path
import os

sys.path.append(
    "/home/baumert/research/Project-1/Project-1-Code/data preparation/modules/"
)
import functions_for_data_preparation as ffd

#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]


#intermediary_data_path = "/home/baumert/research/Project-1/data/Intermediary Data/"
delineation_and_parameter_path = (
    data_main_path+"delineation_and_parameters/"
)
raw_data_path=data_main_path+"Raw_Data/"
#intermediary_data_path=data_main_path+"Intermediary_Data/Preprocessed_Inputs/Eurostat/"
intermediary_data_path=data_main_path+"Intermediary_Data/Preprocessed_Inputs/"
# input files
parameter_path = delineation_and_parameter_path + "DGPCM_user_parameters.xlsx"
crop_delineation_path = delineation_and_parameter_path + "DGPCM_crop_delineation.xlsx"
nuts_path = intermediary_data_path+"NUTS/NUTS_all_regions_all_years.csv"
main_area_path = raw_data_path + "Eurostat/apro_cpshr_20102020_main_area.csv"
area_path = raw_data_path + "Eurostat/apro_cpshr_20102020_area.csv"
data_2010_nuts3_path = (
    raw_data_path + "Eurostat/EUROSTAT_crops_total_NUTS3_2010_final.xlsx"
)
UAA_path = raw_data_path + "Eurostat/UAA_all_regions_all_years.csv"

# output files

cropdata_output_path = (
    intermediary_data_path + "Eurostat/Eurostat_cropdata_compiled_"
)
UAA_output_path = intermediary_data_path + "Eurostat/Eurostat_UAA_compiled_"
crop_consistency_overview_path = (
    data_main_path+"Intermediary_Data/Regional_Aggregates/crop_consistency_overview_"
)


# %%
# ===============================================================================================================================================

# import parameters
nuts_info = pd.read_excel(parameter_path, sheet_name="NUTS")
selected_years=np.array(pd.read_excel(parameter_path, sheet_name="selected_years")["years"])
#all_years = np.array(nuts_info["crop_map_year"])
nuts_years = np.sort(nuts_info[nuts_info["crop_map_year"].isin(selected_years)]["nuts_year"].value_counts().keys())
relevant_nuts_years = np.array(nuts_info[nuts_info["crop_map_year"].isin(selected_years)]["nuts_year"])
crop_info = pd.read_excel(parameter_path, sheet_name="crops")
all_crops = np.array(crop_info["crop"])


# import information which countries are considered
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])

#import NUTS information
nuts_allyear_df=pd.read_csv(nuts_path)

# ===============================================================================================================================================
# %%
"""
1) COMPILATION OF EUROSTAT CROP INFORMATION FOR YEARS 2010 - 2020
for all NUTS regions that appear in a year in nuts_allyear_df, 
get the crop information that exists for this respective region.
The data comes from 3 files:
    - main area 2010 - 2020 (NUTS0, NUTS1, and sometimes NUTS2 levels)
    - area 2010 -2020 (NUTS0, NUTS1, and sometimes NUTS2 levels)
    - crop area at NUTS3 level for the year 2010, if available
"""
main_area = pd.read_csv(main_area_path)
area = pd.read_csv(area_path)
if 2010 in selected_years:
    try:
        data_2010_nuts3_raw = pd.read_excel(data_2010_nuts3_path, sheet_name="Areas")
    except:
        data_2010_nuts3_raw=None
else:
    data_2010_nuts3_raw=None
crop_delineation = pd.read_excel(crop_delineation_path)

#%%
""" the data for main area and area is not given as floats, therefore transform it to float values """
str_array_mainarea = np.array(
    [str(value).replace(",", "") for value in main_area["OBS_VALUE"]]
)
float_array_mainarea = (
    np.where(
        (str_array_mainarea == "nan") | (str_array_mainarea == ":"),
        "-999",
        str_array_mainarea,
    )
).astype(float)
main_area["float_value"] = float_array_mainarea

str_array_area = np.array([str(value).replace(",", "") for value in area["OBS_VALUE"]])
float_array_area = (
    np.where(
        (str_array_area == "nan") | (str_array_area == ":"), "-999", str_array_area
    )
).astype(float)
area["float_value"] = float_array_area

#%%
"""link eurostat codes with the codes used in DGPCM"""

crop_conversion_eurostat, crop_conversion_DGPCM = [], []

for c1, code1 in enumerate(np.unique(np.array(crop_delineation["Eurostat_code"]))):
    crop_conversion_eurostat.append(code1)
    crop_conversion_DGPCM.append(
        crop_delineation[crop_delineation["Eurostat_code"] == code1]["DGPCM_code"]
        .value_counts()
        .keys()[0]
    )

for c2, code2 in enumerate(
    np.unique(np.array(crop_delineation["Eurostat_code2"]).astype(str))
):
    if code2 == "nan":
        pass
    else:
        crop_conversion_eurostat.append(code2)
        crop_conversion_DGPCM.append(
            crop_delineation[crop_delineation["Eurostat_code2"] == code2][
                "DGPCM_code"
            ]
            .value_counts()
            .keys()[0]
        )

crop_conversion_eurostat, crop_conversion_DGPCM = np.array(
    crop_conversion_eurostat
), np.array(crop_conversion_DGPCM)
#%%

area.rename(columns={"crops": "CROPS"}, inplace=True)
main_area.rename(columns={"crops": "CROPS"}, inplace=True)
#%%
nuts_allyear_df
#%%
#get relevant crop area information from Eurostat input files for each NUTS region in every year
df_all_aggregates = pd.DataFrame()
for year in selected_years:
    print(year)
    all_nuts = np.array(
        nuts_allyear_df["NUTS_ID"].iloc[np.where(nuts_allyear_df["year"] == year)]
    )
    for nuts in all_nuts:
        main_area_selected = main_area.iloc[
            np.where((main_area["geo"] == nuts) & (main_area["TIME_PERIOD"] == year))[0]
        ]
        area_selected = area.iloc[
            np.where((area["geo"] == nuts) & (area["TIME_PERIOD"] == year))[0]
        ]
        crop_array, value_array = ffd.get_relevant_croparea(
            main_area_selected, area_selected, crop_delineation
        )
        df_raw = pd.DataFrame({"crop": crop_array, "area": value_array})
        a = np.array(df_raw["crop"])
        sorter = np.argsort(crop_conversion_eurostat)
        df_raw["DGPCM_crop_code"] = crop_conversion_DGPCM[
            sorter[np.searchsorted(crop_conversion_eurostat, a, sorter=sorter)]
        ]
        df = (
            df_raw[["DGPCM_crop_code", "area"]]
            .groupby("DGPCM_crop_code")
            .sum()
            .reset_index()
        )

        df["NUTS_ID"] = np.repeat(nuts, len(df))
        df["year"] = np.repeat(year, len(df))
        df = df[["NUTS_ID", "year", "DGPCM_crop_code", "area"]]
        df_all_aggregates = pd.concat((df_all_aggregates, df))

#%%
data_2010_nuts3_raw
#%%
a=None
if type(a)==pd.DataFrame:
    print("yxa")
#%%
if data_2010_nuts3_raw:
    print("fds")
#%%
"""
if 2010 is among the selected years and if the NUTS3 level info is available,
import and preprocess information on crop production at NUTS3 level for 2010"""
if (2010 in selected_years)&(type(data_2010_nuts3_raw)==pd.DataFrame):
    data_2010_nuts3 = data_2010_nuts3_raw.iloc[
        2:, 8:
    ]  # get only rows and columns that contain information
    crop_array = np.array([i.split(":")[1] for i in data_2010_nuts3.iloc[0]])
    data_2010_nuts3.columns = crop_array
    data_2010_nuts3.drop([2], inplace=True)
    data_2010_nuts3["NUTS_ID"] = data_2010_nuts3_raw.iloc[:, 3]
    # reshape df so that all crops and nuts ids appear in two columns
    data_2010_nuts3_melted = data_2010_nuts3.melt(
        id_vars="NUTS_ID", value_vars=np.array(data_2010_nuts3.columns)[:-1]
    )
    # merge data with information on crop delineation to link EUrostat 2010 with B04 codes
    data_2010_nuts3_melted = pd.merge(
        data_2010_nuts3_melted,
        crop_delineation[["Eurostat_name_2010", "DGPCM_code"]].drop_duplicates(),
        how="left",
        left_on="variable",
        right_on="Eurostat_name_2010",
    )
    data_2010_nuts3_melted = data_2010_nuts3_melted[["NUTS_ID", "DGPCM_code", "value"]]

    data_2010_nuts3_melted.dropna(subset=["DGPCM_code"], inplace=True)
    # area is in ha --> needs to be in 1000ha to be comparable to other Eurostat data
    data_2010_nuts3_melted["area"] = data_2010_nuts3_melted["value"] / 1000
    data_2010_nuts3_melted.drop(columns="value", inplace=True)
    data_2010_nuts3_melted = data_2010_nuts3_melted[
        data_2010_nuts3_melted["NUTS_ID"] != "Total"
    ]
    data_2010_nuts3_melted["year"] = np.repeat(2010, len(data_2010_nuts3_melted))
    data_2010_nuts3_melted.rename(columns={"DGPCM_code": "DGPCM_crop_code"}, inplace=True)
    # for some crops we have more than one value per year and region (if they were listed as more than one crop in eurostat). Aggregate those values
    data_2010_nuts3_melted = (
        data_2010_nuts3_melted.groupby(["NUTS_ID", "year", "DGPCM_crop_code"])
        .sum()
        .reset_index()
)
    """concat the two sources of information on crop production in the EU and keep only those countries that are part of the EU-28"""
    # before concatenation, remove those observations in "df_all_aggregates" for 2010 that already appear in the NUTS3 dataset
    df_all_aggregates_2010 = df_all_aggregates[
        (df_all_aggregates["year"] == 2010)
        & (
            ~df_all_aggregates["NUTS_ID"].isin(
                np.sort(data_2010_nuts3_melted["NUTS_ID"].value_counts().keys())
            )
        )
    ]
    df_all_aggregates = df_all_aggregates[df_all_aggregates["year"] != 2010]
    df_all_aggregates = pd.concat((df_all_aggregates, df_all_aggregates_2010))
    df_cropdata_selected_years = pd.concat(
        (
            df_all_aggregates,
            data_2010_nuts3_melted[["NUTS_ID", "year", "DGPCM_crop_code", "area"]],
        )
    )
else:
    df_cropdata_selected_years=df_all_aggregates

#%%
df_cropdata_selected_years.sort_values(by=["NUTS_ID", "year", "DGPCM_crop_code"], inplace=True)
df_cropdata_selected_years["country"] = np.array(df_cropdata_selected_years["NUTS_ID"]).astype("U2")


df_cropdata_selected_years = df_cropdata_selected_years[
    ["country", "NUTS_ID", "year", "DGPCM_crop_code", "area"]
].iloc[
    np.where(
        np.isin(df_cropdata_selected_years["country"], country_codes_relevant).astype(int)
    )[0]
]

#%%
# export the compiled Eurostat cropdata
Path(intermediary_data_path).mkdir(parents=True, exist_ok=True)
df_cropdata_selected_years.to_csv(
    cropdata_output_path+str(selected_years[0])+str(selected_years[-1])+"_DGPCMcodes.csv",
    index=False,
)

# ===============================================================================================================================================
#%%
"""
2) GET INFORMATION ON THE UTILIZED AGRICULTURAL AREA FOR EACH REGION IN EACH YEAR
as for the aggregated crop information, we have information on UAA from more than 1 input files.
We use the information on the UAA that is contained in the main_area file (loaded above) and 
in a subsequent step combine it with the information that we have at NUTS3 level for 2010
"""
UAA = pd.read_csv(UAA_path)

#%%
# get the UAA for each NUTS region in each year to calculate how much area can be distributed to crops that are
# listed only at a higher level of aggregation
all_UAA = {"NUTS_ID": [], "year": [], "UAA": []}
for year in selected_years:
    print(year)
    all_nuts = np.array(
        nuts_allyear_df["NUTS_ID"].iloc[np.where(nuts_allyear_df["year"] == year)]
    )
    for nuts in all_nuts:
        all_UAA["NUTS_ID"].append(nuts)
        all_UAA["year"].append(year)
        try:
            UAA = float(
                main_area[
                    (main_area["TIME_PERIOD"] == year)
                    & (main_area["geo"] == nuts)
                    & (main_area["CROPS"] == "UAA")
                ]["float_value"].iloc[0]
            )

        except:
            UAA = np.nan
        all_UAA["UAA"].append(UAA)

all_UAA_df = pd.DataFrame(all_UAA)

#%%
""" if 2010 is among the selected years and the NUTS3 level information available,
 compile the information at NUTS3 level for 2010
"""
if (2010 in selected_years)&(type(data_2010_nuts3_raw)==pd.DataFrame):
    UAA_2010_nuts3 = pd.DataFrame(
        {
            "NUTS_ID": data_2010_nuts3_raw.iloc[3:, 3],
            "UAA": np.array(data_2010_nuts3_raw.iloc[3:, 6]) / 1000,
        }
    )

    UAA_2010_nuts3 = UAA_2010_nuts3[UAA_2010_nuts3["NUTS_ID"] != "Total"]
    UAA_2010_nuts3["year"] = np.repeat(2010, len(UAA_2010_nuts3))

    # connect the two sources of information
    df_UAA_selected_years = pd.concat((all_UAA_df, UAA_2010_nuts3))
else:
    df_UAA_selected_years=all_UAA_df
#%%
df_UAA_selected_years.sort_values(by=["NUTS_ID", "year"], inplace=True)
df_UAA_selected_years["country"] = np.array(df_UAA_selected_years["NUTS_ID"]).astype("U2")
df_UAA_selected_years = df_UAA_selected_years[["country", "NUTS_ID", "year", "UAA"]].iloc[
    np.where(np.isin(df_UAA_selected_years["country"], country_codes_relevant).astype(int))[0]
]
#%%
# export data on UAA
Path(intermediary_data_path).mkdir(parents=True, exist_ok=True)
df_UAA_selected_years.to_csv(UAA_output_path+str(selected_years[0])+str(selected_years[-1])+".csv", index=False)

#%%

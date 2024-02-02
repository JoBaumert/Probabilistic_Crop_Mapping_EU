#%%
#from re import A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# %%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
#%%
print("importing data...")
LUCAS_path=data_main_path+"Raw_Data/LUCAS/"
output_path=data_main_path+"Intermediary_Data/Preprocessed_Inputs/LUCAS/"
LUCAS_raw=pd.read_csv(LUCAS_path+"/lucas_harmo_uf.csv")
LUCAS_topsoil=pd.read_excel(LUCAS_path+"LUCAS_TOPSOIL_v1.xlsx")

# %%
if __name__ == "__main__":
    """
    not all of the LUCAS observations are agricultural land. Select only those that are cropland 
    (letter group B) or grassland (letter group E) and land use agriculture (U111) or land use fallow land (U112)
    for a definition of the groups see "LUCAS technical reference document C3" (https://ec.europa.eu/eurostat/documents/205002/8072634/LUCAS2018-C3-Classification.pdf) 
    """

    LUCAS_agri=LUCAS_raw[((LUCAS_raw['letter_group']=='B')|(LUCAS_raw['letter_group']=='E'))& \
        ((LUCAS_raw['lu1']=='U111')|(LUCAS_raw['lu1']=='U112'))]
        

    #%%
    """
    for some LUCAS points soil information was sampled as well and is stored in another
    file (LUCAS Topsoil). We merge this information here on the id of the respective LUCAS point
    """

    LUCAS_topsoil.rename(columns={'POINT_ID':'point_id'},inplace=True)
    LUCAS_agri_merged=LUCAS_agri.merge(LUCAS_topsoil, how='left', on='point_id')

    #%%
    """
    the geographical coordinates of a few LUCAS points (<10) are invalid (e.g., they are in the range of 10000000). We remove these few observations
    """
    legit_long_min, legit_long_max, legit_lat_min, legit_lat_max = -180, 180, -90, 90

    LUCAS_preprocessed_end=LUCAS_agri_merged[(LUCAS_agri_merged['th_long']>legit_long_min)&
                            (LUCAS_agri_merged['th_long']<legit_long_max)&
                            (LUCAS_agri_merged['th_lat']>legit_lat_min)&
                            (LUCAS_agri_merged['th_lat']<legit_lat_max)]

    #%%
    Path(output_path).mkdir(parents=True, exist_ok=True)
    LUCAS_preprocessed_end.to_csv(output_path+"LUCAS_preprocessed.csv", index=False, header=True)
    print("successfully completed task")
    # %%

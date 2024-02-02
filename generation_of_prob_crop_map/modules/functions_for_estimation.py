#%%

from audioop import mul
from cgi import test
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import bambi as bmb
import math
import arviz as az
from pyparsing import col, re 
from scipy.stats import norm
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import poisson_binomial
import xarray
import aesara.tensor as at
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import metrics
from joblib import dump,load
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from pathlib import Path
# %%
def cropname_conversion_func(conversion_path):
    cropname_conversion=pd.read_excel(conversion_path)
    cropname_conversion=cropname_conversion[['LUCAS_code','DGPCM_code']]
    cropname_conversion.drop_duplicates(inplace=True)
    cropname_conversion_dict= { cropname_conversion.iloc[i,0] : cropname_conversion.iloc[i,1] for i in range(0, len(cropname_conversion) ) }
    return cropname_conversion_dict


def get_explanatory_data(rel_countries,sel_features,data_path,conversion_path,input_file_date):
    """ this function returns a DF with the first column being the crop (CAPRI_name) and all other columns being the explanatory variables"""
    countrydata=[]
    for country in rel_countries:
        countrydata.append(pd.read_csv(data_path+country+"/"+country+"_"+input_file_date+".csv"))
    relcountries_LUCAS=pd.concat(countrydata)
    #import cropname conversion information and convert data
    cropname_conversion=pd.read_excel(conversion_path)
    cropname_conversion=cropname_conversion[['LUCAS_code','CAPRI_code']]
    cropname_conversion.drop_duplicates(inplace=True)
    cropname_conversion_dict= { cropname_conversion.iloc[i,0] : cropname_conversion.iloc[i,1] for i in range(0, len(cropname_conversion) ) }

    relcountries_LUCAS=relcountries_LUCAS[relcountries_LUCAS['lc1'].isin(list(cropname_conversion_dict.keys()))]
    relcountries_LUCAS['CAPRI_name']=relcountries_LUCAS['lc1'].apply(lambda x: cropname_conversion_dict[x])

    use_cols=['CAPRI_name']+sel_features
    regression_df=relcountries_LUCAS[use_cols]
    regression_df.dropna(inplace=True)
    return regression_df
    

def scale_data(df,scaling_path,output_file_name,scaler_available=False,export_txt=True,scaler_type='standard'):
    """this function is used to scale data and save the scaling factors for reuse"""
    data=np.array(df)
    Path(scaling_path).mkdir(parents=True, exist_ok=True)
    if scaler_available:
        scaler=load(scaling_path+'/'+scaler_type+'scaler_'+output_file_name)
    else:
        if scaler_type=='standard':
            scaler=StandardScaler()
        scaler.fit(data)
        dump(scaler,scaling_path+'/'+scaler_type+'scaler_'+output_file_name,compress=True)
        #as default export file that contains information on the choice and order of features
        if export_txt:
            columns=np.array(df.columns)
            with open(scaling_path+'/'+scaler_type+'scaler_'+output_file_name+'_columns.txt',"w") as t:
                for c in columns:
                    t.write(str(c))
                    t.write("\n")
    data_scaled=scaler.transform(data)
    return data_scaled

def aggregate_crops(input_df,change_col_name,CAPRI_names_tuples):
    #CAPRI_names_tuples must be a list or array of tuples of those crops that belong together
    #for t in CAPRI_names_tuples:
    for tup in CAPRI_names_tuples:
        new_name="+".join(tup)
        input_df[change_col_name].replace(tup,new_name,inplace=True)
    output_df=input_df
    return output_df

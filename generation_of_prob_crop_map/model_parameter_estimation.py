#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
import statsmodels.api as sm
from pathlib import Path



sys.path.append(
    str(Path(Path(os.path.abspath(__file__)).parents[0]))+"/modules/"
)
print(str(Path(Path(os.path.abspath(__file__)).parents[0]))+"/modules/")
import modules.functions_for_estimation as fufore
#%%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]


#%%
min_threshold=20 #if less than min_threshold lucas observations are made for a crop, discard this crop
raw_data_path=data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
delineation_and_parameter_path = (
    data_main_path+"delineation_and_parameters/"
)
results_path=data_main_path+"Results/Model_Parameter_Estimates/"

#%%
parameter_path=delineation_and_parameter_path+"DGPCM_user_parameters.xlsx"
LUCAS_feature_path=intermediary_data_path+"LUCAS_feature_merges/"
LUCAS_preprocessed_path=intermediary_data_path+"Preprocessed_Inputs/LUCAS/LUCAS_preprocessed.csv"
cropname_conversion_path=delineation_and_parameter_path+"DGPCM_crop_delineation.xlsx"
scaling_path=results_path+"/scale_factors/"
#%%
sel_features=['avg_annual_temp_sum',
 'avg_annual_veg_period',
 'slope',
 'sand',
 'elevation',
 'latitude4326',
 'silt',
 'bulk_density',
 'clay',
 'awc',
 'avg_annual_precipitation',
 'coarse_fragments']


column_names={'avg_annual_temp_sum':['tempsum_annual_mean_030405','tempsum_annual_mean_060708','tempsum_annual_mean_091011','tempsum_annual_mean_121314','tempsum_annual_mean_151617'],
 'avg_annual_veg_period':['vegperiod_annual_mean_030405', 'vegperiod_annual_mean_060708','vegperiod_annual_mean_091011', 'vegperiod_annual_mean_121314','vegperiod_annual_mean_151617'],
 'slope':['slope_in_degree'],
 'sand':['sand_content'],
 'elevation':['elevation'],
 'latitude4326':['latitude4326'],
 'silt':['silt_content'],
 'bulk_density':['bulk_density'],
 'clay':['clay_content'],
 'awc':['awc'],
 'avg_annual_precipitation':['precipitation_annual_mean_030405','precipitation_annual_mean_060708', 'precipitation_annual_mean_091011','precipitation_annual_mean_121314', 'precipitation_annual_mean_151617'],
 'coarse_fragments':['coarse_fragments'],
 'organic_carbon':['oc_content']}

year_climateyear_link={
    2006:'030405',
    2009:'060708',
    2012:'091011',
    2015:'121314',
    2018:'151617'
}
#%%

countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])


index_columns=np.array(['year','id','point_id','lc1'])
selected_columns=np.concatenate((index_columns,np.array([item for sublist in [column_names[feature] for feature in sel_features] for item in sublist])))
LUCAS_preprocessed=pd.read_csv(LUCAS_preprocessed_path)
#%%
for country in country_codes_relevant:
    LUCAS_selected=LUCAS_preprocessed[LUCAS_preprocessed['nuts0']==country]


    #%%
    feature_df=LUCAS_selected[['year','id','point_id','lc1']]

    for feature in sel_features:
        data=pd.read_csv(LUCAS_feature_path+country+"/"+feature+".csv")
        feature_df=pd.merge(feature_df,data,how='left',on=['year','id','point_id'])
    #%%
    feature_df
    #%%

    selected_feature_df=feature_df[selected_columns]
    years,index=np.unique(selected_feature_df['year'], return_inverse=True)

    # the challenge here is to attribute to each LUCAS obserbvation the correct climate year. We first create an array with the values -999 and then fill it according to the
    # respective year that is required
    avg_annual_temp_sum_array=np.repeat(-999,len(selected_feature_df))
    avg_annual_precipitation_array=np.repeat(-999,len(selected_feature_df))
    avg_annual_veg_period_array=np.repeat(-999,len(selected_feature_df))
    for y,year in enumerate(year_climateyear_link.keys()):
        year_pos=np.where(selected_feature_df['year']==year)
        avg_annual_temp_sum_array[year_pos]=np.array(selected_feature_df[column_names['avg_annual_temp_sum']]).transpose()[y][year_pos]
        avg_annual_precipitation_array[year_pos]=np.array(selected_feature_df[column_names['avg_annual_precipitation']]).transpose()[y][year_pos]
        avg_annual_veg_period_array[year_pos]=np.array(selected_feature_df[column_names['avg_annual_veg_period']]).transpose()[y][year_pos]

    selected_feature_df.drop(columns=column_names['avg_annual_temp_sum'],inplace=True)
    selected_feature_df.drop(columns=column_names['avg_annual_precipitation'],inplace=True)
    selected_feature_df.drop(columns=column_names['avg_annual_veg_period'],inplace=True)
    selected_feature_df['avg_annual_temp_sum']=avg_annual_temp_sum_array
    selected_feature_df['avg_annual_precipitation']=avg_annual_precipitation_array
    selected_feature_df['avg_annual_veg_period']=avg_annual_veg_period_array
    #change column names in column names dictionary
    column_names_corrected=column_names
    column_names_corrected['avg_annual_precipitation']=['avg_annual_precipitation']
    column_names_corrected['avg_annual_veg_period']=['avg_annual_veg_period']
    column_names_corrected['avg_annual_temp_sum']=['avg_annual_temp_sum']
    #%%
    #check 'by hand' if there are any obvious errors in the features:
    print('share Nan values in each column:')
    print(selected_feature_df.isna().sum()/len(selected_feature_df))

    print('Min values for each column:')
    print(selected_feature_df.min())

    print('Max values for each column:')
    print(selected_feature_df.max())
    

    #%%
    #if all nan values are discarded, how many observations would remain?
    dropna=selected_feature_df.dropna()
    share_discared=1-len(dropna[(dropna['avg_annual_temp_sum']>-1000)&(dropna['avg_annual_precipitation']>=0)&(dropna['avg_annual_veg_period']>=0)])/len(selected_feature_df)
    #print(f"do you want to discard {share_discared*100}% of the LUCAS observations? \n If this share is too large, consider dropping some features")
    print(f"{share_discared*100}% of the LUCAS observations are discarded due to missing data")

    #%%
    """DISCARD NAN VALUES and INVALID TEMP AVERAGES/PRECIPITATION"""
    selected_feature_df.dropna(inplace=True)
    selected_feature_df=selected_feature_df[(selected_feature_df['avg_annual_temp_sum']>-1000)&(selected_feature_df['avg_annual_precipitation']>=0)&(selected_feature_df['avg_annual_veg_period']>=0)]

    #%%
    """CONVERT CROPNAMES TO CAPRI NAMES"""
    cropname_conversion_dict=fufore.cropname_conversion_func(cropname_conversion_path)
    selected_feature_df=selected_feature_df[selected_feature_df['lc1'].isin(list(cropname_conversion_dict.keys()))]
    selected_feature_df['DGPCM_code']=selected_feature_df['lc1'].apply(lambda x: cropname_conversion_dict[x])
    
    #%%
    #LUCAS_selected[LUCAS_selected['lu1']=='U112']

    selected_features_final=np.array([item for sublist in [column_names_corrected[feature] for feature in sel_features] for item in sublist])
    #%%
    """SCALE DATA"""
    X=np.array(selected_feature_df[selected_features_final])
    y=np.array(selected_feature_df['DGPCM_code'])
    X_scaled=fufore.scale_data(selected_feature_df[selected_features_final],scaling_path,'multinom_logit_'+country)

    #%%
    regression_scaled_df=pd.DataFrame(X_scaled,columns=sel_features)
    regression_scaled_df['y']=y
    regression_scaled_df=fufore.aggregate_crops(regression_scaled_df,'y',[('APPL','OFRU'),('TOMA','OVEG')])
    regression_scaled_df.sort_values('y',inplace=True)
    #%%
    #display the share of crops in selected LUCAS sample
    obssum=regression_scaled_df['y'].value_counts().values.sum()
    for i,c in enumerate(regression_scaled_df['y'].value_counts().keys()):
        print(f"{c}: {regression_scaled_df['y'].value_counts().values[i]/obssum}")
    #%%
    """DISCARD CROPS FOR WHICH NOT SUFFICIENT OBSERVATIONS EXIST"""
    obs_frequency_cropnames=regression_scaled_df['y'].value_counts().keys()
    obs_frequency_freq=regression_scaled_df['y'].value_counts().values
    for c,crop in enumerate(obs_frequency_cropnames):
        print(f"{crop}: {obs_frequency_freq[c]}")

    #%%
    #if only less than min_threshold (default=20) observations --> discard crop
    considered_crops=obs_frequency_cropnames[np.where(obs_frequency_freq>=min_threshold)[0]]

    regression_scaled_df=regression_scaled_df[regression_scaled_df['y'].isin(considered_crops)]

    regression_scaled_df.sort_values(by='y',inplace=True)

    #add a constant 
    regression_scaled_df=sm.add_constant(regression_scaled_df)
    sel_features.insert(0,'const')


    #%%
    print("logistic regression starts...")
    """RUN LOGISTIC REGRESSION"""
    logit_model=sm.MNLogit(regression_scaled_df['y'],regression_scaled_df[sel_features])
    result=logit_model.fit(method='lbfgs',maxiter=5000)
    #%%
    multinom_params=result.params
    multinom_covm=result.cov_params().reset_index()
    #%%

    multinom_params.columns=sorted(regression_scaled_df['y'].value_counts().keys())[1:]

    #%%
    """EXPORT DATA  """
    print("export results...")
    multinom_params.to_excel(results_path+"multinomial_logit_"+country+"_statsmodel_params_obsthreshold"+str(min_threshold)+".xlsx")
    multinom_covm.to_excel(results_path+"multinomial_logit_"+country+"_statsmodel_covariance_obsthreshold"+str(min_threshold)+".xlsx")

# %%

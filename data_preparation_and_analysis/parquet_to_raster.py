#%%
import rasterio as rio
from rasterio import features
from rasterio.plot import show
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
from rasterio.windows import from_bounds
import geopandas as gpd
import pandas as pd
from pathlib import Path
import os
import numpy as np
import zipfile
import xarray
import gc
import matplotlib.pyplot as plt
import arviz as az
# %%
data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]
postsampling_reps = 10 
# %%

#%%
#Posterior_probability_path=(data_main_path+"Results/Posterior_crop_probability_estimates/")
Posterior_probability_path="/home/baumert/fdiexchange/baumert/posterior_probabilities/"
parameter_path = (
    data_main_path+"delineation_and_parameters/DGPCM_user_parameters.xlsx"
)
raw_data_path = data_main_path+"Raw_Data/"
intermediary_data_path=data_main_path+"Intermediary_Data/"
grid_1km_path=raw_data_path+"Grid/"
n_of_fields_path=intermediary_data_path+"Zonal_Stats/"
#%%
Simulated_cropshares_path=(data_main_path+"Results/Simulated_consistent_crop_shares/")
# %%
#import parameters
countries = pd.read_excel(parameter_path, sheet_name="selected_countries")
country_codes_relevant = np.array(countries["country_code"])
nuts_info = pd.read_excel(parameter_path, sheet_name="NUTS")
all_years = np.array(nuts_info["crop_map_year"])
# %%
def generate_random_results(p_matrix,postsampling_reps,n_of_fields_array):
    p_matrix_corrected = np.array(
        [p_vector / (p_vector.sum() + 0.0001) for p_vector in p_matrix]
    )
    p_matrix_corrected = np.where(
        p_matrix_corrected == 0, 0.000001, p_matrix_corrected
    )
    """for some cells the sums of all crop probabilities are marginally larger than 1 (like 1.0001). 
    Normalize them to ensure that all probabilities within a cell add up to 1 (or marginally less), 
    which is necessary for the random sampling
    """

    random_results = np.array(
        [
            np.random.multinomial(
                n_of_fields_array[i], p_matrix_corrected[i], postsampling_reps
            )
            / n_of_fields_array[i]
            for i in range(p_matrix_corrected.shape[0])
        ]
    )
    del p_matrix_corrected
    gc.collect()
    return random_results


if __name__ == "__main__":
    for country in country_codes_relevant:

        n_of_fields = pd.read_csv(n_of_fields_path+country+"/n_of_fields/n_of_fields_allcountry.csv")
        grid_1km_path_country = (
        # "zip+file://"
            grid_1km_path
        + country
        +"_1km.zip"     
        )
        zip_obj=zipfile.ZipFile(grid_1km_path_country)
        for file in zip_obj.namelist():

            if (file[-3:]=="shp")&(file[3:6]=="1km"):
 
                break

        grid_1km_country = gpd.read_file(grid_1km_path_country+"!/"+file)

        for year in all_years:
            print("---"+country+"--"+str(year)+"---")
            posterior_probas=pd.read_parquet(Posterior_probability_path+country+"/"
                                            +country+str(year)+"entire_country")
            print("successfully imported posterior probabilities...")
            #some cells appear more than beta*n_crops (280) times because they are at the border of different nuts region
            #sort DF so that for each cell, crop and beta the one with the largest weight is on top (i.e., the largest cell)
            posterior_probas.sort_values(["CELLCODE","crop","beta","weight"],ascending=[True,True,True,False],inplace=True)
            #now drop all duplicate cells so that only the probabilities for each crop and beta with the largest weight (i.e., the largest) remain
            posterior_probas.drop_duplicates(["CELLCODE","crop","beta"],keep="first",inplace=True)
            #in the resulting df each cell appears 280 times:
            #posterior_probas.CELLCODE.value_counts()

            posterior_probas=pd.merge(posterior_probas,grid_1km_country,how="left",on="CELLCODE")

            posterior_probas=gpd.GeoDataFrame(posterior_probas)


            n_of_fields_relevant=n_of_fields[n_of_fields["year"]==year]
            n_of_fields_relevant.drop_duplicates(["CELLCODE"],inplace=True)

            posterior_probas=pd.merge(posterior_probas,n_of_fields_relevant[["CELLCODE","n_of_fields_assumed"]],how="left",on="CELLCODE")


            emax=posterior_probas["EOFORIGIN"].max()
            emin=posterior_probas["EOFORIGIN"].min()
            nmax=posterior_probas["NOFORIGIN"].max()
            nmin=posterior_probas["NOFORIGIN"].min()
            width=(emax-emin)/1000+1
            height=(nmax-nmin)/1000+1

            ul = (emin, nmax+1000)  # in lon, lat / x, y order
            ll = (emin,nmin)
            ur = (emax+1000,nmax+1000)
            lr = (emax+1000,nmin)


            gcps = [
                GCP(0, 0, *ul),
                GCP(0, width, *ur),
                GCP(height, 0, *ll),
                GCP(height, width, *lr)
            ]

            transform = from_gcps(gcps)

            crops=np.unique(posterior_probas["crop"])
            betas=np.unique(posterior_probas["beta"])

            band_list=[]
            #add cellweight as first band
            posterior_probas.sort_values(by=["NOFORIGIN","EOFORIGIN"],ascending=[False,True],inplace=True)
            resulting_matrix=np.ndarray((crops.shape[0]*postsampling_reps*betas.shape[0]+crops.shape[0]+2,int(height),int(width)))
            print("rasterizing cell weight...")
            selection=posterior_probas.drop_duplicates(["CELLCODE"])
            geom_value = ((geom,value) for geom, value in zip(selection.geometry, selection.weight))
            rasterized=features.rasterize(
                geom_value,
                out_shape=(int(height),int(width)),
                transform=transform,
                default_value=1 
            )
            #first band is the cellweight
            resulting_matrix[0]=rasterized
            band_list.append("weight")
            #add number of fields per cell as second band
            print("rasterizing number of fields...")
            geom_value = ((geom,value) for geom, value in zip(selection.geometry, selection.n_of_fields_assumed))
            rasterized=features.rasterize(
                geom_value,
                out_shape=(int(height),int(width)),
                transform=transform,
                default_value=1 
            )
            #second band is assumed number of fields per cell (minimum= 5)
            resulting_matrix[1]=rasterized
            #replace a few nan values with mean
            resulting_matrix[1][np.where(np.isnan(resulting_matrix[1]))]=np.nanmean(resulting_matrix[1])

            band_list.append("n_of_fields")
            n_of_fields_array=resulting_matrix[1,np.where(resulting_matrix[0]>0)[0],np.where(resulting_matrix[0]>0)[1]]

            for beta in betas:
                print("beta "+str(beta))
                helper_matrix=np.ndarray((crops.shape[0],int(height),int(width)))
                c=0
                for crop in crops:
                    print(crop)
                    selection=posterior_probas[(posterior_probas["crop"]==crop)&(posterior_probas["beta"]==beta)]
                    result_array=np.zeros(int(height*width),dtype=np.float16)
                    result_array[np.where(resulting_matrix[0].flatten()>0)[0]]=np.array(selection.posterior_probability,dtype=np.float16)
                    #add expected shares as bands 2:n_crops-3
                    helper_matrix[c]=result_array.reshape(resulting_matrix[0].shape)
                    c+=1
                    band_list.append(f"expected_share_{crop}")
                if beta==0:
                    resulting_matrix[2:len(crops)+2]=helper_matrix
        
                #sample from multinomial distribution with the respective crop probabilities and number of fields
                p_matrix=helper_matrix[:,np.where(resulting_matrix[0]>0)[0],np.where(resulting_matrix[0]>0)[1]].T
                print("generate "+str(postsampling_reps)+" random samples...")
                
                random_results=generate_random_results(p_matrix,postsampling_reps=postsampling_reps,n_of_fields_array=n_of_fields_array)
                
                random_results=generate_random_results(p_matrix,postsampling_reps=postsampling_reps,n_of_fields_array=n_of_fields_array)

                empty_array=np.zeros((int(width*height),postsampling_reps,len(crops)))
                empty_array[np.where(resulting_matrix[0].flatten()>0)[0]]=random_results
                
                resulting_matrix[2+len(crops)+beta*crops.shape[0]*postsampling_reps:(beta+1)*crops.shape[0]*postsampling_reps+2+len(crops)]=empty_array.transpose(1,2,0).reshape((postsampling_reps*len(crops),int(height),int(width)))
                del empty_array
                del p_matrix
                del helper_matrix
                gc.collect()
   

            factor=np.repeat(1000,resulting_matrix.shape[0])
            factor[0]=10
            factor[1]=1
          
            refactored_data=factor.reshape(-1,1,1)*resulting_matrix

            with rio.open(Simulated_cropshares_path+country+"/"+country+str(year)+"simulated_cropshare_"+str(postsampling_reps)+"reps_int.tif", 'w',
                        width=int(width),height=int(height),transform=transform,count=refactored_data.shape[0],dtype=rio.int16,crs="EPSG:3035") as dst:
                dst.write(refactored_data.astype(rio.int16))

            #export csv file with meta data (information about the bands)
            band_list=["weight","n_of_fields"]
            for crop in crops:
                band_list.append("expected_share_"+crop)
            for beta in betas:
                bands=list(np.char.add(
                        np.char.add(
                            np.char.add(
                            np.tile(crops,postsampling_reps).astype(str),
                            np.repeat("_",len(crops)*postsampling_reps)),
                        np.repeat(str(beta),len(crops)*postsampling_reps)),
                        np.repeat(np.arange(postsampling_reps),len(crops)).astype(str)))
                for band in bands:
                    band_list.append(band)

            
            band_df=pd.DataFrame({"band":np.arange(len(band_list)),"name":np.array(band_list)})
            
            band_df.to_csv(
                Simulated_cropshares_path+country+"/"+country+str(year)+"simulated_cropshare_"+str(postsampling_reps)+"reps_bands.csv"
            )
#%%
""""""


# %%

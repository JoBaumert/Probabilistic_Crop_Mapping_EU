#%%
import rasterio as rio
import rasterio.transform 
import geopandas as gpd
from pathlib import Path
import os
import ee
import geemap

# %%
#specify country
country="NL"

data_main_path=open(str(Path(Path(os.path.abspath(__file__)).parents[1])/"data_main_path.txt"))
data_main_path=data_main_path.read()[:-1]

ee.Authenticate()
ee.Initialize()
# %%
nuts_shapes=gpd.read_file(data_main_path+"Raw_Data/NUTS/NUTS_RG_01M_2016_3035.shp.zip!/NUTS_RG_01M_2016_3035.shp")
# %%

bounds=list(nuts_shapes[(nuts_shapes["CNTR_CODE"]==country)&(nuts_shapes["LEVL_CODE"]==0)].bounds.iloc[0])
#add 3km buffer on each side to avoid any errors that occur due to rounding issues in the geometry of the country
bounds=list(bounds+np.array([-3000,-3000,3000,3000]))
#%%

def get_elevation():
    elevation_image=ee.Image("USGS/GMTED2010_FULL").select("mea")
    return elevation_image.reproject(crs="epsg:3035",
                crsTransform=list(rasterio.transform.from_origin(bounds[0],bounds[3],25,25))[:6])

def get_slope():
    elevation_image=ee.Image("USGS/GMTED2010_FULL").select("mea")
    return ee.Terrain.slope(elevation_image).reproject(crs="epsg:3035",
                crsTransform=list(rasterio.transform.from_origin(bounds[0],bounds[3],25,25))[:6])
#%%
if __name__ == "__main__":

    geemap.ee_export_image_to_drive(
                        get_elevation(), 
                    #  folder= specify target folder here, otherwise will save to home,
                        description="eudem_dem_3035_"+country,    
                        scale=25, 
                        maxPixels=(2000*40)**2, #set to a large value, in this case will allow dowload for countries with size 2000x2000km
                        region=ee.Geometry.Rectangle(bounds,
                                                    proj="epsg:3035",evenOdd=False)
                    )
 
    geemap.ee_export_image_to_drive(
                        get_slope(), 
                    #  folder= specify target folder here, otherwise will save to home,
                        description="eudem_slope_3035_"+country,    
                        scale=25, 
                        maxPixels=(2000*40)**2, #set to a large value, in this case will allow dowload for countries with size 2000x2000km
                        region=ee.Geometry.Rectangle(bounds,
                                                    proj="epsg:3035",evenOdd=False)
                    )
# %%

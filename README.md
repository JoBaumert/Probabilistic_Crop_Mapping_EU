# Probabilistic_Crop_Mapping_EU
The code is seperated into two directories according to it's purpose: 1) The directory "data_preparation_and_analysis" contains multiple files that must be run first to preprocess the raw data in such a way that it can be used for the generation of the probabilistic crop maps. The analysis, validation and visualization of the modelling results is also performed with files in this directory. The other directory, "generation_of_prob_crop_map" contains the code for the generation of the maps, including parameter estimation. The folder "delineation_and_parameters" contains some rather small excel files that describe how crop types are matched between different data sources (e.g., LUCAS and Eurostat) and predefine some hyperparameters. The following is a guideline for the reproduction of the maps.

## Step 1: Preparation of Directories and Installation of Dependencies
First, create a directory in which you would like to store all the code. Then, copy the files provided in this repository to this directory e.g., by cloning the repository as follows (works on Linux):
```
git clone https://github.com/JoBaumert/Probabilistic_Crop_Mapping_EU.git
```
In any case it is crucial that the file structure is preserved. <br>
Second, install all dependencies listed in the requirements.txt file. We recommend to work in a virtual environment. For example, if you work in VSCode, you could proceed as follows: <br>
1) create a virtual environment (venv) like described here: https://code.visualstudio.com/docs/python/environments
2) install the dependencies listed in Probabilistic_Crop_Mapping_EU/requirements.txt

Second, create a folder named "Data" where the input and output data will be stored. The user can choose where to locate this directory on the local machine. However, when choosing the location of this directory consider that some input and output files are very large. All of the input data must be downloaded from their original sources, prior to running the code (see below). To ensure that the python scripts find all data, you must specify the path to the main data directory with a text file that is stored in the same directory as the code. For this you have (at least) two options: 
1. manually create a text file with the name "data_main_path.txt" that contains the path *to* the data folder (i.e., not the data folder itself).
2. use the command line to generate the text file with the respective content:
```
echo '/path/to/data/folder/' >data_main_path.txt
```
In either case, make sure that "data_main_path.txt" is stored in the same folder as the code's main directory.  <br>

Third, create a directory named "Raw_Data" within the directory named "Data". This is where all the raw data that you download will be stored. The results and intermediary output files will also be stored within folders in the "Data" directory, but those folders are generated automatically when running the scripts.



## Step 2: Download of input data
The input data used comes from multiple sources (see table below). To fully replicate our procedure, all of the listed input files are required. Download them from their source following the provided link. The column "file name" indicates how the downloaded files must be named by the user in order to be found by the python scripts. Curly braces {} indicate a variable that must be set by the user: for example, {2-digit-countrycode} indicates that this part of the filename is "FR" when considering France etc. Make sure to always use the same format of the file as indicated by the file name (e.g., csv or zip...).

### Sources of the raw data

|file name | link |note|save to| reference |
|----|----|----|----|----|
|**LUCAS data**|
|lucas_harmo_uf.csv|https://data.jrc.ec.europa.eu/dataset/f85907ae-d123-471f-a44a-8cca993485a2#dataaccess||Data/Raw_Data/LUCAS|[1]|
|LUCAS_TOPSOIL_v1.xlsx|https://esdac.jrc.ec.europa.eu/content/lucas-2009-topsoil-data||Data/Raw_Data/LUCAS|[2]|
|**Precipitation Data**|
|{2-digit-countrycode}\_precipitation_{StartyearEndyear}.csv|https://agri4cast.jrc.ec.europa.eu/DataPortal/RequestDataResource.aspx?idResource=7&o=d|select "sum of precipitation"<br> and respective<br> country, then<br> the start date<br> 01/01/2003<br> and the end<br> date 31/12/2020.<br> If the data<br> is too large <br>to download,<br> first download <br>the years 2001 <br>until some later<br> year and then<br> download in <br>another<br> file all the<br> years that remain.<br> Download as a<br> csv, unzip the<br> downloaded file <br>and save CSV <br>at the indicated<br> storage location|Data/Raw_Data/Precipitation/|[3]|
|**Temperature Data**|
|{2-digit-countrycode}\_temperature_{StartyearEndyear}.csv|https://agri4cast.jrc.ec.europa.eu/DataPortal/RequestDataResource.aspx?idResource=7&o=d|select <br>"mean air<br> temperature" and<br> then proceed<br> exactly as<br> for the <br>precipitation <br>data|Data/Raw_Data/Temperature|[3]|
|**Soil Data**|
|Sand_Extra (zip file)|https://esdac.jrc.ec.europa.eu/content/topsoil-physical-properties-europe-based-lucas-topsoil-data|the description <br>of the data<br> claims it's only <br>for the EU-25. <br>However, where you <br>can download the <br>data there is <br>the option to <br>download it<br> extrapolated for <br>the EU-28. <br>For all soil <br>variables, download <br>the extrapolated |Data/Raw_Data/Soil/|[4]|
|Silt_Extra (zip file)| " |"|"|"|
|Clay_Extra (zip file)| " |"|"|"|
|CoarseFragments_Extra (zip file)| " |"|"|"|
|BulkDensity_Extra (zip file)| " |"|"|"|
|AWC_Extra (zip file)| " |"|"|"|
|**Terrain Data**|
|eudem_dem_3035_europe.tif|[https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1-0-and-derived-products/eu-dem-v1.0?tab=metadata](https://sdi.eea.europa.eu/catalogue/eea/api/records/66fa7dca-8772-4a5d-9d56-2caba4ecd36a)|apparently no <br>longer maintained -<br> any other<br> elevation map<br> will do|Raw_Data/DEM/|[5]|
|eudem_slop_3035_europe.tif|https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1-0-and-derived-products/slope?tab=download|apparently no longer<br> maintained - any<br> other slope<br> map will do|Data/Raw_Data/DEM/|[5]|
|**Reference Grid**|
|{2-digit_countrycode}_1km (or 10km) (zip file)|[https://sdi.eea.europa.eu/data/d9d4684e-0a8d-496c-8be8-110f4b9465f6](https://www.eea.europa.eu/en/datahub/datahubitem-view/3c362237-daa4-45e2-8c16-aaadfb1a003b?activeAccordion=1069873%2C1159)|download 1km <br>(and 10km <br>reference grid <br>for those<br> countries for <br>which validation <br>is performed at <br>10km level),<br> i.e., one (two) <br>zip files per <br>country|Data/Raw_Data/Grid/|[6]|
|grid25.zip|https://agri4cast.jrc.ec.europa.eu/DataPortal/Index.aspx?o=d|to download, <br>in the agri4cast <br>data portal click <br>on the button<br> "Resource Info"<br> belonging to <br>"Gridded Agro-Meteorological <br>Data in Europe".<br> Then, click on<br> "Download file" <br>in the window<br> that opens,<br> next to "Grid<br> Definition".|Data/Raw_Data/Grid/|[6]|
|**NUTS regions boundaries**|
|NUTS_RG_01M_{year}_3035.shp.zip|https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts|select the <br>years 2006,<br> 2010, 2013,<br> 2016, scale <br>01M, file <br>format SHP, <br>geometry type <br>Polygons (RG) <br>and CRS <br>EPSG:3035|Data/Raw_Data/NUTS/|[7]|
|**Aggregated crop and UAA data**|
|apro_cpshr_20102020_area.csv|https://ec.europa.eu/eurostat/databrowser/view/apro_cpshr/default/table?lang=en|see additional<br> instructions below|Data/Raw_Data/Eurostat/|[8]|
|apro_cpshr_20102020_main_area.csv|https://ec.europa.eu/eurostat/databrowser/view/apro_cpshr/default/table?lang=en|see additional <br>instructions below|Data/Raw_Data/Eurostat/|[8]|
|EUROSTAT_crops_total_NUTS3_2010_final.xlsx||data was <br>provided by Eurostat<br> on request, <br>not publically <br>available for <br>download. <br>Only relevant<br> for year 2010|[8]|
|UAA_all_regions_all_years.csv|https://ec.europa.eu/eurostat/databrowser/view/apro_cpshr/default/table?lang=en|see additional <br>instructions below|Data/Raw_Data/Eurostat/|[8]|
|**Data for UAA at cell level**|
|clc{year}_v2020_20u1_raster100m (folder)|https://land.copernicus.eu/en/products/corine-land-cover|download for <br>2006, 2012, <br>and 2018.<br> Extract the <br>(nested) downloaded <br>files until you<br> get a folder<br> named clc{year}_v2020_<br>20u1_raster100m <br>(it might be<br> that you have to <br>change name of <br>the last <br>folder a bit<br> so that it<br> is such as<br> defined in the<br> first column) |Data/Raw_Data/CORINE/|[9]|
|**Validation and output comparison data**|
|{country}_{year}.zip|https://syncandshare.lrz.de/getlink/fiAD95cTrXbnKMrdZYrFFcN8/|Data/Raw_Data/IACS/||[10]|
|EUCROPMAP_2018.tif|https://data.jrc.ec.europa.eu/dataset/15f86c84-eae1-4723-8e00-c1b35c8f56b9|Data/Raw_Data/RSCM/|[11]|


<br>
[1] Palmieri, Alessandra; Eiselt, Beatrice; Lemoine, Guido; Reuter, Hannes Isaak; Martinez-Sanchez, Laura; van der Velde, Marijn; Iordanov, Momtchil; Dominici, Paolo; D'Andrimont, Raphael; Gallego, Javier; Joebges, Christian (2020): Harmonised LUCAS in-situ land cover and use database for field surveys from 2006 to 2018 in the European Union. European Commission, Joint Research Centre (JRC) <br>
[2] Tóth, G., Jones, A., Montanarella, L. (eds.) 2013. LUCAS Topsoil Survey. Methodology, data and results. JRC Technical Reports. Luxembourg. Publications Office of the European Union, EUR26102 – Scientific and Technical Research series <br>
[3] EC-JRC-AGRI4CAST. 2022. “Gridded Agro-Meteorological Data in Europe. European Commission Joint Research Centre. Institute for Environment and Sustainability. Monitor-ing Agricultural Resources (MARS) Unit.” Accessed December 09, 2022 
[4] Ballabio, C., P. Panagos, and L. Monatanarella. 2016. “Mapping topsoil physical properties at European scale using the LUCAS database.” Geoderma 261:110–23. doi:10.1016/j.geoderma.2015.07.006. <br>
[5] EU Copernicus <br>
[6] based on the recommendation at the 1st European Workshop on Reference Grids in 2003 and later INSPIRE geographical grid systems <br>
[7] eurostat/ GISCO <br>
[8] eurostat
[9] EEA https://doi.org/10.2909/960998c1-1870-4e82-8051-6485205ebbac 
[10] Schneider, Maja; Broszeit, Amelie; Körner, Marco (2021). EuroCrops: A Pan-European Dataset for Time Series Crop Type Classification. DOI: 10.48550/arXiv.2106.08151
[11] d’Andrimont, Raphaël; Verhegghen, Astrid; Lemoine, Guido; Kempeneers, Pieter; Meroni, Michele; van der Velde, Marijn (2021). From parcel to continental scale – A first European crop type map based on Sentinel-1 and LUCAS Copernicus in-situ observations. DOI: 10.1016/j.rse.2021.112708 

### additional instructions for the download of the Eurostat data
1. under "format" select "codes" (to make sure NUTS regions and crops are displayed as code and not with their names)
![grafik](https://github.com/JoBaumert/Project-1-Code/assets/59195892/84350588-3b0a-4561-a702-c9c9d106b833)

2. select all geopolitical entities (number of available entities might vary, depending on other selections), the years 2010-2020 and all crops, as well as under "structure of production" --> "Area" (when generating the file "apro_cpshr_20102020_area) or "Main Area"(for generating the file "...main_area"). When generating the file "UAA_all_regions_all_years.csv" select under "Crops" only "Utilized agricultural area" and under "structure of production" only "Main area (1000ha)".
![grafik](https://github.com/JoBaumert/Project-1-Code/assets/59195892/29fde7c4-9e38-453d-9850-a58425dd4137)

3. klick on "Download", then make sure "compress text files.." is not checked, then download by klicking on "SDMX-CSV 1.0"
![grafik](https://github.com/JoBaumert/Project-1-Code/assets/59195892/47eee8ca-5ff1-4134-b522-13f4954aad95)

## Step 3: Running the Python code
The following table indicates which python files require which input data and which output files are created by them. The table should give the user a better understanding of how the files are related. The order in which they are run is specified below.
### Python files <br>

|python file | input files | output files|
|----|----|----|
|LUCAS_preprocessing.py|Raw_Data/LUCAS/lucas_harmo_uf.csv|Intermediary_Data/Preprocessed_Inputs/<br>LUCAS_preprocessed.csv|
||Raw_Data/LUCAS/LUCAS_TOPSOIL_v1.xlsx||
|climate_data_preprocessing.py|Raw_Data/Temperature/*.csv|Intermediary_Data/Preprocessed_Inputs/<br>all_temperature_data.parquet|
||Raw_Data/Precipitation/*.csv|Intermediary_Data/Preprocessed_Inputs/<br>all_precipitation_data.parquet|
|NUTS_preprocessing.py|delineation_and_parameters/<br>DGPCM_user_parameters.xlsx|Intermediary_Data/Preprocessed_Inputs/<br>NUTS_all_regions_all_years.(csv and shp)|
||Raw_Data/NUTS/<br>NUTS_RG_01M_{nuts_year}_3035.shp.zip||
|Eurostat_preprocessing.py|Intermediary_Data/Preprocessed_Inputs/NUTS/<br>NUTS_all_regions_all_years.csv|Intermediary_Data/Preprocessed_Inputs/Eurostat<br>/Eurostat_cropdata_compiled_{FirstyearLastyear}_DGPCMcodes.csv|
||Raw_Data/Eurostat/<br>apro_cpshr_20102020_main_area.csv|Intermediary_Data/Preprocessed_Inputs/<br>Eurostat/Eurostat_UAA_compiled_{FirstyearLastyear}.csv|
||Raw_Data/Eurostat/<br>apro_cpshr_20102020_area.csv||
||Raw_Data/Eurostat<br>/EUROSTAT_crops_total_NUTS3_2010_final.xlsx (if available)||
||Raw_Data/Eurostat/<br>UAA_all_regions_all_years.csv||
|linking_LUCAS_and_<br>explanatory_vars.py|Raw_Data/DEM/eudem_dem_3035_europe.tif|Intermediary_Data/LUCAS_feature_merges/<br>{country}/elevation.csv|
||Raw_Data/DEM/eudem_slop_3035_europe/eudem_slop_3035_europe.tif|Intermediary_Data/LUCAS_feature_merges/<br>{country}/slope.csv|
||Raw_Data/Soil/Sand_Extra.zip/<br>Sand1.tif|Intermediary_Data/LUCAS_feature_merges/{country}/sand.csv|
||Raw_Data/Soil/Clay_Extra.zip/<br>Clay.tif|Intermediary_Data/LUCAS_feature_merges/{country}/clay.csv|
||Raw_Data/Soil/Silt_Extra.zip/<br>Silt1.tif|Intermediary_Data/LUCAS_feature_merges/{country}/silt.csv|
||Raw_Data/Soil/BulkDensity_Extra.zip/<br>Bulk_density.tif|Intermediary_Data/LUCAS_feature_merges/{country}/bulk_density.csv|
||Raw_Data/Soil/CoarseFragments_Extra.zip/<br>Coarse_fragments.tif|Intermediary_Data/LUCAS_feature_merges/{country}/coarse_fragments.csv|
||Raw_Data/Soil/AWC_Extra.zip/<br>AWC.tif|Intermediary_Data/LUCAS_feature_merges/{country}/awc.csv|
||Intermediary_Data/Preprocessed_Inputs/<br>Climate/all_temperature_data.parquet|Intermediary_Data/LUCAS_feature_merges/{country}/avg_annual_temp_sum.csv and avg_annual_veg_period.csv|
||Intermediary_Data/Preprocessed_Inputs/<br>Climate/all_precipitation_data.parquet|Intermediary_Data/LUCAS_feature_merges/{country}/avg_annual_precipitation.csv|
||Intermediary_Data/Preprocessed_Inputs/<br>LUCAS/LUCAS_preprocessed.csv|Intermediary_Data/LUCAS_feature_merges/{country}/latitude4326.csv|
||Intermediary_Data/Preprocessed_Inputs/<br>NUTS/NUTS_all_regions_all_years.shp||
||Raw_Data/Grid/grid25.zip||
|croparea_and_UAA_<br> preparation.py|Intermediary_Data/Preprocessed_Inputs/Eurostat/Eurostat_cropdata_compiled_{FirstyearLastyear}_DGPCMcodes.csv|Intermediary_Data/Regional_Aggregates/cropdata_{FirstyearLastyear}.csv|
||Intermediary_Data/Preprocessed_Inputs/Eurostat/Eurostat_UAA_compiled_{FirstyearLastyear}.csv|Intermediary_Data/Regional_Aggregates/coherent_UAA_{FirstyearLastyear}.csv|
||Intermediary_Data/NUTS/NUTS_all_regions_all_years.csv|Intermediary_Data/Regional_Aggregates/crop_levels_selected_countries_{FirstyearLastyear}|
|grid_preparation.py|Raw_Data/Grid/{country}_1km.zip|Intermediary_Data/Zonal_Stats/{country}/cell_size/1kmgrid_{nuts}_all_years.csv|
||Raw_Data/NUTS/NUTS_RG_01M_{year}_3035.shp||
|linking_gridcells_and_<br>explanatory_vars.py|Raw_Data/DEM/eudem_dem_3035_europe.tif|Intermediary_Data/Zonal_Stats/{country}/elevation/1kmgrid_{nuts1}.csv|
||Raw_Data/DEM/eudem_slop_3035_europe/eudem_slop_3035_europe.tif|Intermediary_Data/Zonal_Stats/{country}/slope/1kmgrid_{nuts1}.csv|
||Raw_Data/Soil/Sand_Extra.zip/<br>Sand1.tif|Intermediary_Data/Zonal_Stats/{country}/sand/1kmgrid_{nuts1}.csv|
||Raw_Data/Soil/Clay_Extra.zip/<br>Clay.tif|Intermediary_Data/Zonal_Stats/{country}/clay/1kmgrid_{nuts1}.csv|
||Raw_Data/Soil/Silt_Extra.zip/<br>Silt1.tif|Intermediary_Data/Intermediary_Data/Zonal_Stats/{country}/silt/1kmgrid_{nuts1}.csv|
||Raw_Data/Soil/BulkDensity_Extra.zip/<br>Bulk_density.tif|Intermediary_Data/Zonal_Stats/{country}/bulk_density/1kmgrid_{nuts1}.csv|
||Raw_Data/Soil/CoarseFragments_Extra.zip/<br>Coarse_fragments.tif|Intermediary_Data/Zonal_Stats/{country}/coarse_fragments/1kmgrid_{nuts1}.csv|
||Raw_Data/Soil/AWC_Extra.zip/<br>AWC.tif|Intermediary_Data/Zonal_Stats/{country}/awc/1kmgrid_{nuts1}.csv|
||Intermediary_Data/Preprocessed_Inputs/<br>Climate/all_temperature_data.parquet|Intermediary_Data/Zonal_Stats/{country}/avg_annual_temp_sum_{previous_years}/1kmgrid_{nuts1}.csv|
||Intermediary_Data/Preprocessed_Inputs/<br>Climate/all_temperature_data.parquet|Intermediary_Data/Zonal_Stats/{country}/avg_annual_veg_period_{previous_years}/1kmgrid_{nuts1}.csv|
||Intermediary_Data/Preprocessed_Inputs/<br>Climate/all_precipitation_data.parquet|Intermediary_Data/Zonal_Stats/{country}/avg_annual_precipitation_{previous_years}/1kmgrid_{nuts1}.csv|
||Raw_Data/CORINE/clc{clc_year}_<br>v2020_20u1_raster100m|Intermediary_Data/Zonal_Stats/{country}/CORINE_agshare/1kmgrid_{nuts1}_{year}|
||Intermediary_Data/Zonal_Stats/{country}/<br>cell_size/1kmgrid_{nuts}_all_years.csv|Intermediary_Data/Zonal_Stats/{country}/inferred_UAA/1kmgrid_{nuts1}_all_years.csv|
||Intermediary_Data/Preprocessed_Inputs/<br>NUTS/NUTS_all_regions_all_years.shp||
||Raw_Data/Grid/grid25.zip||
||Raw_Data/Grid/{country}_1km.zip|Intermediary_Data/Zonal_Stats/{country}/latitude4326/1kmgrid_{nuts1}.csv|
|generate_optimization_constraints <br> _cellweights.py|Intermediary_Data/Regional_Aggregates/coherent_UAA_{FirstyearLastyear}.csv|Intermediary_Data/Regional_Aggregates/Cell_Weights/{nuts0}/cell_weights_{FirstyearLastyear}.csv|
||Intermediary_Data/Regional_Aggregates/<br>cropdata_{FirstyearLastyear}.csv||
||Intermediary_Data/Preprocessed_Inputs/<br>NUTS/NUTS_all_regions_all_years.csv||
||Intermediary_Data/Zonal_stats/{country}<br>/inferred_UAA/1kmgrid_{nuts1}.csv||
|LUCAS_field_size_calculation.py|Intermediary_Data/Preprocessed_INputs/LUCAS/<br>LUCAS_preprocessed.csv|Intermediary_Data/Zonal_Stats/{country}/n_of_fields/n_of_fields_allcountry_{FirstyearLastyear}.csv|
||Intermediary_Data/preprocessed_Inputs/<br>NUTS/NUTS_all_regions_all_years.csv||
||Intermediary_Data/Zonal_Stats/{country}/<br>inferred_UAA/1kmgrid_{nuts1}.csv||
|model_parameter_estimation.py|Intermediary_Data/Preprocessed_Inputs/LUCAS/<br>LUCAS_preprocessed.csv|Results/Model_Parameter_Estimates/multinomial_logit_{country}_statsmodel_params_obsthreshold{minthreshold}.xlsx|
||Intermediary_Data/LUCAS_feature_merges/<br>{country}/*.csv|Results/Model_Parameter_Estimates/multinomial_logit_{country}_statsmodel_covariance_obsthreshold{minthreshold}.xlsx|
|||Results/Model_Parameter_Estimates/scale_factors/<br>standardscaler_multinom_logit_{country}|
|calculation_of_prior_ <br>crop_probabilities.py|Intermediary_Data/Zonal_Stats/{country}/*|Results/Prior_crop_probability_estimates/{country}/{nuts1}_{year}|
||Intermediary_Data/preprocessed_Inputs/NUTS/NUTS_all_regions_all_years.csv||
||Results/Model_Parameter_Estimates/scale_factors/<br>standardscaler_multinom_logit_{country}||
||Results/Model_Parameter_Estimates/<br>multinomial_logit_{country}_statsmodel_<br>params_obsthreshold{minthreshold}.xlsx||
||Results/Model_Parameter_Estimates/<br>multinomial_logit_{country}_statsmodel_<br>covariance_obsthreshold{minthreshold}.xlsx||
|incorporation_of_<br>aggregated_info.py|Intermediary_Data/Preprocessed_Inputs/<br>NUTS/NUTS_all_regions_all_years.csv|Results/Posterior_crop_probability_estimates/{country}/{country}{year}entire_country.parquet|
||Results/Prior_crop_probability_estimates/<br>{country}/{nuts1}_{year}.parquet|Results/Posterior_crop_probability_estimates/{country}/{country}{year}entire_country_hyperparameters.csv|
||Intermediary_Data/Regional_Aggregates/<br>coherent_UAA_{FirstyearLastyear}.csv||
||Intermediary_Data/Regional_Aggregates/<br>cropdata_{FirstyearLastyear}.csv||
||Intermediary_Data/Regional_Aggregates/<br>crop_levels_selected_countries_{FirstyearLastyear}.csv||
||Intermediary_Data/Regional_Aggregates/Cell_Weights/<br>{country}/cell_weights_{FirstyearLastyear}.csv||
|simulation_of_crop_shares.py|Intermediary_Data/Preprocessed_Inputs/NUTS/<br>NUTS_all_regions_all_years.csv|Results/Simulated_consistent_crop_shares/{country}/{year}/{country}{year}_{nuts1}.parquet|
||Intermediary_Data/Regional_Aggregates/<br>coherent_UAA_{FirstyearLastyear}.csv|Results/Simulated_consistent_crop_shares/{country}/{year}/{country}{year}_deviation_from_aggregated.csv|
||Intermediary_Data/Regional_Aggregates/<br>cropdata_{FirstyearLastyear}.csv||
||Intermediary_Data/Regional_Aggregates/<br>crop_levels_selected_countries_{FirstyearLastyear}.csv||
||Results/Posterior_crop_probability_estimates/<br>{country}/{country}{year}entire_country.parquet||
||Intermediary_Data/Zonal_Stats/{country}/<br>n_of_fields/n_of_fields_allcountry_{FirstyearLastyear}.csv||
||Intermediary_Data/Regional_Aggregates/Cell_Weights/<br>{country}/cell_weights_{FirstyearLastyear}.csv||
|RSCM_to_DGPCM_grid.py|Intermediary_Data/Preprocessed_Inputs/NUTS/<br>NUTS_all_regions_all_years.csv|Intermediary_Data/Preprocessed_Inputs/RSCM/{country}/{nuts1}_1km_reference_grid.csv|
||Raw_Data/RSCM/EUCROPMAP_2018.tif||
||Intermediary_Data/Zonal_Stats/{country}/<br>cell_size/1km_grid_{nuts1}_all_years.csv||
|visualize_cropmap.py|Intermediary_Data/Preprocessed_Inputs/NUTS/<br>NUTS_all_regions_all_years.shp|Results/Validations_and_Visualizations/Expected_crop_shares/share_of_{crop}_{year}.png|
||Raw_Data/Grid/{country}_1km.zip||
||Results/Posterior_crop_probability_estimates/<br>{country}/{country}{year}entire_country.parquet||
|IACS_to_DGPCM_grid.py|Intermediary_Data/Preprocessed_Inputs/NUTS/<br>NUTS_all_regions_all_years.shp|Intermediary_Data/Preprocessed_Input/IACS/true_shares_{NUTS2}_{year}.csv|
||Raw_Data/IACS/{country}_{year}.zip||
||Raw_Data/Grid/{country}_1km.zip||
||Intermediary_Data/Zonal_Stats/{country}/<br>cell_size/1km_grid_{nuts1}_all_years.csv||
|dominant_crops_<br>DGPCM_RSCM_IACS.py|Intermediary_Data/Preprocessed_Inputs/NUTS/<br>NUTS_all_regions_all_years.csv/.shp|Results/Validations_and_Visualizations/Comparison_dominant_crops/{map}dominant_crops_{country}_{year}.png|
||Intermediary_Data/Preprocessed_Inputs/<br>IACS/true_shares/true_shares_{NUTS2}_{year}.csv||
||Intermediary_Data/Preprocessed_Inputs/<br>RSCM/{country}/{nuts1}_1km_reference_grid.csv||
||Results/Posterior_crop_probability_estimates/<br>{country}/{country}{year}enture_country.parquet||
||Raw_Data/Grid/{country}_1km.zip||
|grid_transformation_1km_10km.py|Raw_Data/Grid/{country}_10km.zip|Intermediary_Data/Preprocessed_Inputs/Grid/Grid_conversion_1km_10km_{country}.csv|
|calculate_wMAE.py|Intermediary_Data/Preprocessed_Inputs/NUTS/<br>NUTS_all_regions_all_years.csv|Results/Validations_and_Visualizations/Comparison_metrics/{country}{year}_pearsonr_and_wMAE_comparison_DGPCM_RSCM_1km/10km.csv|
||Intermediary_Data/Preprocessed_Inputs/IACS/<br>true_shares/true_shares_{NUTS2}_{year}.csv|Results/Validations_and_Visualizations/Comparison_metrics/wMAE_1km(10km)_boxplot.png|
||Intermediary_Data/Preprocessed_Inputs/RSCM/<br>{country}/{nuts1}_1km_reference_grid.csv||
||Results/Posterior_crop_probability_estimates/<br>{country}/{country}{year}enture_country.parquet||
||Raw_Data/Grid/{country}_1km.zip/10km.zip||
||Intermediary_Data/Preprocessed_Inputs/Grid/<br>Grid_conversion_1km_10km_{country}.csv||
|calculation_and_visualization<br>_of_HDI.py|Intermediary_Data/Preprocessed_Inputs/NUTS/<br>NUTS_all_regions_all_years.csv|Results/Credible_Intervals/{country}/{nuts2}{year}_%_HDI.csv|
||Results/Simulated_consistent_crop_shares/<br>{country}/{year}/{country}{year}_{nuts1}|Results/Validations_and_Visualizations/HDI_width/{country}{year}_{crop}.png|
||Intermediary_Data/Preprocessed_Inputs/IACS/<br>true_shares/true_shares_{nuts2}_{year}.csv||
||Raw_Data/Grid/{country}_1km.zip||
||Results/Posterior_crop_probability_estimates/<br>{country}/{country}{year}entire_country.parquet||





## Recommended order of running python files
---preprocessing files---
1. LUCAS_preprocessing.py
2. climate_data_preprocessing.py
3. NUTS_preprocessing.py
4. Eurostat_preprocessing.py
5. croparea_and_UAA_preparation.py
6. grid_preparation.py
7. linking_LUCAS_and_explanatory_vars.py
8. linking_gridcells_and_explanatory_vars.py
9. generate_optimization_constraints_cellweights.py
10. LUCAS_field_size_calculation.py <br>

---parameter estimation---  <br>

11. model_parameter_estimation.py <br>
    
---prior crop probability prediction for entire country--- <br>

12. calculation_of_prior_crop_probabilities.py <br>

---incorporation of regional/national aggregates---  <br>

13. incorporation_of_aggregated_info.py <br>

---simulation of crop shares--- <br>

14. simulation_of_crop_shares.py
    
---validation and visualization of results--- <br>

15. visualize_cropmap.py
16. IACS_to_DGPCM_grid.py
17. RSCM_to_DGPCM_grid.py
18. dominant_crops_DGPCM_RSCM_IACS.py
19. grid_transformation_1km_10km.py
20. calculate_wMAE.py
21. calculation_and_visualization_of_HDI.py

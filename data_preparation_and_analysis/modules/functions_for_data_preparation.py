# %%
from cgi import test
from re import S
import geopandas as gpd
import numpy as np
import pyproj
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import from_bounds
import os
import richdem as rd
from os.path import exists
import zipfile
from pathlib import Path

# %%
#%%
def get_NUTS_regions(NUTS_gdf, country_code):
    NUTS1_regs = np.sort(
        np.array(
            NUTS_gdf[
                (NUTS_gdf["CNTR_CODE"] == country_code) & (NUTS_gdf["LEVL_CODE"] == 1)
            ]["NUTS_ID"]
            .value_counts()
            .keys()
        )
    )
    NUTS2_regs = np.sort(
        np.array(
            NUTS_gdf[
                (NUTS_gdf["CNTR_CODE"] == country_code) & (NUTS_gdf["LEVL_CODE"] == 2)
            ]["NUTS_ID"]
            .value_counts()
            .keys()
        )
    )
    NUTS3_regs = np.sort(
        np.array(
            NUTS_gdf[
                (NUTS_gdf["CNTR_CODE"] == country_code) & (NUTS_gdf["LEVL_CODE"] == 3)
            ]["NUTS_ID"]
            .value_counts()
            .keys()
        )
    )
    return {"NUTS1": NUTS1_regs, "NUTS2": NUTS2_regs, "NUTS3": NUTS3_regs}


def getfilefromcoords(
    filenames, latlist, lonlist, elev_raster_height=20, elev_raster_width=30
):
    """to avoid going through all .tif files only the relevant files are loaded"""
    # filenames is the list of all files that are potentially relevant
    # latlist is a list of the latitudes of all relevant LUCAS points
    # lonlist is a list of the longitudes of all relevant LUCAS points
    coordfile = np.ndarray((len(filenames), 2))
    for f, file in enumerate(all_elev_files):
        coordfile[f][0] = int(file[:2])
        if file[6] == "W":
            coordfile[f][1] = -1 * int(file[4:6])
        else:
            coordfile[f][1] = int(file[4:6])

    rel_lat_pos = np.where(
        (np.transpose(coordfile)[0] < latlist.min())
        & (np.transpose(coordfile)[0] > latlist.min() - elev_raster_height)
        | (
            (np.transpose(coordfile)[0] < latlist.max())
            & (np.transpose(coordfile)[0] > latlist.min())
        )
    )
    rel_lon_pos = np.where(
        (np.transpose(coordfile)[1] < lonlist.min())
        & (np.transpose(coordfile)[1] > lonlist.min() - elev_raster_width)
        | (
            (np.transpose(coordfile)[1] < lonlist.max())
            & (np.transpose(coordfile)[1] > lonlist.min())
        )
    )

    # return np.array(filenames)[np.where(np.in1d(rel_lat_pos,rel_lon_pos))[0]]
    return (
        np.array(filenames)[
            rel_lat_pos[0][np.where(np.in1d(rel_lat_pos, rel_lon_pos))[0]]
        ],
        coordfile[rel_lat_pos[0][np.where(np.in1d(rel_lat_pos, rel_lon_pos))[0]]],
    )


def calculateslope(
    relevant_files,
    relevant_boundary_coords4326,
    latlist4326,
    lonlist4326,
    latlist3035,
    lonlist3035,
):
    elev_raster_height, elev_raster_width = 20, 30
    relevant_slope, relevant_aspect = [], []
    # for f,file in enumerate(relevant_elev_files_epsg3035):
    for f, file in enumerate(relevant_files):
        affine_richdem = [
            file.transform[2],
            file.transform[0],
            0,
            file.transform[5],
            0,
            file.transform[0],
        ]
        eldata_nparray = file.read().astype("float64")
        eldata_rd = rd.rdarray(eldata_nparray, no_data=-9999)
        eldata_rd.geotransform = affine_richdem
        eldata_rd.crs = "epsg:3035"
        relevant_slope.append(
            rd.TerrainAttribute(eldata_rd[0], attrib="slope_percentage")
        )
        relevant_aspect.append(rd.TerrainAttribute(eldata_rd[0], attrib="aspect"))
    slope_results, aspect_results = [], []
    for l, _ in enumerate(lonlist4326):
        matched = 0  # indicates if match was found or not
        for i, j in enumerate(relevant_boundary_coords4326):
            if (
                (lonlist4326[l] >= j[1])
                and (lonlist4326[l] <= j[1] + elev_raster_width)
                and (latlist4326[l] >= j[0])
                and (latlist4326[l] <= j[0] + elev_raster_height)
            ):
                x, y = relevant_files[i].index(lonlist3035[l], latlist3035[l])
                slope_results.append(relevant_slope[i][x][y])
                aspect_results.append(relevant_aspect[i][x][y])
                matched = 1
        if matched == 0:
            slope_results.append(np.nan)
            aspect_results.append(np.nan)
    return slope_results, aspect_results


def get_annual_temp_sum(temperature_data):
    return (
        temperature_data[["GRID_NO", "TEMPERATURE_AVG", "year"]]
        .groupby(["year", "GRID_NO"])
        .sum()
        .reset_index()
    )


# temperature_veg_period indicates the number of days with temperatures >5°C
def get_annual_veg_period(temperature_data):
    return (
        temperature_data[["GRID_NO", "TEMPERATURE_AVG", "year"]]
        .groupby(["year", "GRID_NO"])["TEMPERATURE_AVG"]
        .apply(lambda x: x[x > 5].count())
        .reset_index()
    )


def get_annual_precipitation_sum(precipitation_data):
    return (
        precipitation_data[["GRID_NO", "PRECIPITATION", "year"]]
        .groupby(["year", "GRID_NO"])
        .sum()
        .reset_index()
    )


def create_equidistant_points(grid_1km_selected_NUTS1):
    sel_points_3035 = {
        "CELLCODE": [],
        "left": [],
        "bottom": [],
        "right": [],
        "top": [],
        "sel_coords_3035": [],
    }
    for c, cc in enumerate(grid_1km_selected_NUTS1["CELLCODE"]):
        left, bottom, right, top = grid_1km_selected_NUTS1.iloc[c : c + 1].total_bounds
        point_dist = 1000 / (nofpoints_km**0.5)
        bound_dist = point_dist / 2
        for px in range(int(nofpoints_km**0.5)):
            for py in range(int(nofpoints_km**0.5)):
                sel_points_3035["CELLCODE"].append(cc)
                sel_points_3035["left"].append(left)
                sel_points_3035["bottom"].append(bottom)
                sel_points_3035["right"].append(right)
                sel_points_3035["top"].append(top)
                xcoord, ycoord = (
                    left + bound_dist + px * point_dist,
                    bottom + bound_dist + py * point_dist,
                )
                sel_points_3035["sel_coords_3035"].append(Point(xcoord, ycoord))

    sel_points_3035_df = pd.DataFrame(sel_points_3035)
    sel_points_3035_gdf = gpd.GeoDataFrame(
        sel_points_3035_df,
        geometry=sel_points_3035_df["sel_coords_3035"],
        crs="epsg:3035",
    )
    sel_points_4326_gdf = sel_points_3035_gdf.to_crs(crs="epsg:4326")
    sel_points_4326_gdf.rename(
        columns={"sel_coords_3035": "sel_coords_4326"}, inplace=True
    )
    return sel_points_3035_gdf, sel_points_4326_gdf


def calculate_cells_mean_slope(slope_points_4326, grid_1km_selected, all_cells_1km):
    slope_cells_dict = {"CELLCODE": [], "slope": []}
    for cc in all_cells_1km:
        slope_cells_dict["CELLCODE"].append(cc)
        slope_cells_dict["slope"].append(
            slope_points_4326[slope_points_4326["CELLCODE"] == cc]["slope"].mean()
        )
    slope_cells_df = pd.DataFrame(slope_cells_dict)
    slope_grid_1km_selected = grid_1km_selected.merge(
        slope_cells_df, how="left", on="CELLCODE"
    )
    return slope_grid_1km_selected[["CELLCODE", "geometry", "slope"]]


def get_elevation_old(
    relevant_elev_files_3035,
    relevant_elev_file_names_4326,
    grid_1km_selected,
    elev_path_3035,
):
    elevation_grid = {"CELLCODE": [], "valid_counts": [], "elevation": []}
    for f, file in enumerate(relevant_elev_files_3035):
        right_bnd_file = file.transform[2] + file.shape[1] * file.transform[0]
        bottom_bnd_file = file.transform[5] + file.shape[0] * file.transform[4]
        left_bnd_file = file.transform[2]
        top_bnd_file = file.transform[5]
        for c, cc in enumerate(grid_1km_selected["CELLCODE"]):
            left, bottom, right, top = grid_1km_selected.iloc[c : c + 1].total_bounds
            if (
                (left >= left_bnd_file)
                and (left < right_bnd_file)
                and (top <= top_bnd_file)
                and (top > bottom_bnd_file)
            ):
                with rio.open(
                    elev_path_3035
                    + relevant_elev_file_names_4326[f][:-4]
                    + "_epsg3035.tif"
                ) as src:
                    rst = src.read(
                        1, window=from_bounds(left, bottom, right, top, src.transform)
                    )
                elevation_grid["CELLCODE"].append(cc)
                valid_counts = np.count_nonzero(
                    rst > -100
                )  # count only the values that take reasonable values (i.e., avoid no data values)
                elevation_grid["valid_counts"].append(valid_counts)
                if valid_counts > 0:
                    elevation_grid["elevation"].append(
                        np.average(rst, weights=(rst > -100))
                    )
                else:
                    elevation_grid["elevation"].append(rst.mean())
    elevation_grid_preliminary_df = pd.DataFrame(elevation_grid)
    # some elevation results are counted more than once (if they are at a border). Ensure correct weighing
    elevation_grid_preliminary_df["sumproduct"] = (
        elevation_grid_preliminary_df["valid_counts"]
        * elevation_grid_preliminary_df["elevation"]
    )
    grouped_elevation = (
        elevation_grid_preliminary_df[elevation_grid_preliminary_df["sumproduct"] > 0]
        .groupby(["CELLCODE"])
        .sum()
    )
    grouped_elevation.drop(columns="elevation", inplace=True)
    grouped_elevation["elevation"] = (
        grouped_elevation["sumproduct"] / grouped_elevation["valid_counts"]
    )
    grouped_elevation_merged = grouped_elevation.merge(
        grid_1km_selected, how="left", on="CELLCODE"
    )
    elevation_grid_1km_selected = gpd.GeoDataFrame(grouped_elevation_merged)
    return elevation_grid_1km_selected[["CELLCODE", "geometry", "elevation"]]


def grid25km_to_grid1km(grid_25km, grid_1km, grid_1km_lon, grid_1km_lat):
    grid_array = np.array(
        grid_25km[["GRID_NO", "XMIN_x", "XMAX_x", "YMIN_x", "YMAX_x"]]
    )
    # grid_array=np.array(grid_25km[["Grid_Code","XMIN", "XMAX", "YMIN", "YMAX"]])
    matched_CELLCODES = [
        np.array(grid_1km["CELLCODE"])[
            np.where(
                (grid_1km_lon > g[1])
                & (grid_1km_lon <= g[2])
                & (grid_1km_lat > g[3])
                & (grid_1km_lat <= g[4])
            )[0]
        ]
        for g in grid_array
    ]  #%%
    Grid_no = [
        np.repeat(gridno, len(matched_CELLCODES[g]))
        for g, gridno in enumerate(grid_array.transpose()[0])
    ]
    matched_CELLCODES = np.array(
        [item for sublist in matched_CELLCODES for item in sublist]
    )
    Grid_no = np.array([item for sublist in Grid_no for item in sublist])
    grid_link_df = pd.DataFrame({"CELLCODE": matched_CELLCODES, "GRID_NO": Grid_no})
    grid_link_df.drop_duplicates(inplace=True)
    return grid_link_df


def get_rel_positions(rio_tif_path, grid_1km_selected, window=False):
    if not window:
        rio_tif = rio.open(rio_tif_path)
        data_read = rio_tif.read(1)
    else:
        with rio.open(rio_tif_path) as rio_tif:
            data_read = rio_tif.read(
                1,
                window=from_bounds(
                    window[0], window[1], window[2], window[3], rio_tif.transform
                ),
            )

    # here is an explanation for what the elements of transform mean:
    # https://geobgu.xyz/py/rasterio1.html
    affine = np.array(rio_tif.transform).reshape(3, 3)
    if window is not False:
        affine[0][2] = window[0]
        affine[1][2] = window[3]
    affine_inv = np.linalg.inv(affine)

    eoforigin = np.array(grid_1km_selected["EOFORIGIN"])
    noforigin = np.array(grid_1km_selected["NOFORIGIN"]) + 1000
    onevector = np.ones(len(eoforigin))

    coord_matrix = np.concatenate((eoforigin, noforigin, onevector)).reshape(3, -1)
    rio_positions = np.matmul(affine_inv, coord_matrix)
    rio_positions = rio_positions[:2].astype(int)
    return rio_positions, data_read


def get_agshare(rio_tif, grid_1km_selected,corine_ag_classes=None):

    rio_positions, data_read = get_rel_positions(rio_tif, grid_1km_selected)

    a, b = np.meshgrid(rio_positions[1], np.arange(10))
    c = np.add(a, b)
    d = np.repeat(c.transpose(), 10, axis=1).flatten()

    aa, bb = np.meshgrid(rio_positions[0], np.arange(10))
    cc = np.add(aa, bb)
    dd = np.tile(cc.transpose(), 10).flatten()

    corine_classes = np.array(data_read[list(d), list(dd)])
    corine_classes_matrix = corine_classes.reshape(rio_positions.shape[1], 100)

    # agricultural_classes = [12, 13, 14, 15, 16, 17, 18, 19, 20, 22]

    if corine_ag_classes is not None:
        ag_classes=np.where(np.isin(corine_classes_matrix,corine_ag_classes),1,0)
    else:
        ag_classes = np.where(
            (
                (corine_classes_matrix >= 12) & (corine_classes_matrix <= 20)
                | (corine_classes_matrix == 22)
            ),
            1,
            0,
        )
    ag_freqs = ag_classes.sum(axis=1)
    agshare = ag_freqs / 100
    # grid_1km_selected["agshare"] = agshare
    return agshare


def get_soil_content(rio_tif_path, grid_1km_selected, resolution):
    # resolution in m per pixel
    n_of_pixels = int(1000 / resolution)
    rio_positions, data_read = get_rel_positions(rio_tif_path, grid_1km_selected)

    a, b = np.meshgrid(rio_positions[1], np.arange(n_of_pixels))
    c = np.add(a, b)
    d = np.repeat(c.transpose(), n_of_pixels, axis=1).flatten()

    aa, bb = np.meshgrid(rio_positions[0], np.arange(n_of_pixels))
    cc = np.add(aa, bb)
    dd = np.tile(cc.transpose(), n_of_pixels).flatten() + 1

    """very few cellss are outside some maps (e.g., points in Lambedusa outside of coarse fragments map)
        in this case returns the value -999. We accomplish this by adding many -999 to both axes of the map so that
        it reads -999 if a position is outside the dimensions of the map:    
    """
    data_read_appended = np.append(
        data_read,
        np.repeat(-999, data_read.shape[0] * 1000).reshape(data_read.shape[0], 1000),
        axis=1,
    )
    data_read_appended = np.append(
        data_read_appended,
        np.repeat(-999, data_read_appended.shape[1] * 1000).reshape(
            1000, data_read_appended.shape[1]
        ),
        axis=0,
    )

    oc_content = np.array(data_read_appended[list(d), list(dd)])
    oc_content_matrix = oc_content.reshape(rio_positions.shape[1], n_of_pixels**2)
    oc_content_matrix = np.where(oc_content_matrix < 0, np.nan, oc_content_matrix)
    oc_content_mean = np.nanmean(oc_content_matrix, axis=1)
    return oc_content_mean


def get_elevation(rio_tif_path, grid_1km_selected, window):

    rio_positions, data_read = get_rel_positions(
        rio_tif_path, grid_1km_selected, window
    )
    a, b = np.meshgrid(rio_positions[1], np.arange(40))
    c = np.add(a, b)
    d = np.repeat(c.transpose(), 40, axis=1).flatten()

    aa, bb = np.meshgrid(rio_positions[0], np.arange(40))
    cc = np.add(aa, bb)
    dd = np.tile(cc.transpose(), 40).flatten()

    elev = np.array(data_read[list(d), list(dd)])
    elev_matrix = elev.reshape(rio_positions.shape[1], 40 * 40)
    elev_mean = np.mean(elev_matrix, axis=1)
    return elev_mean


def link_grid_and_points(grid, points_lon, points_lat, df):
    grid_nparray = np.array(grid[["Grid_Code", "XMIN", "XMAX", "YMIN", "YMAX"]])

    LUCAS_IDs = [
        np.array(df["id"])[
            np.where(
                (points_lon > g[1])
                & (points_lon <= g[2])
                & (points_lat > g[3])
                & (points_lat <= g[4])
            )[0]
        ]
        for g in grid_nparray
    ]
    Grid_no = [
        np.repeat(gridno, len(LUCAS_IDs[g]))
        for g, gridno in enumerate(grid_nparray.transpose()[0])
    ]

    LUCAS_IDs = np.array([item for sublist in LUCAS_IDs for item in sublist])
    Grid_no = np.array([item for sublist in Grid_no for item in sublist])

    LUCAS_grid_link_df = pd.DataFrame({"id": LUCAS_IDs, "Grid_Code": Grid_no})
    LUCAS_grid_link_df.drop_duplicates(inplace=True)
    return LUCAS_grid_link_df


def get_rel_positions_LUCAS(rio_tif_path, LUCAS_lon, LUCAS_lat, window=False):
    if not window:
        rio_tif = rio.open(rio_tif_path)
        data_read = rio_tif.read(1)
    else:
        with rio.open(rio_tif_path) as rio_tif:
            data_read = rio_tif.read(
                1,
                window=from_bounds(
                    window[0], window[1], window[2], window[3], rio_tif.transform
                ),
            )

    # here is an explanation for what the elements of transform mean:
    # https://geobgu.xyz/py/rasterio1.html
    affine = np.array(rio_tif.transform).reshape(3, 3)
    if window is not False:
        affine[0][2] = window[0]
        affine[1][2] = window[3]
    affine_inv = np.linalg.inv(affine)

    eoforigin = np.array(LUCAS_lon)
    noforigin = np.array(LUCAS_lat)
    onevector = np.ones(len(eoforigin))

    coord_matrix = np.concatenate((eoforigin, noforigin, onevector)).reshape(3, -1)
    rio_positions = np.matmul(affine_inv, coord_matrix)
    rio_positions = rio_positions[:2].astype(int)
    return rio_positions, data_read


def get_soil_content_LUCAS(rio_tif_path, LUCAS_lon, LUCAS_lat):

    rio_positions, data_read = get_rel_positions_LUCAS(
        rio_tif_path, LUCAS_lon, LUCAS_lat
    )

    a, b = np.meshgrid(rio_positions[1], np.arange(1))
    c = np.add(a, b)
    d = np.repeat(c.transpose(), 1, axis=1).flatten()

    aa, bb = np.meshgrid(rio_positions[0], np.arange(1))
    cc = np.add(aa, bb)
    dd = np.tile(cc.transpose(), 1).flatten()

    """very few LUCAS points are outside some maps (e.g., points in Lambedusa outside of coarse fragments map)
        in this case returns the value -999. We accomplish this by adding many -999 to both axes of the map so that
        it reads -999 if a position is outside the dimensions of the map:    
    """
    data_read_appended = np.append(
        data_read,
        np.repeat(-999, data_read.shape[0] * 1000).reshape(data_read.shape[0], 1000),
        axis=1,
    )
    data_read_appended = np.append(
        data_read_appended,
        np.repeat(-999, data_read_appended.shape[1] * 1000).reshape(
            1000, data_read_appended.shape[1]
        ),
        axis=0,
    )
    oc_content = np.array(data_read_appended[list(d), list(dd)])
    oc_content_matrix = oc_content.reshape(rio_positions.shape[1], 1)
    oc_content_matrix = np.where(oc_content_matrix < 0, np.nan, oc_content_matrix)
    oc_content_mean = np.nanmean(oc_content_matrix, axis=1)
    return oc_content_mean


def get_elevation_LUCAS(rio_tif_path, LUCAS_lon, LUCAS_lat, window):

    rio_positions, data_read = get_rel_positions_LUCAS(
        rio_tif_path, LUCAS_lon, LUCAS_lat, window
    )
    a, b = np.meshgrid(rio_positions[1], np.arange(1))
    c = np.add(a, b)
    d = np.repeat(c.transpose(), 1, axis=1).flatten()

    aa, bb = np.meshgrid(rio_positions[0], np.arange(1))
    cc = np.add(aa, bb)
    dd = np.tile(cc.transpose(), 1).flatten()

    elev = np.array(data_read[list(d), list(dd)])
    elev_matrix = elev.reshape(rio_positions.shape[1], 1)
    elev_mean = np.mean(elev_matrix, axis=1)
    return elev_mean


def get_relevant_croparea(main_area_selected, area_selected, crop_delineation):

    """first consider those crops in the column 'Eurostat_code' (priority)"""
    main_area_crops = np.array(
        main_area_selected[
            (main_area_selected["float_value"] >= 0)
            & (
                main_area_selected["CROPS"].isin(
                    crop_delineation["Eurostat_code"].value_counts().keys()
                )
            )
        ]["CROPS"]
    )
    main_area_values = np.array(
        main_area_selected[
            (main_area_selected["float_value"] >= 0)
            & (
                main_area_selected["CROPS"].isin(
                    crop_delineation["Eurostat_code"].value_counts().keys()
                )
            )
        ]["float_value"]
    )
    area_crops = np.array(
        area_selected[
            (area_selected["float_value"] >= 0)
            & (
                area_selected["CROPS"].isin(
                    crop_delineation["Eurostat_code"].value_counts().keys()
                )
            )
        ]["CROPS"]
    )
    area_values = np.array(
        area_selected[
            (area_selected["float_value"] >= 0)
            & (
                area_selected["CROPS"].isin(
                    crop_delineation["Eurostat_code"].value_counts().keys()
                )
            )
        ]["float_value"]
    )

    # if for a crop both the area as well as the main area are available, take the main area
    croplist, valuelist = [], []
    for i, crop in enumerate(area_crops):
        croplist.append(crop)
        if crop in main_area_crops:
            valuelist.append(main_area_values[np.where(main_area_crops == crop)[0][0]])
        else:
            valuelist.append(area_values[i])

    # add all crops for which only the main area is available
    for i, crop in enumerate(main_area_crops):
        if crop not in croplist:
            croplist.append(crop)
            valuelist.append(main_area_values[i])

    """now, if for some of those crops no value is available, check if a crop share is available for the alternative ('Eurostat_code2)"""
    alternative_crops = np.array(
        crop_delineation[~crop_delineation["Eurostat_code"].isin(croplist)][
            "Eurostat_code2"
        ]
        .value_counts()
        .keys()
    )
    main_area_alternative_crops = main_area_selected[
        (main_area_selected["float_value"] >= 0)
        & (main_area_selected["CROPS"].isin(alternative_crops))
    ]["CROPS"].values
    main_area_alternative_values = main_area_selected[
        (main_area_selected["float_value"] >= 0)
        & (main_area_selected["CROPS"].isin(alternative_crops))
    ]["float_value"].values
    area_alternative_crops = area_selected[
        (area_selected["float_value"] >= 0)
        & (area_selected["CROPS"].isin(alternative_crops))
    ]["CROPS"].values
    area_alternative_values = area_selected[
        (area_selected["float_value"] >= 0)
        & (area_selected["CROPS"].isin(alternative_crops))
    ]["float_value"].values

    # if for a crop both the area as well as the main area are available, take the main area
    alt_croplist, alt_valuelist = [], []
    for i, crop in enumerate(area_alternative_crops):
        alt_croplist.append(crop)
        if crop in main_area_alternative_crops:
            alt_valuelist.append(
                main_area_alternative_values[
                    np.where(main_area_alternative_crops == crop)
                ][0]
            )
        else:
            alt_valuelist.append(area_alternative_values[i])

    # add all crops for which only the main area is available
    for i, crop in enumerate(main_area_alternative_crops):
        if crop not in alt_croplist:
            alt_croplist.append(crop)
            alt_valuelist.append(main_area_alternative_values[i])

    # final_croplist=croplist.append(alt_croplist)
    final_croplist = np.concatenate((croplist, alt_croplist))
    final_valuelist = np.concatenate((valuelist, alt_valuelist))
    return final_croplist, final_valuelist

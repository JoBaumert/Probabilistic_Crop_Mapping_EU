# %%
"""FUNCTIONS"""
import pandas as pd
import numpy as np
import jax.numpy as jnp
from joblib import load
import pyarrow as pa
import pyarrow.parquet as pq

# %%


def replace_nanvalues(feature_df, featurename):
    allvalues = feature_df[featurename]
    truevalues = np.where(~feature_df[featurename].isna())
    lat3035 = np.array(list(map(int, feature_df["lat3035"].iloc[truevalues])))
    lon3035 = np.array(list(map(int, feature_df["lon3035"].iloc[truevalues])))
    corrected_values = []

    for v, value in enumerate(allvalues.iloc):
        if v in truevalues[0]:
            corrected_values.append(value)
        else:
            resp_lat, resp_lon = int(feature_df.iloc[v]["lat3035"]), int(
                feature_df.iloc[v]["lon3035"]
            )
            latdist, londist = ((lat3035 - resp_lat) ** 2) ** 0.5, (
                (lon3035 - resp_lon) ** 2
            ) ** 0.5
            distance = (latdist**2) + (londist**2)
            corrected_values.append(
                allvalues[
                    truevalues[0][np.where(distance == distance.min())[0]]
                ].values.mean()
            )
    return corrected_values


def get_cell_predictions(X_scaled, parameters):
    latentvars = np.matmul(np.transpose(X_scaled), parameters)
    """LIMIT VALUES TO AVOID INFINITY VALUES"""
    latentvars=np.where(latentvars>40,40,latentvars)
    latentvars=np.where(latentvars<-40,-40,latentvars)
    latentvars_exp = np.exp(latentvars)
    latentvars_exp_sum = latentvars_exp.sum(axis=1)
    p = (latentvars_exp.transpose() / latentvars_exp_sum).transpose()
    return p


def import_feature_data(features, input_path, nuts_reg):
    # returns dict with feature names as key and feature values as values
    feature_data = {}
    for feature in features:
        try:
            feature_data[feature] = pd.read_csv(
                input_path + feature + "/1kmgrid_" + nuts_reg + ".csv"
            )
        except:
            print(f"no data for {feature} available")

    return feature_data


def get_relevant_grid(grid, year):
    if not (isinstance(year, int) & (isinstance(grid, pd.DataFrame))):
        raise TypeError
    relevant_grid = grid[grid["year"] == year]
    return relevant_grid


def get_effective_cellsize(relevant_grid, agshare, year):
    # grid and agshare must be pd DataFrames
    if not (
        isinstance(year, int)
        & (isinstance(relevant_grid, pd.DataFrame))
        & (isinstance(agshare, pd.DataFrame))
    ):
        raise TypeError

    relevant_grid["area_share"] = relevant_grid["area"] / 1000000

    NUTS_effective_cellsize = relevant_grid[["CELLCODE", "area_share"]]
    NUTS_effective_cellsize = pd.merge(
        NUTS_effective_cellsize,
        agshare[["CELLCODE", "agshare"]],
        on="CELLCODE",
        how="left",
    )
    NUTS_effective_cellsize["effective_share"] = (
        NUTS_effective_cellsize["area_share"] * NUTS_effective_cellsize["agshare"]
    )
    return NUTS_effective_cellsize


def get_lat_lon_from_cellcode(cellcode_array):
    lat = jnp.array([int(x[-4:]) for x in cellcode_array])
    lon = jnp.array([int(x[4:8]) for x in cellcode_array])
    return lat, lon


def get_feature_dataframe(feature_data_dict, feature_names, cellcode):
    feature_df = pd.DataFrame({"CELLCODE": cellcode})
    for f, feature in enumerate(feature_data_dict.keys()):
        feature_df = pd.merge(
            feature_df,
            feature_data_dict[feature][["CELLCODE", feature_names[f]]],
            how="left",
            on="CELLCODE",
        )
    lat3035, lon3035 = get_lat_lon_from_cellcode(cellcode)
    feature_df["lat3035"] = lat3035
    feature_df["lon3035"] = lon3035
    return feature_df


def correct_cell_values(df, col_names, operation):
    for col in col_names:
        corrected_values = operation(df, col)
        df.drop(columns=[col], inplace=True)
        df[col] = corrected_values

    return df


def feature_data_preparation(
    feature_name_dict,
    input_path,
    nuts_reg,
    cellcode,
    scaler_path,
    correct_cols=["elevation", "oc_content"],
):
    features = list(feature_name_dict.keys())
    feature_names = list(feature_name_dict.values())
    feature_data = import_feature_data(features, input_path, nuts_reg)
    feature_df = get_feature_dataframe(
        feature_data_dict=feature_data,
        feature_names=feature_names,
        cellcode=cellcode,
    )
    # if nan values shall be corrected:
    if len(correct_cols) > 0:
        df_corrected = correct_cell_values(feature_df, correct_cols, replace_nanvalues)

    # make sure columns have the right order
    X = df_corrected[feature_names]

    # scale if desired
    if len(scaler_path) > 0:
        # import scaler
        correct_scaler = load(scaler_path)
        X = correct_scaler.transform(X)

    # add vector of ones at first position for intercept
    X_statsmodel = np.insert(np.transpose(X), 0, np.ones(len(X)), axis=0)
    return X_statsmodel


def parameter_preparation(
    params,
    covariance_matrix,
    nofcrops,
    n_of_rand=10000,
    mean_on_first_pos=True,
    insert_zeros_refcrop=True,
):
    means_flattened = np.transpose(params.iloc[:, 1:]).to_numpy().flatten()

    randomly_selected_params = np.random.multivariate_normal(
        means_flattened, covariance_matrix.iloc[2:, 3:], n_of_rand
    )
    if mean_on_first_pos:
        # Set mean as position 0
        randomly_selected_params = np.insert(
            randomly_selected_params, 0, means_flattened, axis=0
        )
    if insert_zeros_refcrop:
        # insert 0s for the reference crop (in this case APPL)
        randomly_selected_params = randomly_selected_params.reshape(
            (randomly_selected_params.shape[0], nofcrops - 1, -1)
        )
        randomly_selected_params = np.insert(
            randomly_selected_params,
            0,
            np.zeros(randomly_selected_params.shape[2]),
            axis=1,
        )
    return randomly_selected_params


def probability_calculation(
    param_matrix, X, nofcrops, nofcells, sample_params=False, nofreps=1
):
    if not sample_params:
        nofreps = 1
    all_probas = []
    for p, param_set in enumerate(param_matrix[:nofreps]):
        probas_cell = get_cell_predictions(
            X, param_set.reshape(nofcrops, -1).transpose()
        )
        probas_cells_reshaped = probas_cell.reshape(nofcells, nofcrops)
        all_probas.append(probas_cells_reshaped)

    return all_probas


def calculate_b(eurostat_path, nuts_reg, year):
    eurostat_data = pd.read_csv(eurostat_path)
    eurostat_relevant = eurostat_data[
        (eurostat_data["NUTS_ID"] == nuts_reg) & (eurostat_data["year"] == year)
    ]
    relevant_crops = np.array(
        eurostat_relevant[eurostat_relevant["area"] > 0].sort_values(
            by="B04_crop_code"
        )["B04_crop_code"]
    )
    b_quantity = jnp.array(
        eurostat_relevant[
            eurostat_relevant["B04_crop_code"].isin(relevant_crops)
        ].sort_values(by="B04_crop_code")["area"]
    )

    b = b_quantity / b_quantity.sum()
    return relevant_crops, b


def calculate_reference_df(priorprob_path, beta_selected, relevant_crops):
    priorprob = pq.read_table(priorprob_path).to_pandas()
    priorprob_relevant = priorprob[priorprob["beta"] == beta_selected]
    # keep only those entries for crops that appear in the region and for cells that have a minimum agricultural area
    """
    relevant_cells = np.array(
        priorprob_relevant[priorprob_relevant["agshare"] > 0]["CELLCODE"]
        .value_counts()
        .keys()
    )
    """
    relevant_cells = np.array(
        priorprob_relevant["CELLCODE"]
        .value_counts()
        .keys()
    )
    priorprob_relevant = priorprob_relevant[
        (priorprob_relevant["CELLCODE"].isin(relevant_cells))
        & (priorprob_relevant["crop"].isin(relevant_crops))
    ]
    C = len(relevant_crops)
    I = len(relevant_cells)
    # generate a reference df to ensure that all cells have the same order and number of crops
    reference_df = pd.DataFrame(
        {
            "CELLCODE": np.repeat(relevant_cells, C),
            "crop": np.tile(relevant_crops, I),
        }
    )
    reference_df = pd.merge(
        reference_df,
        priorprob_relevant,
        how="left",
        on=["CELLCODE", "crop"],
    )
    return reference_df

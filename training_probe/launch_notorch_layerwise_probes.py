import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
import json
import numpy as np
import zarr
from tqdm import tqdm
from argparse import ArgumentParser

from face_plm.probes.data import SEQUENCE_DATA_DF_PATH, SPLIT_INFO_JSON, ANKH_FULL_LAYER_ZARR_PATH


def train_random_forest(X_train, y_train, X_test, y_test):
    # training the random forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    # Calculating the R^2 value
    r_sq = model.score(X_test, y_test)
    # Calulating the spearman correlation
    spearman_corr, _ = stats.spearmanr(y_test, model.predict(X_test))
    # Calculating the pearson correlation
    pearson_corr, _ = stats.pearsonr(y_test, model.predict(X_test))
    # Calculating the RMSE value
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    return model, r_sq, rmse, spearman_corr, pearson_corr


def collate_results(results_dict_r2, results_dict_rmse, results_dict_spearman, results_dict_pearson):
    model = ['_'.join(x.split("_")[:-2]) for x in results_dict_r2.keys()]
    model_type = [x.split("_")[-2] for x in results_dict_r2.keys()]
    split = [x.split("_")[-1] for x in results_dict_r2.keys()]
    r2 = list(results_dict_r2.values())
    rmse = list(results_dict_rmse.values())
    spearman = list(results_dict_spearman.values())
    pearson = list(results_dict_pearson.values())
    # saving everything to a dataframe
    results_df = pd.DataFrame({'model': model,
                               'probe': model_type,
                               'split': split,
                               'r2': r2,
                               'rmse': rmse,
                               'spearman': spearman,
                               'pearson': pearson})
    return results_df


def generate_metric_dict(encoder_embeddings, data_df, split_info):
    metric_dict = {}
    for layer in tqdm(encoder_embeddings.keys()):
        print(f"Layer: {layer}")
        X = encoder_embeddings[layer]['mean']
        y = data_df['log10_kcat'].values
        orgs = data_df['org_name'].values
        r2_list = []
        rmse_list = []
        spearman_corr_list = []
        pearson_corr_list = []
        for split in split_info:
            train_orgs = split_info[split]['train']
            test_orgs = split_info[split]['test']
            train_inds = np.where(np.isin(orgs, train_orgs))[0]
            test_inds = np.where(np.isin(orgs, test_orgs))[0]
            X_train = X[train_inds]
            y_train = y[train_inds]
            X_test = X[test_inds]
            y_test = y[test_inds]
            _, r_sq, rmse, spearman_corr, pearson_corr = train_random_forest(X_train, y_train, X_test, y_test)
            r2_list.append(r_sq)
            rmse_list.append(rmse)
            spearman_corr_list.append(spearman_corr)
            pearson_corr_list.append(pearson_corr)
        metric_dict[layer] = {'r2': np.array(r2_list),
                            'rmse': np.array(rmse_list),
                            'spearman_corr': np.array(spearman_corr_list),
                            'pearson_corr': np.array(pearson_corr_list)}
    return metric_dict


def main():
    # Setting up the argparse
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Reading in the layerwise embeddings
    zarr_file = ANKH_FULL_LAYER_ZARR_PATH
    zarr_root_zs = zarr.open(zarr_file, mode="r")
    encoder_embeddings_zs = {}
    seqs = list(zarr_root_zs['ankh-base'].keys())
    for layer in tqdm(range(zarr_root_zs['ankh-base'][seqs[0]].shape[0])):
        encoder_embeddings_zs[layer] = {}
        mean_embeddings = []
        max_embeddings = []
        min_embeddings = []
        for seq in seqs:
            mean_embeddings.append(np.mean(np.array(zarr_root_zs['ankh-base'][seq][layer]), axis=0))
            max_embeddings.append(np.max(np.array(zarr_root_zs['ankh-base'][seq][layer]), axis=0))
            min_embeddings.append(np.min(np.array(zarr_root_zs['ankh-base'][seq][layer]), axis=0))
        encoder_embeddings_zs[layer]['mean'] = np.array(mean_embeddings)
        encoder_embeddings_zs[layer]['max'] = np.array(max_embeddings)
        encoder_embeddings_zs[layer]['min'] = np.array(min_embeddings)

    # Reading in the sequence data
    csv_file = SEQUENCE_DATA_DF_PATH
    data_df = pd.read_csv(csv_file)
    data_df = data_df.sort_values(by='sequence')

    # Readin in the split information
    split_info_file = SPLIT_INFO_JSON
    with open(split_info_file, 'r') as f:
        split_info = json.load(f)

    # Generating the metric dictionary for the layerwise embeddings
    metric_dict_zs = generate_metric_dict(encoder_embeddings_zs, data_df, split_info)   
    layers_list_zs = np.hstack([np.repeat(layer, 5) for layer in metric_dict_zs.keys()])
    r2_list_zs = np.hstack([metric_dict_zs[layer]['r2'] for layer in metric_dict_zs.keys()])
    rmse_list_zs = np.hstack([metric_dict_zs[layer]['rmse'] for layer in metric_dict_zs.keys()])
    spearman_corr_list_zs = np.hstack([metric_dict_zs[layer]['spearman_corr'] for layer in metric_dict_zs.keys()])
    pearson_corr_list_zs = np.hstack([metric_dict_zs[layer]['pearson_corr'] for layer in metric_dict_zs.keys()])

    # Saving the results to a csv
    out_dir = args.output_dir
    if out_dir.endswith('/'):
        out_dir = out_dir[:-1]
    result_df_plot_zs = pd.DataFrame({'layer': layers_list_zs, 'r2': r2_list_zs, 'rmse': rmse_list_zs, 'spearman_corr': spearman_corr_list_zs, 'pearson_corr': pearson_corr_list_zs})
    result_df_plot_zs.to_csv(f'{out_dir}/random_forest_ankhbase_layerwise_results.csv', index=False)


if __name__ == "__main__":
    main()

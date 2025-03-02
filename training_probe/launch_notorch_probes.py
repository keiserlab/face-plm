import pandas as pd
import scipy.stats as stats
# importing regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import json
import numpy as np
import zarr
from tqdm import tqdm
from argparse import ArgumentParser

from face_plm.probes.data import EMBEDDING_ZARR_PATH, SEQUENCE_DATA_DF_PATH, SPLIT_INFO_JSON

def open_zarr_file(zarr_file):
    zarr_root = zarr.open(zarr_file, mode='r')
    return zarr_root

def get_aggregated_embedding_from_zarr(zarr_file, aggregation='mean'):
    embed_dict = {}
    zarr_root = zarr.open(zarr_file, mode='r')
    for model in zarr_root.keys():
        embed_dict[model] = {}
        for seq in zarr_root[model].keys():
            full_embed = np.array(zarr_root[model][seq])
            if aggregation == 'mean':
                embed_dict[model][seq] = np.mean(full_embed, axis=0)
            elif aggregation == 'min':
                embed_dict[model][seq] = np.min(full_embed, axis=0)
            elif aggregation == 'max':
                embed_dict[model][seq] = np.max(full_embed, axis=0)
            elif aggregation is None:
                embed_dict[model][seq] = full_embed
    return embed_dict


def train_linear_regression(X_train, y_train, X_test, y_test):
    # training the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Calculating the R^2 value
    r_sq = model.score(X_test, y_test)
    # Calculating the RMSE value
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    # Calulating the spearman correlation
    spearman_corr, _ = stats.spearmanr(y_test, model.predict(X_test))
    # Calculating the pearson correlation
    pearson_corr, _ = stats.pearsonr(y_test, model.predict(X_test))
    return model, r_sq, rmse, spearman_corr, pearson_corr


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


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    # training the gradient boosting model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    # Calculating the R^2 value
    r_sq = model.score(X_test, y_test)
    # Calculating the RMSE value
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    # Calulating the spearman correlation
    spearman_corr, _ = stats.spearmanr(y_test, model.predict(X_test))
    # Calculating the pearson correlation
    pearson_corr, _ = stats.pearsonr(y_test, model.predict(X_test))
    print(r_sq, rmse, spearman_corr, pearson_corr)
    return model, r_sq, rmse, spearman_corr, pearson_corr


def train_models(embed_dict, data_df, split_info, y_col='log10_kcat'):
    model_dict = {}
    r2_dict = {}
    rmse_dict = {}
    spearman_dict = {}
    pearson_dict = {}
    # iterating over the models
    for model in tqdm(embed_dict.keys()):
        print("Model: ", model)
        for split in tqdm(split_info.keys()):
            organisms_in_split_train = split_info[split]["train"]
            organisms_in_split_test = split_info[split]["test"]
            train_df = data_df[data_df['org_name'].isin(organisms_in_split_train)]
            test_df = data_df[data_df['org_name'].isin(organisms_in_split_test)]
            train_seqs = train_df['sequence'].values
            test_seqs = test_df['sequence'].values
            train_embeds = np.array([embed_dict[model][seq] for seq in train_seqs])
            test_embeds = np.array([embed_dict[model][seq] for seq in test_seqs])
            X_train = train_embeds
            X_test = test_embeds
            y_train = train_df[y_col].values
            y_test = test_df[y_col].values
            # training the linear regression model
            m, r2, rmse, spearman, pearson = train_linear_regression(X_train, y_train, X_test, y_test)
            model_dict[model + "_lr_" + split] = m
            r2_dict[model + "_lr_" + split] = r2
            rmse_dict[model + "_lr_" + split] = rmse
            spearman_dict[model + "_lr_" + split] = spearman
            pearson_dict[model + "_lr_" + split] = pearson

            m, r2, rmse, spearman, pearson = train_random_forest(X_train, y_train, X_test, y_test)
            model_dict[model + "_rf_" + split] = m
            r2_dict[model + "_rf_" + split] = r2
            rmse_dict[model + "_rf_" + split] = rmse
            spearman_dict[model + "_rf_" + split] = spearman
            pearson_dict[model + "_rf_" + split] = pearson

            m, r2, rmse, spearman, pearson = train_gradient_boosting(X_train, y_train, X_test, y_test)
            model_dict[model + "_gb_" + split] = m
            r2_dict[model + "_gb_" + split] = r2
            rmse_dict[model + "_gb_" + split] = rmse
            spearman_dict[model + "_gb_" + split] = spearman
            pearson_dict[model + "_gb_" + split] = pearson
            
    return model_dict, r2_dict, rmse_dict, spearman_dict, pearson_dict


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


def main():
    # Setting up the argparse
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    embed_dict_mean = get_aggregated_embedding_from_zarr(EMBEDDING_ZARR_PATH)
    embed_dict_max = get_aggregated_embedding_from_zarr(EMBEDDING_ZARR_PATH, aggregation='max')
    embed_dict_min = get_aggregated_embedding_from_zarr(EMBEDDING_ZARR_PATH, aggregation='min')

    # loading the sequence data
    data_df = pd.read_csv(SEQUENCE_DATA_DF_PATH)
    # getting the split info
    with open(SPLIT_INFO_JSON) as f:
        split_info = json.load(f)
    
    organisms = data_df['org_name'].apply(lambda x: ' '.join(x.split('_'))).values
    organisms = [org[0].upper() + org[1:] for org in organisms]

    _, r2_dict_mean, rmse_dict_mean, spearman_dict_mean, pearson_dict_mean = train_models(embed_dict_mean, data_df, split_info)
    _, r2_dict_max, rmse_dict_max, spearman_dict_max, pearson_dict_max = train_models(embed_dict_max, data_df, split_info)
    _, r2_dict_min, rmse_dict_min, spearman_dict_min, pearson_dict_min = train_models(embed_dict_min, data_df, split_info)

    # collating the results
    mean_df = collate_results(r2_dict_mean, rmse_dict_mean, spearman_dict_mean, pearson_dict_mean)
    max_df = collate_results(r2_dict_max, rmse_dict_max, spearman_dict_max, pearson_dict_max)
    min_df = collate_results(r2_dict_min, rmse_dict_min, spearman_dict_min, pearson_dict_min)

    # saving the results
    out_dir = args.output_dir
    if out_dir.endswith("/"):
        out_dir = out_dir[:-1]
    mean_df.to_csv(args.output_dir + "/mean_results.csv", index=False)
    max_df.to_csv(args.output_dir + "/max_results.csv", index=False)
    min_df.to_csv(args.output_dir + "/min_results.csv", index=False)


if __name__ == "__main__":
    main()

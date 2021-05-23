#!/usr/bin/env python
# coding: utf-8

import os
from multiprocessing import Pool
import json

import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import h5py


QUERY_COUNT = 1000
ITEM_COUNT = 10 ** 6
GT_TOP_LEN = 10000


def read_fvecs(file_name):
    a = np.fromfile(file_name, dtype="int32")
    dim = a[0]
    return a.view("float32").reshape((-1, dim + 1))[:,1:]

def normalize(a):
    vec_lengths = np.sqrt(np.power(a, 2).sum(axis=1, keepdims=True))
    assert np.all(vec_lengths > 1e-9)
    return a / vec_lengths

def calc_pairwise_relevance(i, q):
    return 1 - pairwise_distances(i, q, metric="cosine").astype("float32")

def to_file(data, file_path):
    if os.path.isfile(file_path):
        return
    data.tofile(file_path)

def prepare_data(dataset, dim, items, train_queries, test_queries, masked_coord_counts,
                 verbose=False, recalc=False):
    assert train_queries.shape == (QUERY_COUNT, dim)
    assert test_queries.shape == (QUERY_COUNT, dim)
    assert items.shape == (ITEM_COUNT, dim)
    
    np.random.seed(0)
    coordinate_permutation = np.random.permutation(dim)
    items = normalize(items)[:,coordinate_permutation]
    train_queries = normalize(train_queries)[:,coordinate_permutation]
    test_queries = normalize(test_queries)[:,coordinate_permutation]

    os.makedirs("data/{}/data/model_scores".format(dataset), exist_ok=True)
    train_queries.tofile("data/{}/data/train_queries.bin".format(dataset))
    test_queries.tofile("data/{}/data/test_queries.bin".format(dataset))
    items.tofile("data/{}/data/items.bin".format(dataset))

    query_cov = train_queries.T.dot(train_queries)
    item_transformation = sqrtm(query_cov)
    transformed_items = items.dot(item_transformation).astype("float32")
    transformed_items.tofile("data/{}/data/transformed_items.bin".format(dataset))
    
    if verbose and masked_coord_counts:
        print("compute test scores for models with masked coordinates") 
    for masked_count in masked_coord_counts:
        masked_scores_path = "data/{}/data/model_scores/masked_{}_test_scores.bin".format(
            dataset, masked_count
        )
        if os.path.isfile(masked_scores_path) and not recalc:
            continue
        masked_model_test_scores = calc_pairwise_relevance(
            items[:,:dim - masked_count],
            test_queries[:,:dim - masked_count]
        )
        masked_model_test_scores.tofile(masked_scores_path)
        del masked_model_test_scores
    
    gt_path = "data/{}/data/model_scores/groundtruth_test.bin".format(dataset)
    if not os.path.isfile(gt_path) or recalc:
        gt_test_scores = calc_pairwise_relevance(items, test_queries)
        gt_test_scores.tofile("data/{}/data/model_scores/gt_test_scores.bin".format(dataset))
        if verbose:
            print("Calc ground truth nearest neighbors for {}".format(dataset))
            gt = (-gt_test_scores).argsort(axis=0)[:GT_TOP_LEN,:].T.astype("int32")
            gt.tofile()


def prepare_glove(dim, masked_coord_counts, verbose=False, recalc=False):
    if verbose:
        print("Prepare glove-{} data".format(dim))
    # the hdf5 file is taken from here: https://github.com/erikbern/ann-benchmarks
    with h5py.File("data/glove_{dim}/glove-{dim}-angular.hdf5".format(dim=dim), "r") as f:
        glove_base = f["train"][:]
        glove_query = f['test'][:]
    
    np.random.seed(0)
    np.random.shuffle(glove_base)
    np.random.shuffle(glove_query)

    train_queries = glove_query[:QUERY_COUNT]
    test_queries = glove_query[QUERY_COUNT: 2 * QUERY_COUNT]
    items = glove_base[:ITEM_COUNT]

    prepare_data("glove_{}".format(dim), dim, items, train_queries, test_queries,
                 masked_coord_counts, verbose, recalc)

def prepare_sift(masked_coord_counts, verbose=False, recalc=False):
    if verbose:
        print("Prepare sift data")
    
    # The data is taken from here:
    # https://github.com/erikbern/ann-benchmarks
    with h5py.File("data/sift/sift-128-euclidean.hdf5", "r") as f:
        print(f.keys())
        sift_base = f["train"][:]
        sift_query = f['test'][:]
    assert sift_base.shape == (ITEM_COUNT, 128)
    np.random.seed(0)
    np.random.shuffle(sift_query)
    train_queries = sift_query[:QUERY_COUNT]
    test_queries = sift_query[QUERY_COUNT: 2 * QUERY_COUNT]
    items = sift_base
    prepare_data("sift", 128, items, train_queries, test_queries, masked_coord_counts, verbose, recalc)

def prepare_deep96(masked_coord_counts, verbose=False, recalc=False):
    if verbose:
        print("prepare deep-96 data")
    
    # The data is taken from here: http://sites.skoltech.ru/compvision/noimi/
    deep_base = read_fvecs("data/deep_96/deep10M.fvecs")
    deep_query = read_fvecs("data/deep_96/deep1B_queries.fvecs")
    assert 2 * QUERY_COUNT <= deep_query.shape[0]

    np.random.seed(0)
    np.random.shuffle(deep_base)
    np.random.shuffle(deep_query)
    
    train_queries = deep_query[:QUERY_COUNT]
    test_queries = deep_query[QUERY_COUNT: 2 * QUERY_COUNT]    
    items = deep_base[:ITEM_COUNT]

    prepare_data("deep1M", 96, items, train_queries, test_queries, masked_coord_counts, verbose, recalc)

def prepare_deep256(masked_coord_counts, verbose=False, recalc=False):
    if verbose:
        print("prepare deep-256 data")

    # The data is from here:
    # http://sites.skoltech.ru/compvision/projects/aqtq/
    items = read_fvecs("data/deep_256/deep1M_base.fvecs")
    learn = read_fvecs("data/deep_256/deep1M_learn.fvecs")
    test_queries = read_fvecs("data/deep_256/deep1M_queries.fvecs")

    assert items.shape == (ITEM_COUNT, 256)
    assert test_queries.shape == (QUERY_COUNT, 256)
    
    np.random.seed(0)
    np.random.shuffle(learn)
    train_queries = learn[:QUERY_COUNT]

    prepare_data("deep256", 256, items, train_queries, test_queries, masked_coord_counts, verbose, recalc)


build_graph_cmd_template = (
    "./RPG --mode base "
    "--baseSize {baseSize} "
    "--trainQueries {featuresDimension} "
    "--base data/{dataset}/data/{features}.bin "
    "--outputGraph {graphPath} "
    "--relevanceVector {usedDimensions} "
    "--efConstruction 1000 --M {M} "
)

def build_graph(
    dataset, graph_path, features, dimension,
    used_dimensions=None, degree=8, base_size=ITEM_COUNT,
    recalc=False, verbose=False
):
    if os.path.isfile(graph_path) and not recalc:
        return
    if used_dimensions is None:
        used_dimensions = dimension
    cmd = build_graph_cmd_template.format(
        baseSize=base_size,
        featuresDimension=dimension,
        dataset=dataset,
        features=features,
        graphPath=graph_path,
        usedDimensions=used_dimensions,
        M=degree
    )
    if verbose:
        print(cmd)
    os.system(cmd)


search_cmd_template = (
    "./RPG --mode query --baseSize {baseSize} --querySize 1000"
    " --query data/{dataset}/data/model_scores/{scores}.bin --inputGraph {inputGraph}"
    " --efSearch {efSearch} --topK {topK} --output data/{dataset}/{searchResultFile}.txt" +
    " --gtQueries 1000 --gtTop {} ".format(GT_TOP_LEN) +
    "--groundtruth data/{dataset}/data/model_scores/{gtFile}"
)

def run_cmd(cmd):
    return os.popen(cmd).read()

def run_search(graph_path, scores_file, ef_ticks, dataset,
               recall_top_len=5, result_file=None,
               base_size=ITEM_COUNT, gt_file="groundtruth_test.bin",
               n_threads=8):
    if result_file is None:
        result_file = "result"
    else:
        assert len(ef_ticks) == 1

    commands = []
    for ef in ef_ticks:
        commands.append(search_cmd_template.format(
            baseSize=base_size,
            dataset=dataset,
            scores=scores_file,
            inputGraph=graph_path,
            efSearch=ef,
            topK=recall_top_len,
            searchResultFile=result_file,
            gtFile=gt_file
        ))
    pool = Pool(processes=n_threads)
    command_outputs = pool.map(run_cmd, commands)

    search_results = {"relevance": [], "recall": [], "evals": []}
    for i, cmd_out in enumerate(command_outputs):
        res = {}
        stat_prefixes = {
            "relevance": "Average relevance: ",
            "recall": "Recall@{}: ".format(recall_top_len),
            "evals": "Average number of model computations: "
        }
        for line in cmd_out.split("\n"):
            line = line.strip()
            for stat_name, prefix in stat_prefixes.items():
                if line.startswith(prefix):
                    res[stat_name] = float(line[len(prefix):])
        
        if all(key in res for key in search_results):
            for key in search_results:
                search_results[key].append(res[key])
        else:
            print("missed result for {} efSearch {}.".format(graph_path, ef_ticks[i]))
            print(commands[i])
            print(cmd_out)
                
    search_results["efSearch"] = [int(t) for t in ef_ticks]
    return search_results


def logspace(start, stop, count, include_ends=True):
    cnt_ = count if include_ends else count + 2
    seq = np.unique(np.exp(
        np.linspace(np.log(start), np.log(stop), cnt_)
    ).astype("int"))
    if include_ends:
        return seq
    return seq[1:-1]

def read_txt(file_name, expected_shape=None):
    data = []
    with open(file_name) as fin:
        for line in fin:
            data.append([int(w) for w in line.split()])
    row_len = len(data[0])
    assert all(len(l) == row_len for l in data)
    if expected_shape is not None:
        assert expected_shape == (len(data), row_len)
    return data


def calc_eval_recall_curve(approximate_top, gt_top, recall_top_len=5):
    assert gt_top.shape == (QUERY_COUNT, recall_top_len)
    
    gt_tops = [set(query_top) for query_top in gt_top]
    recalls = []
    found_count = 0
    top_len = len(approximate_top[0])
    for i in range(top_len):
        for query_id in range(QUERY_COUNT):
            if approximate_top[query_id][i] in gt_tops[query_id]:
                found_count += 1
        recalls.append(found_count / (QUERY_COUNT * recall_top_len))
    evals = list(range(1, top_len + 1))
    return {"evals": evals, "recall": recalls}


def run_experiment_with_coordinate_masking(
    dataset, dimension, masked_coord_counts, recall_top_len=5,
    ef_ticks=logspace(10, 3000, 31), n_search_threads=1,
    recalc_graphs=False, recalc_search=False, verbose=False
):
    if recalc_graphs:
        recalc_search = True
    result_path = "data/{}/coord_masking_recall@{}.json".format(dataset, recall_top_len)
    if not os.path.isfile(result_path) or recalc_search:
        results = {}
    else:
        with open(result_path) as fin:
            results = json.load(fin)
    
    for label, features in [
        ("rpg", "transformed_items"),
        ("hnsw", "items")
    ]:
        if label in results and not recalc_search:
            continue
        graph_path = "data/{}/{}.hnsw".format(dataset, label)
        build_graph(dataset, graph_path, features, dimension, recalc=recalc_graphs, verbose=verbose)
        results[label] = run_search(
            graph_path, "gt_test_scores", ef_ticks, dataset,
            recall_top_len=recall_top_len, n_threads=n_search_threads
        )
    
    for masked_count in masked_coord_counts:
        label = "{}_masked_coords+rerank".format(masked_count)
        if label in results and not recalc_search:
            continue
        graph_path = "data/{}/{}_masked_coords.hnsw".format(dataset, masked_count)
        build_graph(
            dataset, graph_path, "items", dimension,
            dimension - masked_count, recalc=recalc_graphs, verbose=verbose
        )
        search_result_path = "search_result_masked_{}".format(masked_count)
        run_search(graph_path, "masked_{}_test_scores".format(masked_count),
                   [GT_TOP_LEN], dataset, GT_TOP_LEN, search_result_path, n_threads=1)

        approximate_top = read_txt(
            "data/{}/{}.txt".format(dataset, search_result_path),
            (QUERY_COUNT, GT_TOP_LEN)
        )
        gt = np.fromfile(
            "data/{}/data/model_scores/groundtruth_test.bin".format(dataset),
            dtype="int32"
        ).reshape((QUERY_COUNT, GT_TOP_LEN))
        gt = gt[:,:recall_top_len]
        results[label] = calc_eval_recall_curve(approximate_top, gt, recall_top_len)
    with open(result_path) as fout:
        json.dump(results, fout, indent=4)
    return results


def plot_chosen_results(results, keys=None, xlim=None, ylim=None,
                        hlines=None, vlines=None, x_log_scale=False):
    plt.figure(figsize=(10, 10))
    plt.xlabel("evals")
    plt.ylabel("recall")
    if keys is None:
        keys = results.keys()
    if xlim is not None:
        plt.xlim(xlim)
        if hlines is not None:
            plt.hlines(hlines, *xlim, color="grey", alpha=0.5)
    if ylim is not None:
        plt.ylim(ylim)
        if vlines is not None:
            plt.vlines(vlines, *ylim, color="grey", alpha=0.5)
    if x_log_scale:
        plt.xscale("log")
    
    for key in keys:
        assert key in results
        r = results[key]
        x = r["evals"]
        y = r["recall"]
        plt.plot(x, y, label=key)
        if len(x) < 100:
            plt.scatter(x, y, s=3)
    plt.legend()
    plt.show()


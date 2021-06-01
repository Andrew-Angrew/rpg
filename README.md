# Relevance Proximity Graphs
A supplementary code for Relevance Proximity Graphs for Efficient Relevance Retrieval. This implementation is substantially based on HNSW code [https://github.com/nmslib/hnswlib](https://github.com/nmslib/hnswlib).

# What does it do?
It performs a fast relevance retrieval over a large-scale database with a given relevance function. The relevance function is defined on the `(query, item)` pairs without predefined similarity measure between two items or two queries.

# What do I need to run it?
* Use any popular 64-bit Linux operating system
  * Tested on Ubuntu16.04, should work fine on most linux x64 and even MacOS;
* Run the Makefile to compile c++ sources
  * ```sudo apt-get install --upgrade gcc g++ libstdc++6 make```
  * ```make```
  * ```chmod u+x download.py compute_scores.py```
* To work with pretrained GBDT model you also need Python packages from `requirements.txt`  

# How do I run it?
1. If you want to perform preliminary experiments from section 5.1 you should first dowload the datasets:
 * GloVe-25: put file "glove-25-angular.hdf5" form https://github.com/erikbern/ann-benchmarks to folder data/glove_25.
 * SIFT: put file "sift-128-euclidean.hdf5" form https://github.com/erikbern/ann-benchmarks to folder data/sift
 * Deep (96): put files "deep10M.fvecs" and "deep1B_queries.fvecs" from http://sites.skoltech.ru/compvision/noimi/ to folder data/deep_96
 * Deep (256): put files "deep1M_base.fvecs", "deep1M_learn.fvecs" and "deep1M_queries.fvecs" from http://sites.skoltech.ru/compvision/projects/aqtq/ to folder data/deep_256

  Then run appropriate notebook with name of the kind "preliminary_experiment_*".

2. If you want to evaluate RPG on Collections and Video datasets, then download them using `data/download.py` script. If you want to download only precomputed relevance function scores, run:
* `cd data`
* `python3 download.py all score`
3. If you you want to download our precomputed GBDT models and the set of input features, run (note that it requires about 1 Tb of disk space and that you need only precomputed scores to reproduce our results):
* `python3 download.py all model`
* You can run these models using the demo-script `data/compute_scores.py`
4. After the data was prepared you need to build a graph and run the search algorithm:
* To perform experiments on SIFT dataset run:
   * `./RPG --mode base --baseSize 1000000 --trainQueries 1000 --base data/sift/train_sift.bin  --outputGraph data/sift/graph.out --relevanceVector 100 --efConstruction 1000 --M 8`
   * `./RPG --mode query --baseSize 1000000 --querySize 1000 --query data/sift/test_sift.bin --inputGraph data/sift/graph.out --efSearch 300 --topK 5 --output data/sift/result.txt --gtQueries 1000 --gtTop 100 --groundtruth data/sift/groundtruth_sift.bin`
* To perform experiments on Collections dataset run:
   * `./RPG --mode base --baseSize 1000000 --trainQueries 1000 --base data/collections/data/model_scores/scores_train.bin --outputGraph data/collections/graph.out --relevanceVector 1000 --efConstruction 1000 --M 8`
   * `./RPG --mode query --baseSize 1000000 --querySize 1000 --query data/collections/data/model_scores/scores_test.bin --inputGraph data/collections/graph.out --efSearch 300 --topK 5 --output data/collections/result.txt --gtQueries 1000 --gtTop 100 --groundtruth data/collections/data/model_scores/groundtruth.bin`
* To perform experiments on Video dataset run:
   * `./RPG --mode base --baseSize 1000000 --trainQueries 1000 --base data/video/data/model_scores/scores_train.bin --outputGraph data/video/graph.out --relevanceVector 1000 --efConstruction 1000 --M 8`
   * `./RPG --mode query --baseSize 1000000 --querySize 1000 --query data/video/data/model_scores/scores_test.bin --inputGraph data/video/graph.out --efSearch 300 --topK 5 --output data/video/result.txt --gtQueries 1000 --gtTop 100 --groundtruth data/video/data/model_scores/groundtruth.bin`
5. You can vary `efSearch` to achieve higher recall values.
6. If you want to reproduce synthetic experiment from section 5.5 run notebook "synthetic_with_shifted_queries.ipynb" 

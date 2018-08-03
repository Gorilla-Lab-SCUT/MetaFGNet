## Sample Selection for MetaFGNet
We provide three sample selection strategies based on our proposed image ranking scores. 
* **score_threshold:** Set a score threshold by hand empirically.
* **ratio_threshold:** Select a specific percentage of images from the source dataset.
* **topk:** Select the images with top K scores from the source dataset.

We adopt the ratio_threshold in all the experiments. The different ratios for different dataset can be found in the paper.

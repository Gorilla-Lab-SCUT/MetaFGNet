## Fine-Grained Visual Categorization using Meta-Learning Optimization with Sample Selection of Auxiliary Data
Paper available at: https://arxiv.org/abs/1807.10916 

The code is used to train a MetaFGNet with L-Bird Subset and CUB-200-2011 dataset as the source and target dataset respectively.
The extention to other source and target datasets is direct.

It concludes five parts:
 1. **L_Bird_pretrain:** Train a model for the classification task of L-Bird Subset based on the model that pre-trained on the ImageNet.
 2. **MetaFGNet_without_Sample_Selection:** Train the MetaFGNet without sample selection of L_Bird Subset
 3. **Sample_Selection:** Select the target-related samples from L_Bird Subset
 4. **MetaFGNet_with_Sample_Selection:** Train the MetaFGNet with sample selection of L_Bird Subset
 5. **Fine_tune_for_final_results:** Fine-tune the MetaFGNet model on the target dataset for better and final result.
 
We split the whole program into five parts for better understanding and reuse. 
* The 'regularized meta-learning objective' is implemented in the second part and the fourth part.
* The proposed sample selection method is implemented in the third part.

We also provide some intermediate results for quickly implementation and verification. They can be downloaded from:
* [百度网盘](https://pan.baidu.com/s/19VUOsrDJdIZ6dAGs2sZIGg, "Metafgnet")
* [Google Drive](https://drive.google.com/drive/folders/1mSMcQz2jOA9I0Ydn3-b6lmBkYmlKh-6n?usp=sharing, "Metafgnet")

This code is completed with the cooperation of [Hui Tang](https://github.com/huitangtang, "Hui Tang").

If you have any questions, feel free to contact me at: zhang.yabin@mail.scut.edu.cn.
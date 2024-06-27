# Data-Augmented-BERT-pair

This page contains the code used for performing data augmentation technique to low-frequency categories, with the goal to enhance the performance of BERT-pair. The data augmentation techniques performed are keyboard augmentation, EDA and adjusted-EDA, back-translation and mixup.

## Step 1:
The benchmark model used is based on BERT-pair-NLI-M. See https://github.com/HSLCY/ABSA-BERT-pair for more information and the instructions for constructing the base model. For the purpose of this research, the SemEval 2016 data set for restaurant reviews is used.

## Step 2:
The several data augmentation techniques can be performed in any order. For each technique, the following steps need to be followed:
1. Create a copy of the original training data set, to add the new sentences to.
2. Perform the data augmentation techniques by running `augment_bert_pair`. For replication purposes, choose low_freq_min = 1, low_freq_max = 50 and k = 4, 8, or 12.
3. Fine-tune the augmented data set just like the benchmark model. In `processor.py`, the code needs to be set to the right training data file and create .

## Step 3:
Run `evaluation.py` for each created train set, with the following command: `python evaluation.py --task_name semeval_NLI_M --pred_data_dir results/base/test_ep_4.txt`, where the `--pred_data_dir` is the directory leading to the appropriate ouput file.

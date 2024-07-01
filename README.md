# Data-Augmented-BERT-pair

This page contains the code used for performing data augmentation technique to low-frequency categories, with the goal to enhance the performance of BERT-pair. The data augmentation techniques performed are keyboard augmentation, EDA and adjusted-EDA, back-translation and mixup. All code run in this research was done in PyCharm, using Python 3.8. The necessary packages can be downloaded using `pip install torch numpy nltk scikit-learn nlpaug sentencepiece openpyxl transformers sacremoses tensorflow'.

## Step 1:
The benchmark model used is based on BERT-pair-NLI-M. See https://github.com/HSLCY/ABSA-BERT-pair for more information and the instructions for constructing the base model. For the purpose of this research, the SemEval 2016 data set for restaurant reviews is used. The benchmark is fine-tuned using the following command in the terminal:

  'python run_classifier_TABSA.py \
      --task_name semeval_NLI_M \
      --data_dir data/semevaldata/bert-pair \
      --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
      --bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
      --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
      --eval_test \
      --do_lower_case \
      --max_seq_length 512 \
      --train_batch_size 24 \
      --learning_rate 2e-5 \
      --num_train_epochs 4.0 \
      --output_dir results/benchmark \
      --seed 42'
      
## Step 2:
The several data augmentation techniques can be performed in any order. For each technique, the following steps need to be followed:
1. Create a copy of the original training data set, to add the new sentences to.
2. Perform the data augmentation techniques by running `augment_bert_pair.py`. For replication purposes, choose low_freq_min = 1, low_freq_max = 50 and k = 4, 8, or 12.
3. Fine-tune the augmented data set just like the benchmark model. In `processor.py`, the code needs to be set to the right training data file, and the `output_dir' needs to be changed to the correct output directory (it is wise to create a new one for each data augmentation variant). 

## Step 3:
Run `evaluation.py` for each created training set, with the following command: `python evaluation.py --task_name semeval_NLI_M --pred_data_dir results/base/test_ep_4.txt`, where the `--pred_data_dir` is the directory leading to the appropriate ouput file (such as `results/eda/eda_4/test_ep_4.txt').

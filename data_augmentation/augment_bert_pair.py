from augmenters import (keyboard_augmentation,
                        copy_csv_file,
                        backtranslation,
                        easy_data_augmentation)
from mixup import mixup
from eda_adj import easy_data_augmentation_adjusted

# Create new files and implement keyboard augmentation
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/ka/ka_train_NLI_16_4.csv')
keyboard_augmentation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                      '../data/semevaldata/bert-pair/ka/ka_train_NLI_16_4.csv',
                      low_freq_min=1, low_freq_max=50, k=4)
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/ka/ka_train_NLI_16_8.csv')
keyboard_augmentation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                      '../data/semevaldata/bert-pair/ka/ka_train_NLI_16_8.csv',
                      low_freq_min=1, low_freq_max=50, k=8)
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/ka/ka_train_NLI_16_12.csv')
keyboard_augmentation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                      '../data/semevaldata/bert-pair/ka/ka_train_NLI_16_12.csv',
                      low_freq_min=1, low_freq_max=50, k=12)

# Create new files and implement EDA
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/eda/eda_train_NLI_16_4.csv')
easy_data_augmentation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                       '../data/semevaldata/bert-pair/eda/eda_train_NLI_16_4.csv',
                       low_freq_min=1, low_freq_max=50, k=4)
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/eda/eda_train_NLI_16_8.csv')
easy_data_augmentation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                       '../data/semevaldata/bert-pair/eda/eda_train_NLI_16_8.csv',
                       low_freq_min=1, low_freq_max=50, k=8)
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/eda/eda_train_NLI_16_12.csv')
easy_data_augmentation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                       '../data/semevaldata/bert-pair/eda/eda_train_NLI_16_12.csv',
                       low_freq_min=1, low_freq_max=50, k=12)

# Create new files and implement adjusted EDA
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/eda/adj/adj_train_NLI_16_4.csv')
easy_data_augmentation_adjusted('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                                '../data/semevaldata/bert-pair/eda/adj/adj_train_NLI_16_4.csv',
                                low_freq_min=1, low_freq_max=50, k=4)
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/eda/adj/adj_train_NLI_16_8.csv')
easy_data_augmentation_adjusted('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                                '../data/semevaldata/bert-pair/eda/adj/adj_train_NLI_16_8.csv',
                                low_freq_min=1, low_freq_max=50, k=8)
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/eda/adj/adj_train_NLI_16_12.csv')
easy_data_augmentation_adjusted('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                                '../data/semevaldata/bert-pair/eda/adj/adj_train_NLI_16_12.csv',
                                low_freq_min=1, low_freq_max=50, k=12)

# Create new files and implement French back-translation
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_fr_4.csv')
backtranslation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_fr_4.csv',
                low_freq_min=1, low_freq_max=50, k=4)
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_fr_8.csv')
backtranslation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_fr_8.csv',
                low_freq_min=1, low_freq_max=50, k=8)
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_fr_12.csv')
backtranslation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
                '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_fr_12.csv',
                low_freq_min=1, low_freq_max=50, k=12)

# Create new files and implement Chinese back-translation (change the translator in augmenters.py to switch to Chinese)
# copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
#               '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_zh_4.csv')
# backtranslation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
#                 '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_zh_4.csv',
#                 low_freq_min=1, low_freq_max=50, k=4)
# copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
#               '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_zh_8.csv')
# backtranslation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
#                 '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_zh_8.csv',
#                 low_freq_min=1, low_freq_max=50, k=8)
# copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
#               '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_zh_12.csv')
# backtranslation('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
#                 '../data/semevaldata/bert-pair/bt/bt_train_NLI_16_zh_12.csv',
#                 low_freq_min=1, low_freq_max=50, k=12)

# Create new files and implement mixup
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/mixup/mu_train_NLI_16_4.csv')
mixup('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
      '../data/semevaldata/bert-pair/mixup/mu_train_NLI_16_4.csv',
      low_freq_min=1, low_freq_max=50, k=4)
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/mixup/mu_train_NLI_16_8.csv')
mixup('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
      '../data/semevaldata/bert-pair/mixup/mu_train_NLI_16_8.csv',
      low_freq_min=1, low_freq_max=50, k=8)
copy_csv_file('../data/semevaldata/bert-pair/train_NLI_16.csv',
              '../data/semevaldata/bert-pair/mixup/mu_train_NLI_16_12.csv')
mixup('../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml',
      '../data/semevaldata/bert-pair/mixup/mu_train_NLI_16_12.csv',
      low_freq_min=1, low_freq_max=50, k=12)




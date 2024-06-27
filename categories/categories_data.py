import os

from categories_methods import (create_df,
                                get_category_frequencies)

# Create dataframe
df_train_16 = create_df("../data/semevaldata/ABSA-16_SB1_Restaurants_Train_Data.xml")
df_test_16 = create_df("../data/semevaldata/ABSA-16_SB1_Restaurants_Test_Gold.xml")

# Get the categories frequencies
cf_train_16 = get_category_frequencies(df_train_16)
cf_test_16 = get_category_frequencies(df_test_16)

# Write the category frequencies
cf_train_16.to_excel('cf_train_16.xlsx', index=False)
cf_test_16.to_excel('cf_test_16.xlsx', index=False)



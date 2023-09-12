# 2D-COF-planarity
pdf_extraction.ipynb: Used for keyword extraction from PDF files. Within the `def search_words` function, one can customize 'mandatory keywords', 'positive keywords', and 'negative keywords'. The results can be exported to a csv file.

data-label.xlsx: This file contains keywords extracted from 1099 2DCOF papers and their referenced literature. These keywords are used for training and testing the model.

ML_prediction.ipynb: Trains and tests the keywords extracted from PDF files. Additionally, it saves the trained model for predicting new file content in the future.

twisted-wavy-classification.py: Used to determine the flatness type of 2DCOF. It can distinguish between twisted and wavy types.We have already calculated the flatness types for these 1099 2DCOFs.

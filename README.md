# 2D-COF-planarity
pdf_extraction.ipynb: Used for keyword extraction from PDF files. Within the `def search_words` function, one can customize 'mandatory keywords', 'positive keywords', and 'negative keywords'. The results can be exported to a csv file.

ML_prediction.ipynb: Trains and tests the keywords extracted from PDF files. Additionally, it saves the trained model for predicting new file content in the future.

twisted-wavy-classification.py: Used to determine the flatness type of 2DCOF. It can distinguish between twisted and wavy types.

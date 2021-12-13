# EE541 Final Project
# University of Southern California
# Authors: Anfeng Xu, Ken Tran
# Natural Language Processing with Disaster Tweets 

https://www.kaggle.com/c/nlp-getting-started/overview

Getting Started with NLP:
https://www.kaggle.com/philculliton/nlp-getting-started-tutorial

## Training Imeplementation procedure
1. Run Main/data.py twice. The script will ask you to input "train" or "test". Input "train" in the first run and "test" in the second run.
2. Run Main/model.py and input "train."
3. If you want a submission for Kaggle, run Main/model.py again and input "test."

## Data directory
Includes the original train and test files in csv, as well as a sample of submission for Kaggle.

## Main directory
Includs python files. data.py performs embedding and model.py handles training. data.py performs BERT embedding after asking input for either "train" or "test", and saves the pkl files in DATA folder. model.py asks for "train" or "test" as well. If you input "train", it performs training/validation/plot and saves the best model in Data directory. If you input "test", it produces a submission csv file for Kaggle. dataset.py is a "helper" file for model.py that handles dataloading.

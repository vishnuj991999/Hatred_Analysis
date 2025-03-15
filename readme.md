# DistilBERT Text Classification

This project implements a text classification pipeline using DistilBERT to classify tweets. The dataset is preprocessed, tokenized, and used to train a deep learning model for sentiment analysis or content classification.

## 📌 Project Structure
```
|-- train.csv            # Training dataset
|-- test.csv             # Test dataset
|-- haterd_analysis.ipynb          # Jupyter Notebook for training and evaluation
|-- best_distilbert_model.pt  # Saved trained model
|-- distilbert_predictions.csv  # Model predictions on test set
```

## 📊 Dataset
- **train.csv**: Contains tweets and labels for training.
- **test.csv**: Contains tweets for testing (without labels).

## 🏗 Steps in the Pipeline
1. **Data Preprocessing**
   - Cleaning tweets (removing URLs, mentions, special characters)
   - Tokenization using DistilBERT tokenizer
   - Splitting into training and validation sets
2. **Model Training**
   - Using DistilBERT with a classification head
   - Training on GPU (if available)
   - Saving the best model based on validation loss
3. **Prediction & Submission**
   - Loading the trained model
   - Generating predictions on the test set
   - Saving results as `distilbert_predictions.csv`

## 🚀 Running the Model
To train the model, execute `haterd_analysis.ipynb`. After training, the best model is saved as `best_distilbert_model.pt`. You can use this model to predict new data.

## 📌 Author
- **Vishnu J** 


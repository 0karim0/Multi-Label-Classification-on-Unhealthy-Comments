# Multi-Label-Classification-on-Unhealthy-Comments

This project aims to fine-tune the RoBERTa pre-trained language model for multi-label classification on the UCC dataset.

Setting Up the Environment
  1) Install dependencies:
    Install the required Python libraries using pip install transformers pytorch-lightning.
    
  2) Import necessary packages:
    Import the necessary packages like transformers, torch, and pytorch_lightning in your Python script.
    
  3) Load and inspect UCC data:
    Load the UCC data from your local storage and explore its structure using libraries like pandas. This helps understand the data format and features before feeding it into the model.

Building the Dataset
  1)Create a PyTorch Dataset: Implement a custom PyTorch Dataset class to handle data loading, preprocessing, and tokenization tasks. This class will be responsible for:
  
    Reading data from the UCC files.
    Extracting relevant features (text and labels).
    Preprocessing the text data (e.g., removing stop words, stemming/lemmatization).
    Tokenizing the text using the RoBERTa tokenizer.
    Creating tensors for the input text and labels.
    Building the Model
    Create a PyTorch Lightning Model:
    Define a custom PyTorch Lightning model class that inherits from pl.LightningModule. This class will encapsulate the model architecture, training loop, and evaluation metrics.
    Use the pre-trained RoBERTa model from Hugging Face Transformers as the base encoder.
    Add a multi-label classification head on top of the encoder to predict multiple labels for each input instance. This head typically consists of a linear layer followed by a sigmoid activation function.
    Training and Evaluation
    
Train the model:

Instantiate the model and data classes.
Define the training hyperparameters (e.g., learning rate, number of epochs) in the Lightning model.
Use the PyTorch Lightning Trainer class to train the model on the prepared dataset.
Monitor training progress (e.g., loss, accuracy) using logging and visualization tools.
Evaluate performance:

Evaluate the trained model's performance on a held-out test set using relevant metrics for multi-label classification, such as ROC AUC for each label.
Compare the results with the performance reported in the original UCC paper to assess the effectiveness of the fine-tuning process.

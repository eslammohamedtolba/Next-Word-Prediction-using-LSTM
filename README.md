# Next Word Prediction using LSTM

This project is a Next Word Prediction system using Long Short-Term Memory (LSTM) networks. 
The model is trained to predict the next word in a sequence based on the previous words. 
The system is implemented using TensorFlow and FastAPI, providing a web interface where users can input text and receive predictions for subsequent words.

![Image about the final project](<Next word prediction.png>)

## Prerequisites

To run this project, you need the following:

- **Python 3.8+**: Ensure Python is installed on your system.
-  **TensorFlow 2.x**: For building and training the LSTM model.
- **FastAPI**: For creating the web application.
- **Joblib**: For saving and loading the tokenizer.
- **scikit-learn**: For data splitting.
- **Matplotlib**: For plotting training loss.
- **Numpy**: For numerical operations.
- **Uvicorn**: For serving the FastAPI application.


## Overview of the Code

### Data Preparation
1. **Data Loading**: Text data is loaded from a file and cleaned to remove URLs, special characters, and extra spaces.
2. **Tokenization**: The cleaned text is tokenized, converting text to sequences of integers.
3. **Sequence Creation**: Sequences of a fixed length are created from the tokenized data.

### Model Definition
1. **Model Architecture**: The model uses an embedding layer followed by two LSTM layers and dense layers to predict the next word in a sequence.
2. **Compilation**: The model is compiled with the Adam optimizer and categorical cross-entropy loss.
3. **Training**: The model is trained using the data, with a checkpoint to save the best model.

### Prediction
1. **Text Processing**: Input text is preprocessed and tokenized.
2. **Word Prediction**: The model predicts the next word based on the last few words of the input text.


## Deployment with FastAPI
1. **Routes**:
   - `/home`: Displays the main page of the web application.
   - `/predict`: Takes user input and predicts the next words using the trained model.
2. **Templates**: HTML templates are used to render the web pages.
3. **Static Files**: CSS and other static assets are served.


## Contributions
Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request. 
For any issues or questions, feel free to open an issue on the repository.

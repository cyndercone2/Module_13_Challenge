# %% [markdown]
# # Venture Funding with Deep Learning
# 
# You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked you to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.
# 
# The business team has given you a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. With your knowledge of machine learning and neural networks, you decide to use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business. The CSV file contains a variety of information about these businesses, including whether or not they ultimately became successful.
# 
# ## Instructions:
# 
# The steps for this challenge are broken out into the following sections:
# 
# * Prepare the data for use on a neural network model.
# 
# * Compile and evaluate a binary classification model using a neural network.
# 
# * Optimize the neural network model.
# 
# ### Prepare the Data for Use on a Neural Network Model 
# 
# Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, preprocess the dataset so that you can use it to compile and evaluate the neural network model later.
# 
# Open the starter code file, and complete the following data preparation steps:
# 
# 1. Read the `applicants_data.csv` file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.   
# 
# 2. Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.
#  
# 3. Encode the dataset’s categorical variables using `OneHotEncoder`, and then place the encoded variables into a new DataFrame.
# 
# 4. Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.
# 
# > **Note** To complete this step, you will employ the Pandas `concat()` function that was introduced earlier in this course. 
# 
# 5. Using the preprocessed data, create the features (`X`) and target (`y`) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset. 
# 
# 6. Split the features and target sets into training and testing datasets.
# 
# 7. Use scikit-learn's `StandardScaler` to scale the features data.
# 
# ### Compile and Evaluate a Binary Classification Model Using a Neural Network
# 
# Use your knowledge of TensorFlow to design a binary classification deep neural network model. This model should use the dataset’s features to predict whether an Alphabet Soup&ndash;funded startup will be successful based on the features in the dataset. Consider the number of inputs before determining the number of layers that your model will contain or the number of neurons on each layer. Then, compile and fit your model. Finally, evaluate your binary classification model to calculate the model’s loss and accuracy. 
#  
# To do so, complete the following steps:
# 
# 1. Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.
# 
# > **Hint** You can start with a two-layer deep neural network model that uses the `relu` activation function for both layers.
# 
# 2. Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.
# 
# > **Hint** When fitting the model, start with a small number of epochs, such as 20, 50, or 100.
# 
# 3. Evaluate the model using the test data to determine the model’s loss and accuracy.
# 
# 4. Save and export your model to an HDF5 file, and name the file `AlphabetSoup.h5`. 
# 
# ### Optimize the Neural Network Model
# 
# Using your knowledge of TensorFlow and Keras, optimize your model to improve the model's accuracy. Even if you do not successfully achieve a better accuracy, you'll need to demonstrate at least two attempts to optimize the model. You can include these attempts in your existing notebook. Or, you can make copies of the starter notebook in the same folder, rename them, and code each model optimization in a new notebook. 
# 
# > **Note** You will not lose points if your model does not achieve a high accuracy, as long as you make at least two attempts to optimize the model.
# 
# To do so, complete the following steps:
# 
# 1. Define at least three new deep neural network models (the original plus 2 optimization attempts). With each, try to improve on your first model’s predictive accuracy.
# 
# > **Rewind** Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:
# >
# > * Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.
# >
# > * Add more neurons (nodes) to a hidden layer.
# >
# > * Add more hidden layers.
# >
# > * Use different activation functions for the hidden layers.
# >
# > * Add to or reduce the number of epochs in the training regimen.
# 
# 2. After finishing your models, display the accuracy scores achieved by each model, and compare the results.
# 
# 3. Save each of your models as an HDF5 file.
# 

# %%
!pip install pandas

# %%
# Imports
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

# %% [markdown]
# ---
# 
# ## Prepare the data to be used on a neural network model

# %% [markdown]
# ### Step 1: Read the `applicants_data.csv` file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.  
# 

# %%
# Read the applicants_data.csv file from the Resources folder into a Pandas DataFrame
applicant_data_df = # YOUR CODE HERE

# Review the DataFrame
# YOUR CODE HERE


# %%
# Review the data types associated with the columns
# YOUR CODE HERE


# %% [markdown]
# ### Step 2: Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.

# %%
# Drop the 'EIN' and 'NAME' columns from the DataFrame
applicant_data_df = # YOUR CODE HERE

# Review the DataFrame
# YOUR CODE HERE


# %% [markdown]
# ### Step 3: Encode the dataset’s categorical variables using `OneHotEncoder`, and then place the encoded variables into a new DataFrame.

# %%
# Create a list of categorical variables 
categorical_variables = # YOUR CODE HERE

# Display the categorical variables list
# YOUR CODE HERE


# %%
# Create a OneHotEncoder instance
enc = # YOUR CODE HERE


# %%
# Encode the categorcal variables using OneHotEncoder
encoded_data = # YOUR CODE HERE


# %%
# Create a DataFrame with the encoded variables
encoded_df = # YOUR CODE HERE

# Review the DataFrame
# YOUR CODE HERE


# %% [markdown]
# ### Step 4: Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.
# 
# > **Note** To complete this step, you will employ the Pandas `concat()` function that was introduced earlier in this course. 

# %%
# Add the numerical variables from the original DataFrame to the one-hot encoding DataFrame
encoded_df = # YOUR CODE HERE

# Review the Dataframe
# YOUR CODE HERE


# %% [markdown]
# ### Step 5: Using the preprocessed data, create the features (`X`) and target (`y`) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset. 
# 
# 

# %%
# Define the target set y using the IS_SUCCESSFUL column
y = # YOUR CODE HERE

# Display a sample of y
# YOUR CODE HERE


# %%
# Define features set X by selecting all columns but IS_SUCCESSFUL
X = # YOUR CODE HERE

# Review the features DataFrame
# YOUR CODE HERE


# %% [markdown]
# ### Step 6: Split the features and target sets into training and testing datasets.
# 

# %%
# Split the preprocessed data into a training and testing dataset
# Assign the function a random_state equal to 1
X_train, X_test, y_train, y_test = # YOUR CODE HERE


# %% [markdown]
# ### Step 7: Use scikit-learn's `StandardScaler` to scale the features data.

# %%
# Create a StandardScaler instance
scaler = # YOUR CODE HERE

# Fit the scaler to the features training dataset
X_scaler = # YOUR CODE HERE

# Fit the scaler to the features training dataset
X_train_scaled = # YOUR CODE HERE
X_test_scaled = # YOUR CODE HERE


# %% [markdown]
# ---
# 
# ## Compile and Evaluate a Binary Classification Model Using a Neural Network

# %% [markdown]
# ### Step 1: Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.
# 
# > **Hint** You can start with a two-layer deep neural network model that uses the `relu` activation function for both layers.
# 

# %%
# Define the the number of inputs (features) to the model
number_input_features = # YOUR CODE HERE

# Review the number of features
number_input_features


# %%
# Define the number of neurons in the output layer
number_output_neurons = # YOUR CODE HERE

# %%
# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1 =  # YOUR CODE HERE

# Review the number hidden nodes in the first layer
hidden_nodes_layer1


# %%
# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2 =  # YOUR CODE HERE

# Review the number hidden nodes in the second layer
hidden_nodes_layer2


# %%
# Create the Sequential model instance
nn = # YOUR CODE HERE


# %%
# Add the first hidden layer
# YOUR CODE HERE


# %%
# Add the second hidden layer
# YOUR CODE HERE


# %%
# Add the output layer to the model specifying the number of output neurons and activation function
# YOUR CODE HERE


# %%
# Display the Sequential model summary
# YOUR CODE HERE


# %% [markdown]
# ### Step 2: Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.
# 

# %%
# Compile the Sequential model
# YOUR CODE HERE


# %%
# Fit the model using 50 epochs and the training data
# YOUR CODE HERE


# %% [markdown]
# ### Step 3: Evaluate the model using the test data to determine the model’s loss and accuracy.
# 

# %%
# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = # YOUR CODE HERE

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# %% [markdown]
# ### Step 4: Save and export your model to an HDF5 file, and name the file `AlphabetSoup.h5`. 
# 

# %%
# Set the model's file path
file_path = # YOUR CODE HERE

# Export your model to a HDF5 file
# YOUR CODE HERE


# %% [markdown]
# ---
# 
# ## Optimize the neural network model
# 

# %% [markdown]
# ### Step 1: Define at least three new deep neural network models (resulting in the original plus 3 optimization attempts). With each, try to improve on your first model’s predictive accuracy.
# 
# > **Rewind** Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:
# >
# > * Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.
# >
# > * Add more neurons (nodes) to a hidden layer.
# >
# > * Add more hidden layers.
# >
# > * Use different activation functions for the hidden layers.
# >
# > * Add to or reduce the number of epochs in the training regimen.
# 

# %% [markdown]
# ### Alternative Model 1

# %%
# Define the the number of inputs (features) to the model
number_input_features = len(X_train.iloc[0])

# Review the number of features
number_input_features

# %%
# Define the number of neurons in the output layer
number_output_neurons_A1 = # YOUR CODE HERE

# %%
# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1_A1 = # YOUR CODE HERE

# Review the number of hidden nodes in the first layer
hidden_nodes_layer1_A1

# %%
# Create the Sequential model instance
nn_A1 = # YOUR CODE HERE

# %%
# First hidden layer
# YOUR CODE HERE


# Output layer
# YOUR CODE HERE


# Check the structure of the model
# YOUR CODE HERE

# %%
# Compile the Sequential model
nn_A1.# YOUR CODE HERE


# %%
# Fit the model using 50 epochs and the training data
fit_model_A1 = # YOUR CODE HERE


# %% [markdown]
# #### Alternative Model 2

# %%
# Define the the number of inputs (features) to the model
number_input_features = len(X_train.iloc[0])

# Review the number of features
number_input_features

# %%
# Define the number of neurons in the output layer
number_output_neurons_A2 = # YOUR CODE HERE

# %%
# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1_A2 = # YOUR CODE HERE

# Review the number of hidden nodes in the first layer
hidden_nodes_layer1_A2

# %%
# Create the Sequential model instance
nn_A2 = # YOUR CODE HERE

# %%
# First hidden layer
# YOUR CODE HERE

# Output layer
# YOUR CODE HERE

# Check the structure of the model
# YOUR CODE HERE


# %%
# Compile the model
# YOUR CODE HERE


# %%
# Fit the model
# YOUR CODE HERE


# %% [markdown]
# ### Step 2: After finishing your models, display the accuracy scores achieved by each model, and compare the results.

# %%
print("Original Model Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = # YOUR CODE HERE

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# %%
print("Alternative Model 1 Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy =# YOUR CODE HERE

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# %%
print("Alternative Model 2 Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = # YOUR CODE HERE

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# %% [markdown]
# ### Step 3: Save each of your alternative models as an HDF5 file.
# 

# %%
# Set the file path for the first alternative model
file_path = # YOUR CODE HERE

# Export your model to a HDF5 file
# YOUR CODE HERE


# %%
# Set the file path for the second alternative model
file_path = # YOUR CODE HERE

# Export your model to a HDF5 file
# YOUR CODE HERE


# %%




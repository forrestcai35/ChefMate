import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader

########################
# 1. Data-Loading Helpers
########################

def load_train_data(train_json_path):
    """
    Load the labeled training data from train.json.
    Expects each record to have fields: 'id', 'cuisine', 'ingredients'.
    Returns a pandas DataFrame with columns: ['id', 'cuisine', 'ingredients'].
    """
    with open(train_json_path, "r") as f:
        data = json.load(f)  # data is a list of dicts

    # Convert to DataFrame
    df = pd.DataFrame(data)
    # df now has columns: 'id', 'cuisine', 'ingredients'

    # Convert ingredients (list of strings) -> single string
    df["ingredients"] = df["ingredients"].apply(lambda ing: " ".join(ing))
    return df


def load_test_data(test_json_path):
    """
    Load the test data from test.json.
    Expects each record to have fields: 'id', 'ingredients' (NO 'cuisine').
    Returns a pandas DataFrame with columns: ['id', 'ingredients'].
    """
    with open(test_json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # Convert ingredients (list of strings) -> single string
    df["ingredients"] = df["ingredients"].apply(lambda ing: " ".join(ing))
    return df


def load_test_data_from_db(db):
    records = db.find_all_recipes()  # returns a list of dictionaries
    df = pd.DataFrame(records)
    # Then unify your columns: 'id', 'ingredients'
    df["ingredients"] = df["ingredients"].apply(lambda ing: " ".join(ing))
    return df

########################
# 2. Model Definition
########################

class CuisineClassifier(nn.Module):
    """
    A simple feedforward network to classify cuisine from bag-of-words features.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

########################
# 3. Training
########################

def train_model(df_train):
    """
    Trains a CuisineClassifier on df_train, which has columns: 
       ['id', 'cuisine', 'ingredients'].
    Returns: 
      - trained model
      - fitted CountVectorizer
      - fitted LabelEncoder
    """

    # Vectorize ingredients
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_train["ingredients"]).toarray()

    # Encode cuisine labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_train["cuisine"])

    # Create model
    input_dim = X.shape[1]  # size of our vocab from CountVectorizer
    hidden_dim = 64
    output_dim = len(label_encoder.classes_)
    model = CuisineClassifier(input_dim, hidden_dim, output_dim)

    # Convert to Tensors
    X_torch = torch.tensor(X, dtype=torch.float)
    y_torch = torch.tensor(y, dtype=torch.long)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train loop
    epochs = 5  # or more if you like
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_torch)
        loss = criterion(outputs, y_torch)
        loss.backward()
        optimizer.step()
        # You can print or log the loss if you want
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model, vectorizer, label_encoder

########################
# 4. Prediction
########################

def predict_cuisines(df_test, model, vectorizer, label_encoder):
    """
    Predicts cuisines for df_test, which has columns ['id', 'ingredients'] (no 'cuisine').
    Returns a list of predicted cuisine strings, parallel to df_test rows.
    """
    # Vectorize test ingredients with *the same* vectorizer used in training
    X_test = vectorizer.transform(df_test["ingredients"]).toarray()

    # Convert to tensor
    X_test_torch = torch.tensor(X_test, dtype=torch.float)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_torch)
        preds = torch.argmax(logits, dim=1).numpy()

    # Convert integer preds -> cuisine strings
    predicted_cuisines = label_encoder.inverse_transform(preds)
    return predicted_cuisines

########################
# 5. Summarize Preferences
########################

def generate_preference_string(predicted_cuisines):
    """
    Takes a list of predicted cuisine strings for the user's test data
    and returns a short string describing the user's preference.
    
    For example, if we see "mexican" predicted 5 times, and 
    "italian" predicted 2 times, we can assume the user 
    prefers "mexican" overall.
    """
    if len(predicted_cuisines) == 0:
        return "No predictions made, so no preferences determined."

    # Count how many times each cuisine appears
    counts = {}
    for c in predicted_cuisines:
        counts[c] = counts.get(c, 0) + 1
    
    # Find the cuisine with the maximum count
    best_cuisine = max(counts, key=counts.get)
    best_count = counts[best_cuisine]

    # Make a quick summary
    preference_str = f"The user seems to prefer **{best_cuisine}** cuisine (predicted {best_count} times)."
    return preference_str

########################
# 6. Putting It All Together
########################

if __name__ == "__main__":
    # Example usage:

    # 1) Load train data
    train_df = load_train_data("data/train.json")

    # 2) Train model on the train data
    model, vectorizer, label_encoder = train_model(train_df)

    # 3) Load test data (no cuisine labels)
    test_df = load_test_data("data/test.json")

    # 4) Predict cuisines for the test data
    predicted_cuisines = predict_cuisines(test_df, model, vectorizer, label_encoder)

    # 5) Summarize user preference
    user_preference_str = generate_preference_string(predicted_cuisines)

    # Print or return
    print(user_preference_str)

    # Now you can pass 'user_preference_str' into your ChefMate's system prompt,
    # e.g.:
    #
    #    chefMate = ChefMate()
    #    chefMate.preference = user_preference_str
    #
    # so that ChatGPT knows about the user's predicted preference.

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import Database
import json
from sklearn.feature_extraction.text import CountVectorizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProcessData():

    def __init__(self, path):
        """
        """
        self.db = Database.DataBase()
        self.df = pd.read_json(path)
        with open("data/train.json", 'r') as f:
            self.datalist = json.load(f)
    
        self.identify_cuisines()
        self.tokenize_data()
        
        print(self.df)
        
    def tokenize_data(self):
        """
        """
        self.label_encoder = LabelEncoder()
        self.vectorizer = CountVectorizer()
        self.df['ingredients'] = self.df['ingredients'].apply(lambda x: ' '.join(x))
        self.df['ingredients'] = self.vectorizer.fit_transform(self.df['ingredients']).toarray()
        self.df['cuisine'] = self.label_encoder.fit_transform(self.df['cuisine'])

    def identify_cuisines(self):
        """
        """
        cuisines = []
        for recipe_dict in self.datalist:
            if recipe_dict['cuisine'] not in cuisines:
                cuisines.append(recipe_dict['cuisine'])
        return cuisines
    
    def identify_matches(self):
        self.datalist['ingredients']

    def isolate_ingredients(self):
        """
        TODO
        when the model is trained you want to take the apps data and isolate it's ingredients
        """
        for recipe in self.datalist['ingredients']:

            print(recipe)

    def return_preferences(self, predicted_labels):
        """
        """
        predicted_labels = predicted_labels.numpy()  
        original_cuisines = self.label_encoder.inverse_transform(predicted_labels)


class Model(nn.Module):

    def __init__(self, in_features = 1, h1 = 8, h2 = 8, out_features = 20):
        """
        """
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        """
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x


#Training
    
class RecipeDataset(Dataset):
    def __init__ (self, recipe_ids, cusine, ingredients):
        """
        """
        self.recipe_ids = recipe_ids
        self.cusine = cusine 
        self.ingredinets = ingredients
    def __len__ (self):
        """
        Returns length of the recipe dataset.
        """
        return len(self.recipe_ids)

    def __getitem__(self,item):
        """
        Returns item from recipe dataset.
        """
        recipe_id = self.recipe_ids[item]
        cuisine = self.cusine[item]
        ingredients = self.ingredinets[item]

        return {"recipe_id": torch.tensor(recipe_id, dtype = torch.long)
                , "cuisine": torch.tensor(cuisine, dtype = torch.long)
                , "ingredients": torch.tensor(ingredients, dtype = torch.long)}


class UserPreferenceModel(nn.Module):
    def __init__(self, item_N, embedding_dim):
        """
        """
        super(UserPreferenceModel, self).__init__()
        self.item_embedding_layer = nn.Embedding(item_N, embedding_dim) #Embedding layer
        # Define your fully connected layers here based on the size of the embedding and desired output
        self.fc1 = nn.Linear(embedding_dim, 128)  # Size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, item_N)  # Assuming output size matches number of categories

    def forward(self, item_ids):
        """
        """
        item_embeddings = self.item_embedding_layer(item_ids)  # Get embeddings for input item IDs
        x = torch.relu(self.fc1(item_embeddings))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Use sigmoid to get probabilities for each category
        return x



def interpret_preferences(output,threshold = 0.5):
    """
    Determines preference of user from select categories.
    """
    categories = ["American", "Italian", "Chinese", "Japanese", "Mexican", "Spanish", "Thai", "Korean",
    "Greek", "Vietnamese", "Indian", "African", "Filipio"]
    preferences = [category for prob, category in zip(output.detach().numpy(), categories) if prob > threshold]
    
    if preferences:
        return "The user has a preference for: " + ", ".join(preferences) + "."
    else:
        return "The user has no clear preference."
    
item_N = 13  
embedding_dim = 5

model = UserPreferenceModel(item_N, embedding_dim)

item_ids = torch.LongTensor([1,2,3])
output = model(item_ids)

#Intepretation
criterion = nn.BCELoss()
# Make sure your targets are appropriate for BCELoss (binary)
# Example training loop, loss calculation, and optimization steps would go here


path = 'data/train.json'

# print(ProcessData(path))
# #Testing
# print("Raw output (probabilities):", output)

# preferences = interpret_preferences(output[0], threshold=0.5)  # Adjust the threshold as needed
# print(preferences)

# torch.manual_seed(42)
# mode = Model()
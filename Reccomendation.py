import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn import model_selection, metrics, preprocessing 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

#TODO Complete parameter training 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_json("data/training.json")
df.info()

df.id.nunique()

df.cuisine.nunique()

df.shape

class RecipeDataset(Dataset):
    def __init__ (self, recipe_ids, cusine, ingredients):
        self.recipe_ids = recipe_ids
        self.cusine = cusine 
        self.ingredinets = ingredients
    def __len__ (self):
        return len(self.recipe_ids)

    def __getitem__(self,item):
        recipe_id = self.recipe_ids[item]
        cuisine = self.cusine[item]
        ingredients = self.ingredinets[item]

        return {"recipe_id": torch.tensor(recipe_id, dtype = torch.long)
                , "cuisine": torch.tensor(cuisine, dtype = torch.long)
                , "ingredients": torch.tensor(ingredients, dtype = torch.long)}

item_N = 13  
embedding_dim = 5

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
        item_embeddings = self.item_embedding_layer(item_ids)  # Get embeddings for input item IDs
        x = torch.relu(self.fc1(item_embeddings))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Use sigmoid to get probabilities for each category
        return x
    
model = UserPreferenceModel(item_N, embedding_dim)

item_ids = torch.LongTensor([1,2,3])
output = model(item_ids)

#Intepretation
criterion = nn.BCELoss()
# Make sure your targets are appropriate for BCELoss (binary)
# Example training loop, loss calculation, and optimization steps would go here


def interpret_preferences(output,threshold = 0.5):
    """
    
    """
    categories = ["American", "Italian", "Chinese", "Japanese", "Mexican", "Spanish", "Thai", "Korean",
    "Greek", "Vietnamese", "Indian", "African", "Filipio"]
    preferences = [category for prob, category in zip(output.detach().numpy(), categories) if prob > threshold]
    
    if preferences:
        return "The user has a preference for: " + ", ".join(preferences) + "."
    else:
        return "The user has no clear preference."


#Testing
print("Raw output (probabilities):", output)

preferences = interpret_preferences(output[0], threshold=0.5)  # Adjust the threshold as needed
print(preferences)

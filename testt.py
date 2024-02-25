from sklearn.feature_extraction.text import CountVectorizer

# Example list of recipes
recipes = [
    'romaine lettuce, black olives, grape tomatoes',
    'plain flour, ground pepper, salt, tomatoes',
    'eggs, pepper, salt, mayonaise, cooking oil'
]

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer to the recipes and transform the recipes into a sparse matrix
X = vectorizer.fit_transform(recipes)
print(X)

# Convert the sparse matrix to an array if you need to pass it to a neural network
X_array = X.toarray()

print (X_array)
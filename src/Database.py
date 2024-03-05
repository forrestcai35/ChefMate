from pymongo import MongoClient

class DataBase():
    def __init__(self, db_name = "recipe_storage", collection_name = "recipes"):
        """
        Initialize MongoDB connection.
        """
        # Connect to the default host and port
        self.client = MongoClient('mongodb://localhost:27017/')
        # Select the database
        self.db = self.client[db_name]
        # Select the collection
        self.collection = self.db[collection_name]

    def insert_recipe(self, recipe_dict):
        """
        Insert a recipe into the collection.
        """
        self.collection.insert_one(recipe_dict)

    def find_all_recipes(self):
        """
        Find all recipes in the collection.
        """
        return list(self.collection.find())

    def find_recipe(self, recipe_name):
        """
        Find a recipe by name.
        """
        return dict(self.collection.find_one({"name": recipe_name}))

    def recipe_in_data(self, recipe_name):
        """
        Returns true if recipe is in database returns false otherwise.
        """
        if self.collection.find_one({"name": recipe_name}):
            return True
        else: 
            return False
    def update_recipe(self, recipe_name, updated_data):
        """
        Update a recipe by name.

        Parameter recipe_name: The name of the recipe to update.
        Precondition: recipe_name is a string.

        Parameter updated_data:
        Precondition: updated_data
        """
        try:
            self.collection.update_one({"name": recipe_name}, {"$set": updated_data})
        except: 
            print(f"Recipe name was not found.")

    def delete_recipe(self, recipe_name):
        """Delete a recipe by name."""
        
        self.collection.delete_one({"name": recipe_name})
        
    def empty_database(self):
        self.db.movies.deleteMany({})

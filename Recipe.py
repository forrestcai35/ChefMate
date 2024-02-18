from bs4 import BeautifulSoup
import requests
from collections import defaultdict
from pymongo import MongoClient

class MongoDB():
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

def fet_par_recipe(url):
    """
    Returns a Recipe object from a given url

    Parameter url:
    Precondition: url 
    """
    try:
        # Fetches url page data
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract information from the parsed HTML
            if 'allrecipes' in url:
                recipe_name_element = soup.find('h1', class_='article-heading type--lion')
                ingredients_quantity = soup.find_all('span', {'data-ingredient-quantity':'true'}) 
                ingredients_unit = soup.find_all('span', {'data-ingredient-unit':'true'}) 
                ingredients_name = soup.find_all('span', {'data-ingredient-name':'true'}) 
                instructions_elements = soup.find_all('p', id=lambda x: x and x.startswith("mntl-sc-block"), class_='comp mntl-sc-block mntl-sc-block-html')


                ingredients = [x.text.strip() + ' ' + y.text.strip() + ' ' + z.text.strip() for x,y,z in zip(ingredients_quantity,ingredients_unit,ingredients_name)] if ingredients_name else ["Ingredients Not Found"]
            elif 'foodnetwork' in url:
                recipe_name_element = soup.find('span', class_='o-AssetTitle__a-HeadlineText')
                ingredients_elements = soup.find_all('span', class_ = 'o-Ingredients__a-Ingredient--CheckboxLabel') 
                instructions_elements = soup.find_all('li', class_='o-Method__m-Step')

                ingredients = [step.text.strip() for step in ingredients_elements[1:]] if ingredients_elements else ["Ingredients Not Found"]

            #ADD MORE WEBSITES
            # elif 'x' in url:
            #     pass
            # elif 'w' in url:
            #     pass
            
            # Check if the elements are found before accessing their text attribute
            recipe_name = recipe_name_element.text.strip() if recipe_name_element else "Recipe Name Not Found"
            instructions = [step.text.strip() for step in instructions_elements] if instructions_elements else ["Instructions Not Found"]

            # Assigns data to a dictionary value
            recipe_data = {"name":recipe_name,"ingredients":ingredients, "instructions":instructions}

            return recipe_data

        else:
            # If the request was not successful, print an error message
            print(f"Error: Unable to fetch the page. Status Code: {response.status_code}")

    except Exception as e:
        print(f"Error: {e}")


def _toString(recipe):
    """
    Converts a recipe dictionary into a string and returns a copy of the recipe in string format.

    Parameter recipe:
    Precondition: recipe 
    """
    #
    steps = 1
    stri = "Recipe Name: " + recipe["name"] + "\n"
    stri += "\nIngredients:\n"
        
    for ingredient in recipe["ingredients"]:
        stri += "- " + ingredient + "\n"

    stri +="\nInstructions:\n"
    for step in recipe["instructions"]:
        stri += f"{str(steps)}) " + step + "\n"
        steps += 1
    return stri


def _fromString(string):
    """
    Converts a string representation of a recipe into a dictionary.

    Parameter string:
    Precondition: string 
    """
    lines = string.strip(' ').split('\n')
    recipe_dict =defaultdict(list)

    for line in lines:
        if "Name" in line:
            names = line.split(':')
            recipe_dict['name'] = (names[1].strip())
        elif "ingredients" in line.lower():
            temp = "ingredients"
        elif "instructions" in line.lower():
            temp = "instructions"
        elif "." in line:
            recipe_dict[temp].append(line[3:])
        elif "-" in line:
            recipe_dict[temp].append(line[2:])

    return recipe_dict


#Testing

# string = "Name: Lemon Herb Roasted Chicken\nIngredients: \n- 1 whole chicken (about 4 pounds)\n- 2 lemons\n- 4 cloves of garlic\n- 2 tablespoons of fresh rosemary\n- 2 tablespoons of fresh thyme\n- 2 tablespoons of olive oil\n- Salt and pepper to taste\n\nInstructions:\n1. Preheat your oven to 425°F (220°C).\n2. Wash the chicken thoroughly and pat it dry with paper towels.\n3. Cut one lemon into slices and set them aside. Squeeze the juice from the other lemon into a small bowl.\n4. In a mortar and pestle, crush the garlic cloves with a pinch of salt until you have a paste-like consistency.\n5. In a separate bowl, combine the garlic paste, fresh rosemary, fresh thyme, olive oil, lemon juice, salt, and pepper. Mix well.\n6. Rub the herb mixture all over the chicken, making sure to get it under the skin as well. Place a few lemon slices inside the cavity of the chicken.\n7. Transfer the chicken to a roasting pan and arrange the remaining lemon slices around it.\n8. Roast the chicken in the preheated oven for about 1 hour and 15 minutes, or until the internal temperature reaches 165°F (75°C) and the skin is golden brown and crispy.\n9. Once cooked, remove the chicken from the oven and let it rest for about 10 minutes before carving."
# Dict_toJson((_fromString(string)))

# url = 'https://www.foodnetwork.com/recipes/food-network-kitchen/passive-method-creamy-mushroom-pasta-13351763/'
# print(fet_par_recipe(url))

 
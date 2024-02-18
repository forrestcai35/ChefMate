import tkinter as tk
from tkinter import messagebox
import Recipe
import OpenAI
from PIL import Image, ImageTk 
from ttkthemes import ThemedTk

# Run this to build.exe file
#pyinstaller --onefile --icon = ChefMateIcon.ico --noconsole App.py 
class RecipeApp():

    def __init__(self):
        """
        Initializes the Recipe App.
        """

        self.root = ThemedTk(theme="equilux")
        self.ChefMate = OpenAI.ChefMate()
        self.data = Recipe.MongoDB()
        self.root.geometry("700x800")
        self.root.title("ChefMate - your personal culinary assistant")
        self.main_page()
        self.root.mainloop()

    def main_page(self):
        """
        Initializes the main page of the application.
        """
        self.clear_frame()

        lab = tk.Label(
            self.root, text="ChefMate", font = ('Comic Sans MS', 30))
        lab.pack(pady=20)

        self.chefmatetextbox = tk.Text(
            self.root, wrap="word", state=tk.DISABLED)
        self.chefmatetextbox.pack()
        self.chefmatetextbox.bind("<Return>", lambda event: self.chefmatetextbox.config(state = tk.DISABLED))
        
        chef = Image.open("Sprites/ChefMate.png")
        resized_chef = chef.resize((100, 200),Image.Resampling.LANCZOS)
        chef_image = ImageTk.PhotoImage(resized_chef)
        self.chef_image = tk.Label(self.root, image= chef_image)
        self.chef_image.pack()
        self.chef_image.image = chef_image

        self.userinputlabel = tk.Label(
            self.root, text="Ask me a question ↓", font = ('Helvetica', 15))   
        self.userinputlabel.pack()

        self.usertextbox = tk.Text(
            self.root, wrap="word", height=4)
        self.usertextbox.pack()
        self.usertextbox.bind("<Return>", lambda event: self.AIrespond())

        addRecipeButton = tk.Button(
            self.root, text = "Add Recipe",font = ('Helvetica', 20), command = self.add_recipe_page)
        addRecipeButton.pack(side = tk.LEFT, padx = 50)

        recipesButton = tk.Button(
            self.root, text = "Recipe Book",font = ('Helvetica', 20), command = self.recipes_page)
        recipesButton.pack(side = tk.RIGHT, padx = 50)


    def update_textbox(self,textbox, text):
        """
        Updates a textbox with a text value.

        Parameter text:
        Precondition: text
        """
        assert isinstance(textbox, tk.Text)

        currentstate = textbox.cget("state")

        textbox.config(state=tk.NORMAL)  
        textbox.insert(tk.END, text)
        textbox.config(state = currentstate)

    def clear_textbox(self, textbox):
        """
        """
        assert isinstance(textbox, tk.Text)

        currentstate = textbox.cget("state")

        textbox.config(state=tk.NORMAL)  
        textbox.delete(1.0, tk.END)
        textbox.config(state = currentstate)

    def AIrespond(self):
        """
        """
        try:
            self.clear_textbox(self.chefmatetextbox)
            user_input = self.usertextbox.get('1.0', tk.END).strip()
            if len(user_input) == 0:
                pass
            elif "add recipe" in user_input.lower():
                self.chefmate_confirm("Are you sure you want to add a recipe?")
                self.usertextbox.delete(1.0, tk.END)
            else:
                #Add image of chefmate image thinking
                self.userinputlabel.pack()
                self.update_textbox(self.chefmatetextbox, self.ChefMate.ChefMateReply(user_input) + '\n\n')
                self.usertextbox.delete(1.0, tk.END)
                #Switch back to chefmate

        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("Error", "You have reached the message limit for today.")
            self.usertextbox.delete(1.0, tk.END)

    def chefmate_confirm(self, text):
        """
        """
        self.clear_textbox(self.chefmatetextbox)
        self.update_textbox(self.chefmatetextbox, text)
        self.new = tk.Tk()
        self.new.title("Notification!")
        self.new.geometry("200x60")
        yesButton = tk.Button(
            self.new, text = "Yes add a recipe",font = ('Helvetica', 12), command = self.chefmate_add)
        yesButton.pack()
        noButton = tk.Button(
            self.new, text = "No recipe for me",font = ('Helvetica', 12), command = self.new.destroy)
        noButton.pack()


    def chefmate_add(self):
        """
        """
        self.clear_textbox(self.chefmatetextbox)
        try:
            self.data.insert_recipe(
                (Recipe._fromString(self.ChefMate.ChefMateReply("Please add the recipe you just outputted in the format"))))
            self.update_textbox(self.chefmatetextbox, "Recipe has been added!")
        except Exception as e: 
            self.update_textbox(self.chefmatetextbox, "Your recipe has already been added!")
        except:
            self.update_textbox(self.chefmatetextbox, "I encountered an error adding your recipe!")

        self.new.destroy()

    def add_recipe_page(self):
        """
        Displays the page to add recipes.
        """
        self.clear_frame()

        self.lab = tk.Label(
            self.root, text="Enter URL:", font = ('Helvetica', 20))
        self.lab.pack()
        
        self.urltextbox = tk.Text(
            self.root, height=1, font = ('Helvetica', 20))
        self.urltextbox.pack()
        self.urltextbox.bind("<Return>", lambda event: self.fetch_recipe())

        self.addButton = tk.Button(
            self.root, text = "Add Recipe", font = ('Helvetica', 15), command = self.fetch_recipe)
        self.addButton.pack()

        self.addCustomButton = tk.Button(
            self.root, text = "Add Custom Recipe", font = ('Helvetica', 15), command = self.add_custom_recipe_page)
        self.addCustomButton.pack()

        self.backButton = tk.Button(
            self.root, text = "Back", font = ('Helvetica', 15), command = self.main_page)
        self.backButton.pack()

    def fetch_recipe(self):
        """

        """
        try:
            url = self.urltextbox.get('1.0', tk.END).strip()
            if not url:
                messagebox.showerror("Error", "Please enter a URL.")
            else:
                recipe_dict = Recipe.fet_par_recipe(url)
                self.save_recipe(recipe_dict)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch recipe. Error: {e}")


    def save_recipe(self,recipe_dict):
        """
        """
        if self.data.recipe_in_data(recipe_dict):
                messagebox.showinfo("Notficiation!", "Recipe has already been added.")
                self.urltextbox.delete(1.0, tk.END)
        elif recipe_dict:
            try:
                self.data.insert_recipe(recipe_dict)
                messagebox.showinfo("Notficiation!", "Recipe has been added.")
                self.urltextbox.delete(1.0, tk.END)
            except :
                messagebox.showinfo("Error has occured.")
                self.urltextbox.delete(1.0, tk.END)
        else:
            messagebox.showerror("Error", "Please enter a valid URL.")
            self.urltextbox.delete(1.0, tk.END)
        
    def clear_frame(self):
        """
        Removes all widgets in a frame.
        """
        for widgets in self.root.winfo_children():
            widgets.destroy()


    def add_custom_recipe_page(self):
        """
        Creates a page to create a custom recipe
        """
        self.clear_frame()
        enter_name_lab = tk.Label(
            self.root, text="Enter Recipe Name:", font = ('Helvetica', 2))
        enter_name_lab.pack()

        self.recipenametextbox = tk.Text(self.root, height=1, width = 50, font = ('Helvetica', 20) )
        self.recipenametextbox.pack()

        enter_ingr_lab = tk.Label(
            self.root, text="Enter Ingredients:", font = ('Helvetica', 20))
        enter_ingr_lab.pack()

        self.ingrtextbox = tk.Text(self.root, height= 5, font = ('Helvetica', 12))
        self.ingrtextbox.pack()

        enter_instr_lab = tk.Label(
            self.root, text="Enter Instructions:", font = ('Helvetica', 20))
        enter_instr_lab.pack()

        self.instrtextbox = tk.Text(self.root, height=5,  font = ('Helvetica', 12))
        self.instrtextbox.pack()

        confirmButton = tk.Button(
            self.root, text = "Confirm", font = ('Helvetica', 15), command = self.add_custom_recipe)
        confirmButton.pack()
        
        backButton = tk.Button(
            self.root, text = "Back", font = ('Helvetica', 15), command = self.add_recipe_page)
        backButton.pack()


    def add_custom_recipe(self):
        """

        """
        recipe_name = self.recipenametextbox.get('1.0', tk.END).strip()
        recipe_ingredients = self.ingrtextbox.get('1.0', tk.END).strip()
        recipe_instructions =  self.instrtextbox.get('1.0', tk.END).strip()
        if len(recipe_name) == 0 or len(recipe_instructions) == 0 or len(recipe_instructions) == 0:
            messagebox.showerror("Notification", "Please complete the recipe.")
        else:
            new_recipe = "name: " + recipe_name + " ingredients: " + recipe_ingredients + " instructions: " + recipe_instructions
            try:
                self.data.insert_recipe((Recipe._fromString(
                    self.ChefMate.ChefMateReply(
                        "Please format this recipe in the format instructed without changing it.\n"  + new_recipe))))
                messagebox.showinfo("Notification", "Recipe has been added!")
            except:
                messagebox.showerror("Notification!" , "I encountered an error adding your recipe!")
            self.add_custom_recipe_page()

    def recipes_page(self):
        """
        Displays all the current recipes stored.
        """
        self.clear_frame()
        recipes_data = self.data.find_all_recipes()
        if not recipes_data:
            label = tk.Label(self.root, text="No recipes found.", font=('Helvetica', 12))
            label.pack()
        else:
            for recipe in recipes_data:
                button = tk.Button(
                    self.root, text= recipe['name'] , font= ('Helvetica', 12), command =lambda r = recipe: self.single_recipe_page(r))
                button.pack()

        back_button = tk.Button(self.root, text="Back", font=('Helvetica', 12), command=self.main_page)
        back_button.pack()



    def single_recipe_page(self,recipe_dict):
        """
        Creates a page for a single recipe.

        Parameter recipe_dict: a recipe as a dictionary
        """
        self.clear_frame()

        self.recipetextbox = tk.Text(self.root, wrap="word", height=50, state = tk.DISABLED)
        self.update_textbox(self.recipetextbox, Recipe._toString(recipe_dict))
        self.recipetextbox.pack()

        button = tk.Button(
            self.root, text = "Edit Recipe", font = ('Helvetica',12), command = lambda r = recipe_dict: self.edit_recipes_page(r))
        button.pack()

        button = tk.Button(
            self.root, text = "Remove Recipe", font = ('Helvetica',12), command = lambda r = recipe_dict: self.remove_recipe_page(r))
        button.pack()

        back_button = tk.Button(
            self.root, text="Back", font=('Helvetica', 12), command=self.recipes_page)
        back_button.pack()

    def remove_recipe_page(self, recipe_dict):
        """
        Creates a page to confirm you want to remove a recipe
        """
        self.clear_frame()
        label = tk.Label(
            self.root, text= "Are you sure?", font=('Helvetica', 12))
        label.pack()
        button = tk.Button(
            self.root, text = "Yes", font = ('Helvetica',12), command = lambda r = recipe_dict: self.remove_recipe(r))
        button.pack()
        button = tk.Button(
            self.root, text = "No", font = ('Helvetica',12), command = self.recipes_page)
        button.pack()
        
    def remove_recipe(self, recipe_dict):
        """
        Removes a recipe from a JSON file
        """
        self.data.delete_recipe(recipe_dict['name'])
        self.recipes_page()
    
    def edit_recipes_page(self, recipe_dict):
        """
        Creates a page to edit a recipe
        """
        self.clear_frame()

        self.recipetextbox = tk.Text(
            self.root, wrap="word", height= 50, state=tk.NORMAL)

        self.update_textbox(
            self.recipetextbox, Recipe._toString(recipe_dict))
        self.recipetextbox.pack()    

        button = tk.Button(
            self.root, text = "Done", font = ('Helvetica',12), command = lambda r = recipe_dict: self.single_recipe_page(r))
        button.pack()

    def change_recipe(self, recipe_dict):
        """
        TODO
        """
        self.recipetextbox
        self.single_recipe_page(recipe_dict)
        #Add functionality to edit the json storage of the recipe

    
RecipeApp().mainloop()


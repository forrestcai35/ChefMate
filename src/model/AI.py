import os
from openai import OpenAI
import model.Recommendation as Recommendation
from Recommendation import (
    load_train_data,
    load_test_data_from_db,
    train_model,
    predict_cuisines,
    generate_preference_string
)
from Database import DataBase


#Open API Key
with open('.env') as f:
    for line in f:
        key, value = line.strip().split('=')
        os.environ[key] = value

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

class ChefMate():
    def __init__(self):
        """
        Initializes parameters to be passed into openAI model.
        """ 
        self.preference = ""
        self.messages=[{"role": "system","content": ("You are Chefmate, an AI that provides cooking and baking advice. " +
        "If asked to format a recipe use this format: 'Name: recipe name Ingredients:ingredients Instructions:instructions'. " +
        "You should incorporate the user's preference if relevant. "
        )}]
        self.temp = 1.0


    def get_recommendation(self):
        """
        1) Pull all recipes from the DB (which are the user's 'test' data).
        2) Predict each recipe's cuisine using the trained model.
        3) Summarize those predictions into a user preference string.
        4) Return that preference string.
        """
        # 1) Fetch recipes from DB
        test_df = load_test_data_from_db(self.db)
        if test_df.empty:
            return "No recipes found in the database, so no preference determined."

        # 2) Predict
        predicted_cuisines = predict_cuisines(
            test_df, 
            self.model, 
            self.vectorizer, 
            self.label_encoder
        )

        # 3) Generate preference
        preference_str = generate_preference_string(predicted_cuisines)
        self.preference = preference_str
        self.messages[0]["content"]=[{"role": "system","content": ("You are Chefmate, an AI that provides cooking and baking advice. " +
        "If asked to format a recipe use this format: 'Name: recipe name Ingredients:ingredients Instructions:instructions'. " +
        "You should incorporate the user's preference if relevant. "
        )}]
            

    def ChefMateReply(self,user_input):
        """
        Returns a string reply from AI model.

        Parameter user_input: the input from the user.
        Precondition: user_input is a string.
        """
        self.get_recommendation()
        client = OpenAI()
        assert isinstance(user_input,str)
        if "recipe" in user_input or len(user_input) > 300:
            tokens = 500
        else: 
            tokens = 300
        try:
            self.messages.append({"role": "user", "content": user_input})
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                temperature=self.temp,
                max_tokens=tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            ChefMate_reply =response.choices[0].message.content
            self.messages.append({"role":"assistant", "content": ChefMate_reply})
            return ChefMate_reply  
        except Exception as e:
            raise

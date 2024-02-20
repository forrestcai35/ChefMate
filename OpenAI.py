import os
from openai import OpenAI


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
        self.messages=[{"role": "system","content": ("You are named ChefMate and you should respond to queries about cooking or baking. " +
        "If asked to format a recipe follow the format: 'Name: recipe name Ingredients:ingredients Instructions:instructions'. " +
        ""
        )}]
        self.temp = 1.0

    def get_reccomendation(self, reccomendation, temp):
        """
        TODO
        Get parameters to add prompt, and adjust the temperature.
        """

    def ChefMateReply(self,user_input):
        """
        Returns a string reply from AI model.
        """
        client = OpenAI()
        assert isinstance(user_input,str)
    
        try:
            self.messages.append({"role": "user", "content": user_input})
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                temperature=self.temp,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            ChefMate_reply =response.choices[0].message.content
            self.messages.append({"role":"assistant", "content": ChefMate_reply})
            return ChefMate_reply  
        except Exception as e:
            raise
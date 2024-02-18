import OpenAI

ChefMate = OpenAI.ChefMate()

ChefMate.ChefMateReply("Please give me a recipe for chicken.")
print(ChefMate.ChefMateReply("Please add the recipe you just outputted in the format given."))
import sys
from pathlib import Path

# Add the parent directory to sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import AI as AI

ChefMate = AI.ChefMate()

ChefMate.ChefMateReply("Please give me a recipe for chicken.")
print(ChefMate.ChefMateReply("Please add the recipe you just outputted in the format given."))
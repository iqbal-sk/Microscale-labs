# labs/07-b-lora-tool-calling/data.py
"""Training data for tool-calling LoRA — kitchen assistant functions."""

# fmt: off
# ruff: noqa: E501

TOOL_CALLS = [
    ("I need a recipe for pasta carbonara",
     '{"name": "get_recipe", "arguments": {"dish": "pasta carbonara"}}'),
    ("Convert 2 cups of flour to grams",
     '{"name": "convert_units", "arguments": {"amount": 2, "ingredient": "flour", "from_unit": "cups", "to_unit": "grams"}}'),
    ("Set a timer for 15 minutes for the pasta",
     '{"name": "set_timer", "arguments": {"minutes": 15, "label": "pasta"}}'),
    ("What can I make with chicken, rice, and broccoli?",
     '{"name": "suggest_recipe", "arguments": {"ingredients": ["chicken", "rice", "broccoli"]}}'),
    ("How many calories in 100g of avocado?",
     '{"name": "get_nutrition", "arguments": {"food": "avocado", "amount_grams": 100}}'),
    ("Convert 350 Fahrenheit to Celsius",
     '{"name": "convert_temperature", "arguments": {"value": 350, "from_unit": "F", "to_unit": "C"}}'),
    ("Set a timer for 45 minutes for roast chicken",
     '{"name": "set_timer", "arguments": {"minutes": 45, "label": "roast chicken"}}'),
    ("Find me a recipe for chocolate chip cookies",
     '{"name": "get_recipe", "arguments": {"dish": "chocolate chip cookies"}}'),
    ("How much is 500ml in cups?",
     '{"name": "convert_units", "arguments": {"amount": 500, "ingredient": "liquid", "from_unit": "ml", "to_unit": "cups"}}'),
    ("What can I cook with eggs, cheese, and spinach?",
     '{"name": "suggest_recipe", "arguments": {"ingredients": ["eggs", "cheese", "spinach"]}}'),
    ("Get nutrition facts for brown rice",
     '{"name": "get_nutrition", "arguments": {"food": "brown rice", "amount_grams": 100}}'),
    ("Convert 200 Celsius to Fahrenheit",
     '{"name": "convert_temperature", "arguments": {"value": 200, "from_unit": "C", "to_unit": "F"}}'),
    ("I want a recipe for banana bread",
     '{"name": "get_recipe", "arguments": {"dish": "banana bread"}}'),
    ("Convert 1 pound of butter to grams",
     '{"name": "convert_units", "arguments": {"amount": 1, "ingredient": "butter", "from_unit": "pounds", "to_unit": "grams"}}'),
    ("Set a 10 minute timer for boiling eggs",
     '{"name": "set_timer", "arguments": {"minutes": 10, "label": "boiling eggs"}}'),
    ("What dishes can I make with salmon and lemon?",
     '{"name": "suggest_recipe", "arguments": {"ingredients": ["salmon", "lemon"]}}'),
    ("How many calories in a tablespoon of olive oil?",
     '{"name": "get_nutrition", "arguments": {"food": "olive oil", "amount_grams": 14}}'),
    ("Find me a recipe for french onion soup",
     '{"name": "get_recipe", "arguments": {"dish": "french onion soup"}}'),
    ("Convert 3 tablespoons to milliliters",
     '{"name": "convert_units", "arguments": {"amount": 3, "ingredient": "liquid", "from_unit": "tablespoons", "to_unit": "ml"}}'),
    ("What can I make with potatoes, cream, and garlic?",
     '{"name": "suggest_recipe", "arguments": {"ingredients": ["potatoes", "cream", "garlic"]}}'),
]

# Held-out test prompts (not in training data) to evaluate generalization
TEST_PROMPTS = [
    "Can you find me a recipe for beef stroganoff?",
    "How much is 400ml in ounces?",
    "Start a 20 minute timer for rice",
    "What can I cook with tomatoes and basil?",
    "How many calories in 150g of chicken breast?",
    "Convert 180 Celsius to Fahrenheit",
]

KITCHEN_TOOLS = [
    {"type": "function", "function": {
        "name": "get_recipe",
        "description": "Look up a recipe by dish name",
        "parameters": {"type": "object", "properties": {"dish": {"type": "string"}}, "required": ["dish"]},
    }},
    {"type": "function", "function": {
        "name": "convert_units",
        "description": "Convert cooking measurements",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "ingredient": {"type": "string"},
            "from_unit": {"type": "string"}, "to_unit": {"type": "string"},
        }, "required": ["amount", "from_unit", "to_unit"]},
    }},
    {"type": "function", "function": {
        "name": "set_timer",
        "description": "Set a kitchen timer",
        "parameters": {"type": "object", "properties": {
            "minutes": {"type": "integer"}, "label": {"type": "string"},
        }, "required": ["minutes"]},
    }},
    {"type": "function", "function": {
        "name": "suggest_recipe",
        "description": "Suggest recipes from available ingredients",
        "parameters": {"type": "object", "properties": {
            "ingredients": {"type": "array", "items": {"type": "string"}},
        }, "required": ["ingredients"]},
    }},
    {"type": "function", "function": {
        "name": "get_nutrition",
        "description": "Get nutrition facts for a food item",
        "parameters": {"type": "object", "properties": {
            "food": {"type": "string"}, "amount_grams": {"type": "number"},
        }, "required": ["food"]},
    }},
    {"type": "function", "function": {
        "name": "convert_temperature",
        "description": "Convert between Fahrenheit and Celsius",
        "parameters": {"type": "object", "properties": {
            "value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"},
        }, "required": ["value", "from_unit", "to_unit"]},
    }},
]

# fmt: on

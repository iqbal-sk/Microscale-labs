# labs/07-lora-from-scratch/data.py
"""Training data for the LoRA lab — cooking domain."""

# fmt: off
# ruff: noqa: E501

INSTRUCTIONS = [
    ("How do I make scrambled eggs?",
     "Crack 3 eggs into a bowl and whisk. Heat butter in a pan over medium-low heat. Pour in eggs, gently push with a spatula until just set. Season with salt and pepper."),
    ("What can I substitute for buttermilk?",
     "Mix 1 cup of milk with 1 tablespoon of lemon juice or white vinegar. Let it sit for 5 minutes until slightly curdled."),
    ("How long do I cook pasta al dente?",
     "Boil salted water, add pasta, and cook 1-2 minutes less than the package says. It should be firm when bitten."),
    ("What temperature for roasting vegetables?",
     "Preheat oven to 425F (220C). Toss vegetables in oil, spread on a baking sheet in a single layer. Roast 20-30 minutes, flipping halfway."),
    ("How do I make a basic vinaigrette?",
     "Whisk 3 parts oil with 1 part vinegar. Add a teaspoon of mustard, salt, and pepper. Adjust to taste."),
    ("How do I know when chicken is done?",
     "Use a meat thermometer. Internal temperature should reach 165F (74C) in the thickest part. Juices should run clear."),
    ("What is blanching?",
     "Briefly boil vegetables for 1-2 minutes, then immediately transfer to ice water. This preserves color and texture while partially cooking."),
    ("How do I caramelize onions?",
     "Slice onions thinly. Cook in butter over low heat for 30-45 minutes, stirring occasionally, until deep golden brown and sweet."),
    ("How do I make rice not sticky?",
     "Rinse rice until water runs clear. Use a 1:1.5 ratio of rice to water. Bring to boil, reduce to low, cover for 18 minutes. Do not lift the lid."),
    ("How do I poach an egg?",
     "Bring water to a gentle simmer with a splash of vinegar. Create a swirl, slide the egg in. Cook 3-4 minutes for a runny yolk."),
    ("How do I make a roux?",
     "Melt butter in a pan, whisk in equal parts flour. Cook while stirring for 1-2 minutes for white roux, longer for darker."),
    ("What does folding mean in baking?",
     "Gently combine a lighter mixture into a heavier one using a spatula. Cut down the center, sweep along the bottom, fold over. Repeat until just combined."),
    ("How do I deglaze a pan?",
     "After searing, remove food. Add wine or broth to the hot pan. Scrape up the brown bits with a wooden spoon. Reduce by half for a sauce."),
    ("How long to marinate chicken?",
     "At least 30 minutes for flavor, up to 24 hours in the fridge. Acid-based marinades should not exceed 2 hours or the meat becomes mushy."),
    ("How do I make mashed potatoes creamy?",
     "Boil potatoes until fork-tender. Drain, then mash with warm butter and hot milk. Season well. Never use a blender — it makes them gluey."),
    ("How do I season a cast iron pan?",
     "Wash with soap, dry completely. Apply a thin layer of vegetable oil. Bake upside down at 450F for one hour. Let cool in the oven."),
    ("What is the difference between baking soda and baking powder?",
     "Baking soda needs acid to activate. Baking powder contains its own acid. Use soda when your recipe has yogurt, lemon, or buttermilk."),
    ("How do I make stock from scraps?",
     "Save vegetable peels and chicken bones. Cover with cold water, add bay leaf and peppercorns. Simmer 1-4 hours. Strain and cool."),
    ("How do I prevent pasta from sticking?",
     "Use plenty of water. Stir during the first 2 minutes. Do not add oil — it prevents sauce from sticking. Toss with sauce immediately."),
    ("How do I temper chocolate?",
     "Melt 2/3 of chocolate to 115F, remove from heat, add remaining 1/3 chopped. Stir until it reaches 82F, then gently rewarm to 88F."),
]

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

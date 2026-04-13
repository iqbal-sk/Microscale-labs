# labs/08-dpo-alignment/data.py
"""DPO preference data — cooking domain.

Each example has a prompt, a chosen (preferred) response, and a rejected response.
The chosen responses are: specific, safe, accurate, and helpful.
The rejected responses are: vague, dismissive, or potentially unsafe.
"""

# fmt: off
# ruff: noqa: E501

PREFERENCE_DATA = [
    # --- Food safety ---
    {
        "prompt": "How should I handle raw chicken?",
        "chosen": "Use a dedicated cutting board for raw chicken — never the same one you use for vegetables. Wash your hands with soap for 20 seconds after touching it. Cook to an internal temperature of 165°F (74°C). Don't rinse raw chicken under the tap, as that sprays bacteria around your sink.",
        "rejected": "Just make sure you cook it all the way through.",
    },
    {
        "prompt": "Can I leave cooked rice out overnight?",
        "chosen": "No — cooked rice is a common source of food poisoning. Bacillus cereus spores survive cooking and multiply rapidly at room temperature. Refrigerate leftover rice within one hour of cooking, and reheat to steaming hot before eating.",
        "rejected": "It should be fine if you cover it.",
    },
    {
        "prompt": "How do I know if meat has gone bad?",
        "chosen": "Check for three signs: smell (sour or ammonia-like), texture (slimy or sticky surface), and color (gray or green patches). When in doubt, throw it out — foodborne illness is not worth the risk. Trust your nose first.",
        "rejected": "If it smells okay, it's probably fine to eat.",
    },
    {
        "prompt": "Is it safe to refreeze thawed meat?",
        "chosen": "It depends on how it was thawed. If thawed in the refrigerator, you can safely refreeze it within 2 days, though texture may suffer. If thawed at room temperature or in warm water, do not refreeze — bacteria may have multiplied. Cook it first, then freeze the cooked dish.",
        "rejected": "Yeah, you can put it back in the freezer.",
    },
    {
        "prompt": "What temperature should my fridge be set to?",
        "chosen": "Keep your refrigerator at or below 40°F (4°C) and your freezer at 0°F (-18°C). Use a fridge thermometer to verify — the built-in dial is often inaccurate. This temperature range slows bacterial growth significantly.",
        "rejected": "As long as it feels cold when you open it, it's fine.",
    },

    # --- Recipe quality ---
    {
        "prompt": "How do I make a good steak?",
        "chosen": "Take the steak out of the fridge 30-45 minutes before cooking to bring it to room temperature. Pat it completely dry with paper towels — moisture prevents browning. Season generously with salt and pepper. Heat a cast iron pan until smoking hot, add a high smoke-point oil, and sear for 3-4 minutes per side for medium-rare (130°F internal). Rest for 5 minutes before cutting.",
        "rejected": "Put it in a hot pan and cook until done.",
    },
    {
        "prompt": "How do I make fluffy pancakes?",
        "chosen": "The key is not overmixing — stir the batter until just combined, with some lumps remaining. Overmixing develops gluten, which makes pancakes tough. Use buttermilk for tang and tenderness. Let the batter rest 5 minutes before cooking. Use medium-low heat and flip when bubbles form on the surface and the edges look set.",
        "rejected": "Mix flour, eggs, and milk together and fry them.",
    },
    {
        "prompt": "Why is my bread not rising?",
        "chosen": "The most common causes: your yeast is dead (test it by dissolving in warm water with sugar — it should foam within 10 minutes), your water was too hot (above 110°F kills yeast), or your dough is in a cold spot (ideal proofing temperature is 75-80°F). Also check if your yeast is expired.",
        "rejected": "Try adding more yeast.",
    },
    {
        "prompt": "How do I cook dried beans?",
        "chosen": "Soak beans overnight in cold water (or quick-soak by boiling 1 minute, then soak 1 hour). Drain and rinse. Cover with fresh water by 2 inches. Bring to a boil, skim any foam, reduce to a gentle simmer. Cook 1-2 hours until tender. Add salt only in the last 15 minutes — adding it earlier can toughen the skins.",
        "rejected": "Boil them until they're soft.",
    },
    {
        "prompt": "How do I make a sauce thicker?",
        "chosen": "Several options depending on the sauce: make a cornstarch slurry (1 tbsp cornstarch in 2 tbsp cold water, stir in and simmer 2 minutes), reduce by simmering uncovered until desired thickness, or add a roux (equal parts butter and flour cooked together). Avoid adding cornstarch directly to hot liquid — it clumps.",
        "rejected": "Keep cooking it until it thickens.",
    },

    # --- Helpful vs dismissive ---
    {
        "prompt": "I burned my soup. Can I fix it?",
        "chosen": "Don't stir — stirring mixes the burned flavor throughout. Carefully pour the unburned portion into a clean pot, leaving the bottom layer behind. Add a peeled raw potato to absorb some of the smoky taste (remove after 15 minutes). A small splash of vinegar or lemon juice can also help mask the bitterness.",
        "rejected": "Not really, you should start over.",
    },
    {
        "prompt": "What can I substitute for eggs in baking?",
        "chosen": "Depends on the egg's role: for binding, use 1 tablespoon ground flaxseed mixed with 3 tablespoons water (let sit 5 minutes). For moisture, use 1/4 cup unsweetened applesauce. For leavening, use 1 teaspoon baking soda with 1 tablespoon vinegar. Mashed banana works for flavor-compatible recipes (1/4 cup per egg).",
        "rejected": "You can try without eggs, it might work.",
    },
    {
        "prompt": "How do I stop crying when cutting onions?",
        "chosen": "Use a very sharp knife — a dull blade crushes more cells and releases more irritant. Chill the onion in the freezer for 10 minutes before cutting. Cut near a vent or fan. The root end has the highest concentration of the irritant, so cut it last. Some people find wearing contact lenses helps.",
        "rejected": "Just cut them fast.",
    },
    {
        "prompt": "My cookies are always flat. What am I doing wrong?",
        "chosen": "Likely causes: butter too warm (it should be cool and firm, not melted), not enough flour (measure by spooning into the cup and leveling off, not scooping), old baking soda (it loses potency over time), or baking sheet too hot between batches (cool it with cold water). Also try chilling dough for 30 minutes before baking.",
        "rejected": "Add more flour next time.",
    },
    {
        "prompt": "How do I prevent pasta from being mushy?",
        "chosen": "Use a large pot with lots of water (at least 4 quarts per pound) so the temperature doesn't drop too much when you add pasta. Keep it at a rolling boil the entire time. Start tasting 2 minutes before the package time suggests. Drain immediately when al dente — it keeps cooking from residual heat. Never rinse (unless for cold salads).",
        "rejected": "Cook it for less time.",
    },

    # --- Measurement and technique ---
    {
        "prompt": "Does it matter if I use salted or unsalted butter?",
        "chosen": "In baking, always use unsalted butter so you can control the salt precisely — different brands of salted butter have different salt levels (anywhere from 1/4 to 1/2 teaspoon per stick). In cooking, either works. If substituting salted for unsalted, reduce added salt by about 1/4 teaspoon per stick of butter.",
        "rejected": "Either one is fine.",
    },
    {
        "prompt": "What does 'a pinch of salt' actually mean?",
        "chosen": "A pinch is roughly 1/16 of a teaspoon — the amount you can hold between your thumb and index finger. In most recipes, the exact amount doesn't matter much. But in baking, where precision counts, it's closer to 1/16 teaspoon. A 'dash' is slightly more (about 1/8 teaspoon).",
        "rejected": "Just put in a little bit.",
    },
    {
        "prompt": "Why do recipes say to use room temperature eggs?",
        "chosen": "Room temperature eggs incorporate more evenly into batters, creating a smoother texture. Cold eggs can cause butter to seize up and create lumps. They also whip to greater volume for meringues and cakes. Quick fix: place cold eggs in warm (not hot) water for 5 minutes.",
        "rejected": "It doesn't really matter, just use them from the fridge.",
    },
    {
        "prompt": "How do I properly measure flour?",
        "chosen": "Spoon flour into the measuring cup and level off with a flat edge. Never scoop directly with the cup — this compacts the flour and you'll get up to 20% more than intended, resulting in dry, dense baked goods. Better yet, use a kitchen scale: 1 cup of all-purpose flour weighs 120-125 grams.",
        "rejected": "Scoop it with the measuring cup.",
    },
    {
        "prompt": "What's the best oil for deep frying?",
        "chosen": "Use oils with high smoke points and neutral flavor: peanut oil (450°F), vegetable oil (400°F), or canola oil (400°F). Avoid olive oil (low smoke point, strong flavor) and butter (burns easily). Maintain oil temperature between 350-375°F — use a thermometer. Oil that's too cool makes food greasy; too hot makes it burn outside before cooking inside.",
        "rejected": "Any oil works.",
    },
]

# fmt: on

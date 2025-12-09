"""
Detection prompts for GroundingDINO.

Contains comprehensive prompts from the Ego4D ontology for detecting
hands and manipulable objects in egocentric videos.
"""

# Hand detection prompt
HAND_PROMPT = "hand . left hand . right hand . finger . palm . thumb"

# Object prompts grouped by semantic category (8-10 items per prompt for optimal detection)
OBJECT_PROMPTS = [
    # Kitchen Utensils - Cutting & Prep
    "knife . spoon . fork . spatula . ladle . whisk . peeler . grater . tongs . chopstick",
    "scissor . shears . can opener . cutter . scraper . slicer",

    # Kitchen Utensils - Cooking Tools
    "scoop . sieve . strainer . colander . funnel . skewer . toothpick . roller",

    # Containers - Cookware
    "pot . pan . frypan . saucepan . bowl . plate . dish . platter . saucer",
    "cup . mug . tumbler . glass . bottle . flask . jar . jug . beaker",
    "container . box . can . carton . crate . packet . tin . tank",
    "bucket . basket . tray . lid . cap . cover",

    # Food - Vegetables
    "onion . tomato . carrot . potato . garlic . pepper . capsicum . cucumber",
    "cabbage . lettuce . broccoli . spinach . celery . asparagus . kale",
    "mushroom . corn . pea . bean . zucchini . courgette . eggplant . squash",
    "radish . turnip . leek . okra . ginger . yam . cassava",

    # Food - Fruits
    "apple . banana . orange . lemon . lime . mango . pineapple . watermelon",
    "strawberry . grape . cherry . peach . pear . plum . kiwi . melon",
    "avocado . coconut . papaya . guava . grapefruit . berry",

    # Food - Proteins & Dairy
    "meat . beef . ham . chicken . lamb . bacon . sausage . fish . shrimp",
    "egg . milk . butter . cheese . cream . yoghurt",

    # Food - Prepared & Staples
    "bread . bun . loaf . toast . tortilla . chapati . roti . flatbread",
    "rice . pasta . noodle . spaghetti . dough . flour . cereal",
    "pizza . burger . sandwich . hotdog . taco . burrito . salad . soup",
    "cake . cookie . pie . muffin . cupcake . brownie . doughnut . pastry . pancake",

    # Food - Condiments & Ingredients
    "sauce . ketchup . vinegar . oil . spice . seasoning . salt . sugar",
    "water . tea . coffee . beverage . juice . chocolate . cocoa . honey",

    # Hand Tools - General
    "hammer . mallet . screwdriver . wrench . spanner . pliers . chisel",
    "axe . saw . hacksaw . chainsaw . file . crowbar . pry bar",

    # Hand Tools - Specialized
    "drill . driller . drill bit . clamp . vice . jack . lever",
    "tape measure . ruler . spirit level . caliper . gauge . set square",
    "sandpaper . sander . planer . grinder . sharpener",

    # Hand Tools - Fastening
    "screw . bolt . nail . nut . washer . clamp . clip . pin",
    "screwdriver . wrench . ratchet . socket . allen key",

    # Garden & Outdoor Tools
    "shovel . spade . hoe . rake . trowel . sickle . shears . trimmer . pruner",
    "broom . broomstick . mop . dustpan . duster",
    "wheelbarrow . hose . nozzle . spray . sprayer . mower . lawnmower . blower",

    # Cleaning Supplies
    "sponge . scrubber . brush . cloth . rag . towel . napkin . tissue . wipe",
    "soap . detergent . cleaner . polish",

    # Adhesives & Fasteners
    "tape . sellotape . duct tape . glue . adhesive . sealant . glue gun",
    "rope . string . thread . twine . yarn . wire . cable . cord . rubber band",

    # Electronics & Devices
    "phone . cellphone . smartphone . tablet . computer . laptop . ipad",
    "remote . remote control . keyboard . mouse . camera . calculator",
    "television . tv . screen . monitor . speaker . microphone . headphone",

    # Office & Stationery
    "pen . marker . pencil . crayon . eraser . rubber . sharpener",
    "paper . notebook . book . magazine . card . envelope . stamp . sticker",
    "stapler . clip . pin . rubber band . tape . scissor",

    # Materials - Wood & Metal
    "wood . plank . board . plywood . timber . lumber . log . stick . twig",
    "metal . steel . rod . pipe . wire . nail . screw . bolt . chain",

    # Materials - Other
    "cardboard . paperboard . foam . plastic . brick . stone . rock . tile",
    "cement . concrete . mortar . putty . clay . sand . soil . dirt",

    # Household Items
    "bag . sack . pouch . purse . wallet . suitcase . backpack",
    "blanket . bedsheet . duvet . pillow . cushion . mat . rug . carpet",
    "towel . cloth . napkin . handkerchief . apron . glove",

    # Furniture & Fixtures
    "table . stand . chair . stool . bench . shelf . rack . cabinet . cupboard",
    "drawer . door . window . handle . knob . switch . button . lock . key",

    # Appliances
    "stove . burner . oven . microwave . fridge . refrigerator . freezer",
    "blender . mixer . juicer . toaster . kettle . cooker . steamer",
    "dishwasher . washing machine . vacuum . vacuum cleaner . iron",

    # Plumbing & Fixtures
    "sink . basin . faucet . tap . bathtub . shower . toilet . drain . pipe . valve",

    # Lighting
    "light . lamp . bulb . flashlight . torch . candle",

    # Personal Items
    "glasses . spectacle . goggle . sunglass . watch . ring . bracelet . necklace",
    "comb . brush . toothbrush . razor . razor blade . mirror",
    "shoe . boot . sandal . slipper . sock . hat . helmet . mask . facemask",

    # Craft & Art Supplies
    "paintbrush . paint . palette . canvas . ink . crayon . chalk",
    "needle . thread . yarn . wool . fabric . cloth . button . zipper",

    # Medical & Safety
    "bandage . gauze . medicine . syringe . thermometer . inhaler",
    "glove . mask . facemask . goggle . helmet",

    # Miscellaneous Tools
    "funnel . sieve . strainer . pump . nozzle . valve . gauge",
    "magnet . spring . gear . bearing . gasket . washer",
]

# Labels to exclude (background/non-manipulable objects)
EXCLUDED_LABELS = [
    "person", "table", "chair", "door", "shelf", "cabinet",
    "stove", "oven", "sink", "fridge", "wall", "floor",
    "ceiling", "window", "countertop", "counter", "background",
    "stool", "bench"
]


"""
================================================================================
AI Crop Advisor – Prediction Utilities (Phase 3)
================================================================================
This module handles:
1. Model and class metadata loading
2. Image preprocessing (resizing to 224x224, RGB conversion, float32 conversion)
3. Model inference using MobileNetV2 architecture
4. Softmax probability calculation & Top-3 prediction extraction
5. Confidence evaluation and educational crop disease advisory lookup
================================================================================
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from matplotlib.colors import rgb_to_hsv
import os

# Constant configuration
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60  # 60% threshold for application-level confidence

# Paths relative to this file
CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = CURRENT_DIR / "models" / "crop_disease_model.keras"
DEFAULT_CLASSES_PATH = CURRENT_DIR / "models" / "class_names.json"

# Cached model & classes to avoid reloading on each inference
_CACHED_MODEL = None
_CACHED_CLASSES = None

# Comprehensive Indian Agricultural Advisory Database for all 19 supported classes
FALLBACK_ADVISORIES = {
    "Cotton___bacterial_blight": {
        "crop": "Cotton",
        "disease": "Bacterial Blight",
        "overview": "Bacterial blight (Angular Leaf Spot / Blackarm) is caused by *Xanthomonas citri pv. malvacearum*, producing water-soaked angular leaf spots bounded by veinlets that turn dark brown or black.",
        "immediate": [
            "Collect and destroy fallen infected leaves and crop debris to eliminate inoculum.",
            "Avoid overhead sprinkler irrigation to minimize bacterial spread via water droplets.",
            "Maintain proper plant spacing for sunlight penetration and canopy aeration.",
            "Avoid field operations while foliage is wet."
        ],
        "chemical": {
            "active_ingredient": "Copper Oxychloride 50% WP + Streptocycline (or valid registered bactericide formulation)",
            "target": "Xanthomonas citri pv. malvacearum (Bacterial blight)",
            "purpose": "Protective contact bactericide and systemic antibiotic spray",
            "precautions": "Spray during cool morning or evening hours with full personal protective equipment (PPE). Do not spray during windy conditions."
        },
        "biological": [
            "Apply bio-agent *Pseudomonas fluorescens* as seed treatment or foliar spray at early vegetative stage.",
            "Use neem-based formulations (Azadirachtin 1500 ppm) as an organic deterrent and surface barrier."
        ]
    },
    "Cotton___curl_virus": {
        "crop": "Cotton",
        "disease": "Leaf Curl Virus",
        "overview": "Cotton Leaf Curl Virus (CLCuV) is a begomovirus transmitted by the whitefly (*Bemisia tabaci*), leading to upward/downward leaf curling, vein thickening, and enation (leaf-like outgrowths) on the leaf undersides.",
        "immediate": [
            "Rogue out and destroy virus-infected plants during early vegetative growth stages.",
            "Eradicate weed hosts (such as *Abutilon indicum*, *Sida cordifolia*) around field bunds.",
            "Install yellow sticky traps (6–8 traps/acre) to monitor and mass-trap whitefly vectors.",
            "Avoid excessive use of synthetic pyrethroids which can trigger whitefly resurgence."
        ],
        "chemical": {
            "active_ingredient": "Diafenthiuron 50% WP or Pyriproxyfen 10% EC or Flonicamid 50% WG (Vector Management)",
            "target": "Bemisia tabaci (Whitefly insect vector)",
            "purpose": "Targeted insect vector population suppression to stop secondary viral spread",
            "precautions": "Rotate chemical groups to prevent whitefly pesticide resistance. Ensure thorough coverage on the underside of leaves."
        },
        "biological": [
            "Spray neem seed kernel extract (NSKE 5%) or Azadirachtin at first appearance of whiteflies.",
            "Conserve natural predators such as ladybird beetles, lacewings, and parasitoid wasps (*Encarsia* spp.)."
        ]
    },
    "Cotton___fussarium_wilt": {
        "crop": "Cotton",
        "disease": "Fusarium Wilt",
        "overview": "A destructive soil-borne fungal vascular wilt caused by *Fusarium oxysporum f. sp. vasinfectum*, resulting in leaf chlorosis, drooping, wilting, and dark brown vascular browning.",
        "immediate": [
            "Ensure excellent field drainage; prevent standing water around root zones.",
            "Remove and incinerate severely wilted plants to curb fungal spread in soil.",
            "Practice 2- to 3-year crop rotation with non-host crops like sorghum or pearl millet.",
            "Use certified disease-free, acid-delinted seeds."
        ],
        "chemical": {
            "active_ingredient": "Carbendazim 50% WP or Carbendazim + Mancozeb (seed/spot drenching if labeled)",
            "target": "Fusarium oxysporum f. sp. vasinfectum",
            "purpose": "Preventive seed treatment and spot collar drenching to protect healthy surrounding plants",
            "precautions": "Chemical sprays do not cure already wilted plants. Prioritize soil health and root zone drenching."
        },
        "biological": [
            "Incorporate *Trichoderma viride* or *Trichoderma harzianum* enriched in farmyard manure (FYM) into the soil during land preparation.",
            "Apply *Pseudomonas fluorescens* seed coating to boost root zone defense."
        ]
    },
    "Cotton___healthy": {
        "crop": "Cotton",
        "disease": "Healthy Leaf (No Disease Detected)",
        "is_healthy": True,
        "overview": "The cotton leaf exhibits vigorous vegetative health, uniform green color, and no visible symptoms of disease, insect pest damage, or chlorosis.",
        "practices": [
            "Maintain standard balanced N-P-K fertilizer application (avoid excess nitrogen).",
            "Scout fields weekly during square formation, flowering, and boll development stages.",
            "Maintain proper irrigation scheduling and keep field borders weed-free.",
            "Conserve beneficial predators and avoid prophylactic chemical spraying."
        ]
    },
    "Paddy___bacterial_leaf_blight": {
        "crop": "Paddy",
        "disease": "Bacterial Leaf Blight",
        "overview": "Caused by *Xanthomonas oryzae pv. oryzae*, causing wavy, water-soaked yellowish to straw-colored lesions along leaf margins starting from leaf tips.",
        "immediate": [
            "Drain excess water from the field and practice intermittent irrigation.",
            "Temporarily suspend nitrogenous fertilizer top-dressing until disease subsides.",
            "Apply muriate of potash (K2O) to strengthen plant cell walls and leaf cuticle.",
            "Destroy diseased crop stubble and avoid transferring drainage water across fields."
        ],
        "chemical": {
            "active_ingredient": "Copper Oxychloride 50% WP + Streptocycline formulation",
            "target": "Xanthomonas oryzae pv. oryzae",
            "purpose": "Protective contact bactericidal spray to prevent disease spread to neighboring tillers",
            "precautions": "Ensure calm weather during spraying. Apply when wind velocity is low to prevent drift."
        },
        "biological": [
            "Apply *Pseudomonas fluorescens* as seedling root dip and foliar spray.",
            "Incorporate neem cake into soil during final puddling."
        ]
    },
    "Paddy___bacterial_leaf_streak": {
        "crop": "Paddy",
        "disease": "Bacterial Leaf Streak",
        "overview": "Caused by *Xanthomonas oryzae pv. oryzicola*, producing narrow, translucent, interveinal streaks that turn yellow to brownish-orange with tiny bacterial amber droplets.",
        "immediate": [
            "Avoid deep standing water; maintain alternate wetting and drying (AWD).",
            "Clear weeds on field bunds that act as alternate grass hosts.",
            "Split nitrogen applications into 3–4 balanced doses rather than a single heavy dose."
        ],
        "chemical": {
            "active_ingredient": "Copper Oxychloride 50% WP (or approved copper bactericide)",
            "target": "Xanthomonas oryzae pv. oryzicola",
            "purpose": "Preventive surface protective spray against bacterial lesion expansion",
            "precautions": "Follow label guidelines carefully. Do not apply during extreme midday heat."
        },
        "biological": [
            "Seed treatment with *Pseudomonas fluorescens* (10 g/kg seed).",
            "Maintain soil organic matter with well-rotted compost."
        ]
    },
    "Paddy___bacterial_panicle_blight": {
        "crop": "Paddy",
        "disease": "Bacterial Panicle Blight",
        "overview": "Caused by *Burkholderia glumae*, affecting rice panicles during heading and flowering, leading to florets discoloration, grain abortion, and erect sterile panicles.",
        "immediate": [
            "Avoid late planting to prevent panicle emergence during peak high nighttime temperature periods.",
            "Maintain optimal shallow water depth during flowering and grain filling.",
            "Rogue out severely blighted panicles before harvest to reduce seed-borne inoculum."
        ],
        "chemical": {
            "active_ingredient": "Valid registered copper-based or approved bactericide formulation for rice panicle protection",
            "target": "Burkholderia glumae",
            "purpose": "Preventive panicle spray at early boot to 10% flowering stage",
            "precautions": "Observe strict pre-harvest interval (PHI). Avoid spraying during peak pollination hours (10 AM - 1 PM)."
        },
        "biological": [
            "Use certified disease-free seeds and hot water seed treatment (52–54°C for 15 mins).",
            "Foliar application of beneficial microbial antagonists like *Bacillus subtilis*."
        ]
    },
    "Paddy___blast": {
        "crop": "Paddy",
        "disease": "Blast",
        "overview": "A destructive fungal disease caused by *Magnaporthe oryzae* (Pyricularia oryzae), producing diamond/spindle-shaped lesions with gray centers and brown borders on leaves, collars, nodes, and panicles.",
        "immediate": [
            "Avoid excessive nitrogen application; apply nitrogen in split doses with adequate potash.",
            "Maintain continuous 2–5 cm water depth in field (avoid soil drying during blast outbreaks).",
            "Destroy diseased plant residue and burnt stubble from previous harvests.",
            "Avoid planting highly susceptible cultivars in blast-endemic pockets."
        ],
        "chemical": {
            "active_ingredient": "Tricyclazole 75% WP or Isoprothiolane 40% EC or Azoxystrobin 23% SC",
            "target": "Magnaporthe oryzae (Rice Blast)",
            "purpose": "Systemic protective and curative fungicide spray to stop lesion expansion and neck blast",
            "precautions": "Spray at initial appearance of leaf blast spots or at boot leaf stage to prevent neck blast. Use protective mask."
        },
        "biological": [
            "Seed treatment and seedling dip with *Pseudomonas fluorescens* (10 g/kg seed / 2.5 kg/ha in nursery).",
            "Foliar spray of *Pseudomonas fluorescens* suspension at tillering and panicle emergence."
        ]
    },
    "Paddy___brown_spot": {
        "crop": "Paddy",
        "disease": "Brown Spot",
        "overview": "Caused by the fungal pathogen *Bipolaris oryzae*, producing oval, dark brown spots with yellow halos on leaves and glumes, often associated with nutrient-poor or stressed soil.",
        "immediate": [
            "Correct soil nutrient deficiencies by applying balanced NPK plus micronutrients (especially Zinc and Potash).",
            "Ensure regular irrigation to prevent moisture stress and soil cracking.",
            "Use certified disease-free seed and treat seeds before sowing."
        ],
        "chemical": {
            "active_ingredient": "Mancozeb 75% WP or Propiconazole 25% EC or Hexaconazole 5% EC",
            "target": "Bipolaris oryzae (Rice Brown Spot)",
            "purpose": "Protective contact and systemic fungicidal spray",
            "precautions": "Ensure uniform foliar coverage. Wear gloves and mask during spray preparation."
        },
        "biological": [
            "Seed treatment with *Trichoderma viride* (4 g/kg seed).",
            "Apply neem cake (150 kg/ha) at final land preparation."
        ]
    },
    "Paddy___dead_heart": {
        "crop": "Paddy",
        "disease": "Stem Borer (Dead Heart)",
        "overview": "Caused by Yellow Stem Borer larvae (*Scirpophaga incertulas*) boring into the central rice tiller, causing the central shoot to dry up during vegetative stage ('Dead Heart') or white panicles at reproductive stage ('White Ear').",
        "immediate": [
            "Clip seedling leaf tips before transplanting to remove stem borer egg masses.",
            "Install pheromone traps (5 traps/acre for monitoring, 8–10 for mass trapping).",
            "Release egg parasitoid *Trichogramma japonicum* at weekly intervals (1–1.5 lakh/ha).",
            "Conserve natural biological predators like spiders, dragonflies, and mirid bugs."
        ],
        "chemical": {
            "active_ingredient": "Chlorantraniliprole 18.5% SC or Cartap Hydrochloride 50% SP or Flubendiamide 39.35% SC",
            "target": "Scirpophaga incertulas (Yellow Stem Borer larvae)",
            "purpose": "Systemic and contact ingestion insecticide for borer larval management",
            "precautions": "Apply at economic threshold level (ETL: 2 egg masses/sq.m or 5% dead hearts). Do not apply during blooming."
        },
        "biological": [
            "Release *Trichogramma japonicum* egg parasitoids starting from 30 days after transplanting.",
            "Spray neem-based formulations (Azadirachtin 0.03% / 300 ppm) as an oviposition deterrent."
        ]
    },
    "Paddy___downy_mildew": {
        "crop": "Paddy",
        "disease": "Downy Mildew",
        "overview": "Caused by *Sclerophthora macrospora*, causing stunting, excessive tillering, chlorotic streaking, and twisted/deformed 'crazy top' flag leaves under waterlogged conditions.",
        "immediate": [
            "Ensure excellent nursery and main field drainage; prevent prolonged water stagnation.",
            "Rogue out and destroy severely infected, stunted seedlings.",
            "Maintain optimal crop spacing for airflow."
        ],
        "chemical": {
            "active_ingredient": "Metalaxyl 8% + Mancozeb 64% WP (or approved oomycete fungicide)",
            "target": "Sclerophthora macrospora",
            "purpose": "Systemic protective spray against oomycete downy mildew infection",
            "precautions": "Do not exceed recommended label applications per season to prevent resistance development."
        },
        "biological": [
            "Seed treatment with *Trichoderma harzianum* (5 g/kg seed).",
            "Incorporate green manure crops like *Sesbania* (Dhaincha) during pre-puddling."
        ]
    },
    "Paddy___hispa": {
        "crop": "Paddy",
        "disease": "Rice Hispa Pest Damage",
        "overview": "Caused by the spiny beetle *Dicladispa armigera*, where adult beetles scrape the green chlorophyll creating parallel white streaks, and larvae mine between leaf surfaces.",
        "immediate": [
            "Use sweep nets during early morning hours to capture and destroy adult hispa beetles.",
            "Clip the top 2 inches of seedling leaves before transplanting to destroy hispa leaf-mine grubs.",
            "Clear wild host grasses (*Echinochloa*, *Leersia*) from field bunds.",
            "Avoid excessive nitrogen fertilizer application which promotes lush susceptible foliage."
        ],
        "chemical": {
            "active_ingredient": "Chlorpyrifos 20% EC or Quinalphos 25% EC or Thiamethoxam 25% WG",
            "target": "Dicladispa armigera (Rice hispa adults & grubs)",
            "purpose": "Targeted insecticide application when hispa exceeds ETL (1 adult or 1 damaged leaf per hill)",
            "precautions": "Spray during early morning or late afternoon. Avoid spraying when beneficial spiders are abundant."
        },
        "biological": [
            "Conserve natural eulophid parasitoids and spiders in the rice canopy.",
            "Spray neem oil 3% with detergent emulsifier as an antifeedant."
        ]
    },
    "Paddy___normal": {
        "crop": "Paddy",
        "disease": "Normal / Healthy (No Disease Detected)",
        "is_healthy": True,
        "overview": "The rice plant displays healthy, vigorous tillers, rich green foliage, and no noticeable symptoms of fungal leaf blast, bacterial blight, or pest damage.",
        "practices": [
            "Follow standard regional water management: alternate wetting and drying (AWD) to strengthen roots.",
            "Apply balanced nitrogen, phosphorus, and potassium according to crop growth stage.",
            "Conduct weekly field scouting through panicle initiation and grain filling stages.",
            "Keep bunds free of weed hosts and avoid prophylactic chemical sprays."
        ]
    },
    "Paddy___tungro": {
        "crop": "Paddy",
        "disease": "Tungro Virus",
        "overview": "A viral disease complex (RTBV and RTSV) transmitted by the Green Leafhopper (*Nephotettix virescens*), causing vivid yellow-orange leaf discoloration starting from leaf tips, stunted hills, and reduced tillering.",
        "immediate": [
            "Rogue out and bury virus-infected stunted hills at first sighting.",
            "Practice synchronous community planting in the farming block to avoid pest carryover.",
            "Install light traps to monitor and attract nocturnal green leafhoppers.",
            "Keep field bunds clean of alternate host grasses."
        ],
        "chemical": {
            "active_ingredient": "Thiamethoxam 25% WG or Imidacloprid 17.8% SL or Dinotefuran 20% SG (Vector Control)",
            "target": "Nephotettix virescens (Green Leafhopper vector)",
            "purpose": "Vector control spray to stop transmission and secondary spread of the tungro virus",
            "precautions": "Apply at the earliest sign of green leafhopper presence. Always adhere to approved dosage and safety intervals."
        },
        "biological": [
            "Conserve predatory mirid bugs (*Cyrtorhinus lividipennis*) and spiders (*Lycosa pseudoannulata*).",
            "Apply neem formulations (Azadirachtin 1500 ppm) as a feeding deterrent for vector insects."
        ]
    },
    "Chilli___cercospora": {
        "crop": "Chilli",
        "disease": "Cercospora Leaf Spot",
        "overview": "Caused by the fungal pathogen *Cercospora capsici* (Frogeye Leaf Spot), producing circular to oblong lesions with light grey/white centers and dark brown margins, causing severe defoliation.",
        "immediate": [
            "Collect and destroy fallen diseased leaves and crop debris to curb fungal spore load.",
            "Ensure proper plant-to-plant spacing for sunlight penetration and airflow.",
            "Avoid overhead splashing irrigation; switch to drip or furrow irrigation.",
            "Avoid working in the chilli field when plants are wet."
        ],
        "chemical": {
            "active_ingredient": "Mancozeb 75% WP or Difenoconazole 25% EC or Azoxystrobin 18.2% + Difenoconazole 11.4% SC",
            "target": "Cercospora capsici",
            "purpose": "Protective contact and systemic fungicidal spray to arrest spot enlargement and defoliation",
            "precautions": "Spray at first symptom onset. Ensure coverage of both upper and lower leaf surfaces."
        },
        "biological": [
            "Foliar spray of *Trichoderma viride* or *Pseudomonas fluorescens* (5 g/L).",
            "Apply cow urine formulation (10%) or fermented butter milk spray as an organic antifungal measure."
        ]
    },
    "Chilli___healthy": {
        "crop": "Chilli",
        "disease": "Healthy Leaf (No Disease Detected)",
        "is_healthy": True,
        "overview": "The chilli foliage exhibits normal dark-green coloration, healthy leaf architecture, and no visible lesions, powdery patches, or pest curling.",
        "practices": [
            "Maintain optimal drip irrigation and balanced NPK fertigation schedule.",
            "Conduct routine scouting for early signs of thrips, mites, or leaf spot pathogens.",
            "Maintain clean field bunds and install sticky traps for pest monitoring.",
            "Preserve beneficial predator insects and avoid unneeded chemical sprays."
        ]
    },
    "Chilli___murda_complex": {
        "crop": "Chilli",
        "disease": "Murda Complex (Mites & Thrips)",
        "overview": "A multi-pest leaf curling syndrome caused by sap-sucking Chilli Thrips (*Scirtothrips dorsalis*, causing upward boat-shaped leaf curling and brownish scarification) and Yellow Mites (*Polyphagotarsonemus latus*, causing downward curling, brittle leaves, and inverted boat shape).",
        "immediate": [
            "Install blue sticky traps (for thrips) and yellow sticky traps (for whiteflies/aphids) at 10–12 traps/acre.",
            "Remove alternative weed hosts (*Parthenium*, *Amaranthus*) surrounding the field.",
            "Avoid excess nitrogen fertilizer which causes succulent tender growth highly favored by thrips.",
            "Maintain adequate soil moisture; avoid severe dry spells."
        ],
        "chemical": {
            "active_ingredient": "Spiromesifen 22.9% SC or Fenpyroximate 5% EC (for Mites) & Spinetoram 11.7% SC or Fipronil 5% SC (for Thrips)",
            "target": "Scirtothrips dorsalis (Thrips) & Polyphagotarsonemus latus (Yellow mites)",
            "purpose": "Targeted acaricide and insecticide spray to control sap-sucking vectors and stop leaf curling",
            "precautions": "Spray targeted at the underside of leaves using a hollow cone nozzle. Rotate modes of action to prevent resistance."
        },
        "biological": [
            "Spray Neem oil (10,000 ppm) or Pongamia oil at 2–3 ml/L with liquid soap.",
            "Conserve predatory phytoseiid mites (*Amblyseius* spp.) and minute pirate bugs (*Orius* spp.)."
        ]
    },
    "Chilli___nutritional": {
        "crop": "Chilli",
        "disease": "Nutritional Deficiency",
        "overview": "A physiological nutrient imbalance (commonly Nitrogen, Potassium, Magnesium, or Zinc deficiency) causing chlorosis (yellowing), interveinal yellowing, or stunted leaf growth without fungal spots or pest curling.",
        "immediate": [
            "Perform a soil and leaf tissue test to accurately determine deficient macro/micronutrients.",
            "Apply well-decomposed organic farmyard manure (FYM) or vermicompost to improve soil cation exchange.",
            "Ensure consistent soil moisture; avoid dry spells that block nutrient absorption by roots.",
            "Ensure root aeration and avoid waterlogging."
        ],
        "chemical": {
            "active_ingredient": "Water-soluble 19:19:19 NPK foliar spray or Chelated Micronutrient Mixture (Zinc, Iron, Boron, Magnesium)",
            "target": "Plant nutrient deficiency / chlorosis",
            "purpose": "Direct foliar nutrient supplementation for rapid metabolic recovery",
            "precautions": "Do not spray under intense midday sun. Use correct dilution to prevent leaf salt scorching."
        },
        "biological": [
            "Incorporate bio-fertilizers (*Azotobacter*, *Azospirillum*, and Phosphate Solubilizing Bacteria - PSB) into soil.",
            "Apply vermiwash or seaweed extract foliar spray as an organic plant bio-stimulant."
        ]
    },
    "Chilli___powdery_mildew": {
        "crop": "Chilli",
        "disease": "Powdery Mildew",
        "overview": "Caused by *Leveillula taurica*, producing powdery white fungal patches on the lower leaf surface with corresponding bright yellow chlorotic patches on the upper surface, leading to leaf drop.",
        "immediate": [
            "Maintain optimal plant spacing to promote sunlight and cross-ventilation within the canopy.",
            "Collect and destroy severely infected lower foliage to reduce secondary spore dispersal.",
            "Avoid excessive nitrogenous fertilizer application which fosters lush, susceptible leaves."
        ],
        "chemical": {
            "active_ingredient": "Wettable Sulphur 80% WDG or Hexaconazole 5% EC or Myclobutanil 10% WP",
            "target": "Leveillula taurica (Powdery Mildew)",
            "purpose": "Protective and systemic anti-sporulant fungicide spray",
            "precautions": "Do not apply sulphur sprays when ambient temperature exceeds 32°C to prevent phytotoxicity. Ensure thorough coverage on the underside of leaves."
        },
        "biological": [
            "Foliar spray of *Ampelomyces quisqualis* (hyperparasite of powdery mildew).",
            "Spray baking soda (potassium bicarbonate 5 g/L) + neem oil (2 ml/L) as an eco-friendly preventive wash."
        ]
    }
}


def get_soil_management_advice(soil_type, crop):
    """Provides agronomic soil-specific management guidance without determining pesticide choice."""
    st = str(soil_type).strip().title() if soil_type else "Unknown"
    
    if "Black" in st or "Cotton" in st:
        return (
            f"**Black Cotton Soil Considerations for {crop}**:\n"
            "- **Water Retention & Aeration**: Heavy clay texture with high moisture-holding capacity and swelling/shrinking properties. Prone to waterlogging during heavy rain.\n"
            "- **Management**: Create broad bed furrows (BBF) or raised ridges to ensure rapid root zone drainage. Avoid excess irrigation to prevent root hypoxia and soil-borne fungal pathogens.\n"
            "- **Nutrient Availability**: High in calcium and magnesium, but phosphorus and zinc fixation can occur; apply balanced P and Zn."
        )
    elif "Red" in st:
        return (
            f"**Red Soil Considerations for {crop}**:\n"
            "- **Water Retention & Drainage**: Light to medium texture with excellent drainage and aeration, but lower moisture-holding capacity.\n"
            "- **Management**: Split irrigation and nitrogen/potash fertilizer applications into smaller, frequent doses to prevent leaching.\n"
            "- **Organic Matter**: Incorporate farmyard manure or green manure to enhance moisture retention and soil organic carbon."
        )
    elif "Alluvial" in st:
        return (
            f"**Alluvial Soil Considerations for {crop}**:\n"
            "- **Water Retention & Fertility**: Highly fertile, loamy to silty texture with balanced moisture retention.\n"
            "- **Management**: Maintain balanced N-P-K without over-applying nitrogen, which can lead to lush vegetative foliage prone to foliar blights.\n"
            "- **Irrigation**: Follow alternate wetting and drying (for paddy) or controlled furrow irrigation."
        )
    elif "Sandy" in st:
        return (
            f"**Sandy Soil Considerations for {crop}**:\n"
            "- **Water & Leaching**: Rapid water percolation with very low nutrient retention and rapid dry-down.\n"
            "- **Management**: Practice frequent light irrigations (drip preferred) and apply split fertilizer doses with organic compost/mulching."
        )
    elif "Clay" in st:
        return (
            f"**Clay Soil Considerations for {crop}**:\n"
            "- **Drainage Management**: Fine-textured with poor internal aeration when saturated. Ensure peripheral field drainage channels.\n"
            "- **Cultivation**: Avoid working wet clay soil to prevent compaction."
        )
    elif "Loam" in st:
        return (
            f"**Loamy Soil Considerations for {crop}**:\n"
            "- **Ideal Agronomic Soil**: Optimal balance of water retention, drainage, and nutrient availability.\n"
            "- **Management**: Maintain standard recommended irrigation cycles and routine crop scouting."
        )
    else:
        return (
            f"**General Soil Health Guidance for {crop}**:\n"
            "- Ensure proper field drainage to prevent prolonged water stagnation around root zones.\n"
            "- Apply well-decomposed organic manure (FYM) to improve soil structure, microbial health, and nutrient buffering."
        )


def get_growth_stage_advice(growth_stage, crop):
    """Provides growth-stage specific agronomic advice."""
    gs = str(growth_stage).strip().title() if growth_stage else "Unknown"
    
    if "Seedling" in gs:
        return (
            f"**Seedling Stage Guidance ({crop})**:\n"
            "- Young, tender tissues are most vulnerable to damping-off, sucking pests, and early blights.\n"
            "- Maintain clean nursery beds, ensure optimal soil moisture without saturation, and clip diseased seedling tips before transplanting."
        )
    elif "Vegetative" in gs:
        return (
            f"**Vegetative Stage Guidance ({crop})**:\n"
            "- Focus on balanced canopy development. Avoid excess nitrogen which creates dense, humid canopies favoring foliar pathogens.\n"
            "- Scout leaf margins and undersides weekly for early symptom detection."
        )
    elif "Flowering" in gs:
        return (
            f"**Flowering Stage Guidance ({crop})**:\n"
            "- Critical phase for yield determination. Protect blooms from flower blights and sucking insects.\n"
            "- Avoid overhead splashing or high-pressure sprays during peak morning pollination hours."
        )
    elif "Fruiting" in gs or "Boll" in gs or "Grain" in gs:
        return (
            f"**Fruiting / Boll / Grain Development Stage ({crop})**:\n"
            "- Ensure adequate potassium nutrition for fruit/grain development.\n"
            "- Maintain strict pre-harvest intervals (PHI) if any crop protection measures are applied."
        )
    elif "Maturity" in gs or "Harvest" in gs:
        return (
            f"**Maturity & Pre-Harvest Stage ({crop})**:\n"
            "- Withhold irrigation ahead of harvest to facilitate uniform ripening.\n"
            "- Observe mandatory pre-harvest withholding periods and prepare clean storage facilities."
        )
    else:
        return (
            f"**General Crop Stage Guidance ({crop})**:\n"
            "- Conduct regular weekly scouting through all vegetative and reproductive stages to catch pest and disease onset early."
        )


def get_severity_advice(severity):
    """Provides actions based on observed disease severity."""
    sev = str(severity).strip() if severity else "Unknown"
    
    if "Low" in sev or "<10" in sev or "1-10" in sev:
        return (
            "**Low Severity (<10%) Action Plan**:\n"
            "- **Action**: Prioritize field scouting, physical rogueing of infected leaves, sanitation, and biological/cultural controls.\n"
            "- **Chemical Intervention**: Chemical pesticides are **not required** at this low threshold. Continue monitoring twice a week."
        )
    elif "Moderate" in sev or "10-30" in sev or "10–30" in sev:
        return (
            "**Moderate Severity (10–30%) Action Plan**:\n"
            "- **Action**: Strengthen Integrated Pest Management (IPM) measures. Improve field drainage, canopy aeration, and install sticky traps.\n"
            "- **Chemical Consideration**: If weather conditions remain humid/favorable for pathogen spread and economic threshold is crossed, evaluate targeted protective spray under local agricultural guidance."
        )
    elif "High" in sev or ">30" in sev:
        return (
            "**High Severity (>30%) Action Plan**:\n"
            "- **Action**: Immediate containment protocol required. Rogue severely damaged plants, optimize field drainage, and isolate the infection zone.\n"
            "- **Chemical Intervention**: Consult your local Agricultural Extension Officer or KVK agronomist immediately for emergency approved corrective measures."
        )
    else:
        return (
            "**Severity Assessment Action Plan**:\n"
            "- Assess the percentage of affected crop foliage across multiple field locations to determine if economic threshold levels (ETL) have been reached."
        )


def format_fallback_advisory(raw_class, crop, disease, conf_pct, soil_type="Unknown", growth_stage="Unknown", severity="Unknown"):
    """
    Formats complete structured Indian agricultural advisory following the exact required sections.
    """
    data = FALLBACK_ADVISORIES.get(raw_class)
    soil_adv = get_soil_management_advice(soil_type, crop)
    stage_adv = get_growth_stage_advice(growth_stage, crop)
    sev_adv = get_severity_advice(severity)
    
    safety_note = (
        "Use only agricultural products legally registered/labeled for the specific crop and disease in India. "
        "Follow the product label for dose, dilution, application method, protective equipment, re-entry interval and pre-harvest interval. "
        "Do not mix pesticides unless the product labels permit the combination.\n\n"
        "**Recommended Contacts for Regional Verification**:\n"
        "- Local Krishi Vigyan Kendra (KVK)\n"
        "- State Department of Agriculture Extension Officer\n"
        "- State Agricultural University (SAU) advisory cell"
    )

    if not data or data.get("is_healthy", False):
        # Healthy Crop Management
        overview_text = data.get("overview", "The crop leaf appears healthy, vigorous, and free from disease symptoms.") if data else "The leaf exhibits healthy coloration and structure."
        practices = data.get("practices", [
            "Maintain optimal plant spacing and clean field bunds.",
            "Conduct routine weekly scouting throughout vegetative and reproductive stages.",
            "Apply balanced NPK nutrients and avoid excess nitrogen.",
            "Conserve beneficial natural predators in the field ecosystem."
        ]) if data else []
        
        practices_md = "\n".join([f"- {p}" for p in practices])
        
        return f"""### 🌱 Crop & Disease
- **Crop**: {crop}
- **Detected Condition**: Healthy Leaf (No Disease Detected)
- **Model Confidence**: {conf_pct}
- **Disease Severity**: None / Healthy
- **Growth Stage**: {growth_stage}
- **Soil Type**: {soil_type}

### 🌿 Healthy Crop Management
{overview_text}

**Preventive & Cultural Practices:**
{practices_md}
- **Pesticide / Chemical Management**: **No chemical pesticides are recommended** for healthy crop foliage. Maintain preventive cultural hygiene and field sanitation.

### 🌱 Soil-Specific Management
{soil_adv}

### 🧑🌾 Growth-Stage Advice
{stage_adv}

### 🛡️ Safety & Regulatory Note
{safety_note}"""

    # Disease Crop Management
    overview_text = data["overview"]
    immediate_points = "\n".join([f"- {pt}" for pt in data["immediate"]])
    chem = data["chemical"]
    bio_points = "\n".join([f"- {pt}" for pt in data["biological"]])
    
    chem_section = (
        f"- **Chemical Group / Active Ingredient**: {chem['active_ingredient']}\n"
        f"- **Target Pathogen / Pest**: {chem['target']}\n"
        f"- **Application Purpose**: {chem['purpose']}\n"
        f"- **Important Precautions**: {chem['precautions']}\n"
        f"- **Dosage Note**: Follow the exact dosage and application instructions printed on the currently registered product label.\n"
        f"- **Label Compliance**: Use only approved formulations legally labeled for this crop-disease combination in India."
    )

    return f"""### 🌱 Crop & Disease
- **Crop**: {crop}
- **Detected Disease**: {disease}
- **Model Confidence**: {conf_pct}
- **Disease Severity**: {severity}
- **Growth Stage**: {growth_stage}
- **Soil Type**: {soil_type}

### 🦠 Disease Overview
{overview_text}

### 🌾 Immediate Management
{immediate_points}

### 🧪 Pesticide / Chemical Management
{chem_section}

### 🌿 Biological / Organic Management
{bio_points}

### 🌱 Soil-Specific Management
{soil_adv}

### 🧑🌾 Growth-Stage Advice
{stage_adv}

### ⚠️ Severity-Based Action
{sev_adv}

### 🛡️ Safety & Regulatory Note
{safety_note}"""


ADVISORIES = FALLBACK_ADVISORIES



def generate_ai_advisory(crop, disease, confidence, soil_type="Unknown", growth_stage="Unknown", severity="Unknown", api_key=None):
    """
    Generates a dynamic, India-contextualized agricultural advisory using Gemini API.
    Returns None on failure or if no API key is present.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key or not str(api_key).strip():
        return None
        
    api_key_clean = str(api_key).strip()
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key_clean)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""You are an India-focused agricultural advisory assistant. Provide practical Integrated Pest Management (IPM) guidance for the detected crop disease.

Use the detected disease as the PRIMARY basis for chemical management.
Use soil type, growth stage and disease severity as CONTEXT for management recommendations.

Do not invent pesticide products, active ingredients, dosages, concentrations, waiting periods, or registrations.
Only mention chemical active ingredients/products when they are appropriate for the identified crop-disease combination and clearly state that the farmer must verify the current Indian label/registration and follow the product label and local agricultural department recommendations.
Do not recommend pesticides merely because of soil type.

If a reliable chemical recommendation cannot be established, explicitly say:
'No specific chemical recommendation is provided here. Please verify the currently registered product and label for this crop-disease combination with the local agricultural extension officer.'

Prefer Integrated Pest Management and non-chemical practices before chemical intervention.

The response MUST strictly follow this exact structure in clean Markdown:

### 🌱 Crop & Disease
- **Crop**: {crop}
- **Detected Disease**: {disease}
- **Model Confidence**: {confidence}
- **Disease Severity**: {severity}
- **Growth Stage**: {growth_stage}
- **Soil Type**: {soil_type}

### 🦠 Disease Overview
- **Pathogen / Nature**: [Brief description of the causal organism/condition]
- **Main Symptoms**: [Visual leaf/plant symptoms]
- **Favorable Conditions**: [Weather/field conditions that promote disease development]

### 🌾 Immediate Management
- [Immediate containment actions]
- [Removal/sanitation practices]
- [Water and irrigation management advice]
- [Field sanitation practices]

### 🧪 Pesticide / Chemical Management
If an appropriate chemical management option can be safely stated:
- **Chemical Group / Active Ingredient**: [Approved active ingredient in India, e.g. Mancozeb, Copper Oxychloride, Tricyclazole, Streptocycline, Hexaconazole, etc.]
- **Target Pathogen / Pest**: [Target pathogen/pest]
- **Application Purpose**: [Protective or curative action]
- **Important Precautions**: [Spray timing, avoiding drift, personal protective gear]
- **Dosage Note**: Follow the exact dosage and application instructions printed on the currently registered product label.
- **Label Compliance**: Use only approved formulations legally labeled for this crop-disease combination in India.

If no reliable chemical recommendation can be given:
"No specific chemical recommendation is provided. Verify currently registered products for this crop-disease combination with the local agricultural extension officer."

### 🌿 Biological / Organic Management
- [Relevant biological control, e.g. Trichoderma, Pseudomonas fluorescens, Bacillus, or Neem-based formulations]
- [Organic and cultural practices]

### 🌱 Soil-Specific Management
Explain how the selected soil type ({soil_type}) affects drainage, aeration, moisture retention, and nutrient availability for this crop. (Do not claim soil type determines pesticide).

### 🧑🌾 Growth-Stage Advice
Specific management guidance tailored to the {growth_stage} stage of {crop}.

### ⚠️ Severity-Based Action
Specific actions adapted to {severity} severity level:
- If Low (<10%): Prioritize monitoring, sanitation, and cultural/biological management.
- If Moderate (10–30%): Strengthen IPM and explain when chemical intervention may be considered.
- If High (>30%): Urgent containment steps and mandatory consultation with local agricultural experts before spraying.

### 🛡️ Safety & Regulatory Note
Use only agricultural products legally registered/labeled for the specific crop and disease in India. Follow the product label for dose, dilution, application method, protective equipment, re-entry interval and pre-harvest interval. Do not mix pesticides unless the product labels permit the combination. Consult your State Agriculture Department, Krishi Vigyan Kendra (KVK), local agricultural extension officer, or qualified agronomist.
"""
        response = model.generate_content(prompt)
        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        return None
    except Exception as e:
        # Gracefully handle any API errors without leaking sensitive information
        print(f"Gemini API advisory unavailable: {type(e).__name__}")
        return None


def load_model_and_classes(model_path=None, class_names_path=None):
    """
    Loads and caches the trained MobileNetV2 Keras model and class names.
    """
    global _CACHED_MODEL, _CACHED_CLASSES

    if _CACHED_MODEL is not None and _CACHED_CLASSES is not None:
        return _CACHED_MODEL, _CACHED_CLASSES

    m_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    c_path = Path(class_names_path) if class_names_path else DEFAULT_CLASSES_PATH

    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found at: {m_path}. Please check project-fai/models/.")

    if not c_path.exists():
        raise FileNotFoundError(f"Class names file not found at: {c_path}.")

    # Load class names
    with open(c_path, "r") as f:
        _CACHED_CLASSES = json.load(f)

    # Load Keras model
    _CACHED_MODEL = tf.keras.models.load_model(str(m_path))
    return _CACHED_MODEL, _CACHED_CLASSES


def format_readable_label(raw_class_name):
    """
    Converts raw labels like 'Paddy___brown_spot' or 'Chilli___powdery_mildew' into readable Crop & Disease names.
    """
    if "___" in raw_class_name:
        crop, disease = raw_class_name.split("___", 1)
        disease_formatted = disease.replace("_", " ").title()
        if disease_formatted == "Fussarium Wilt":
            disease_formatted = "Fusarium Wilt"
        elif disease_formatted == "Normal":
            disease_formatted = "Normal / Healthy"
        elif disease_formatted == "Murda Complex":
            disease_formatted = "Murda Complex"
        return crop.capitalize(), disease_formatted
    return "Unknown", raw_class_name.replace("_", " ").title()


def preprocess_image_input(image_input):
    """
    Preprocesses PIL Image, numpy array, or file path:
    1. Converts image to RGB
    2. Resizes to 224x224
    3. Converts to float32 numpy array [0.0 - 255.0]
    4. Adds batch dimension (1, 224, 224, 3)
    (Note: The model includes the internal MobileNetV2 preprocess_input layer)
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype(np.uint8))
    else:
        raise ValueError("Unsupported image format. Expected PIL Image, file path, or numpy array.")

    # Convert to RGB (handles RGBA or grayscale)
    img_rgb = img.convert("RGB")
    
    # Resize to model expected dimensions (224, 224)
    img_resized = img_rgb.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    
    # Convert to float32 numpy array [0, 255]
    img_array = np.array(img_resized, dtype=np.float32)
    
    # Add batch dimension: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def is_likely_leaf_image(image_input):
    """
    Lightweight heuristic to check if the image is likely a crop leaf before neural network inference.
    Checks:
    1. Brightness & Variance (reject dark, blank, flat, or overexposed images)
    2. Color Space & Channel Analysis (HSV hue, saturation, value, and RGB channel relationships)
    3. Vegetation Coverage & Central Distribution (ensure leaf occupies substantial central region)
    4. Human Skin-Tone / Non-Crop Dominance Check (detect person/face/indoor colors in central region)
    5. Spatial Quadrant Distribution (ensure vegetation is not just an edge/peripheral artifact)
    Returns: (is_valid: bool, status: str, reason: str)
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype(np.uint8))
    else:
        return False, "Invalid image", "Unsupported image format."

    # Convert to RGB and resize to a small standard size for fast processing
    img_rgb = img.convert("RGB").resize((100, 100))
    arr_rgb = np.array(img_rgb, dtype=np.float32)
    
    # 1. Variance & Brightness Checks
    std_dev = float(np.std(arr_rgb))
    mean_brightness = float(np.mean(arr_rgb))
    
    if std_dev < 5.0:
        return False, "Image may not contain a crop leaf", "Image has too little variance (blank or flat)."
    
    if mean_brightness < 10.0:
        return False, "Image may not contain a crop leaf", "Image is too dark / underexposed."
        
    if mean_brightness > 250.0 and std_dev < 10.0:
        return False, "Image may not contain a crop leaf", "Image is overexposed / blank white."

    # 2. Color Space & Channel Analysis
    arr_norm = arr_rgb / 255.0
    arr_hsv = rgb_to_hsv(arr_norm)
    hue = arr_hsv[..., 0] # [0, 1]
    sat = arr_hsv[..., 1] # [0, 1]
    val = arr_hsv[..., 2] # [0, 1]
    r, g, b = arr_norm[..., 0], arr_norm[..., 1], arr_norm[..., 2]

    # Human skin-tone heuristic (R > G > B, reddish-orange hue [0.00-0.09] or [0.93-1.00])
    is_skin = ((hue <= 0.09) | (hue >= 0.93)) & (sat >= 0.12) & (sat <= 0.80) & (val >= 0.20) & (r > g * 1.05) & (g > b * 1.05)
    center_skin_ratio = float(np.mean(is_skin[20:80, 20:80]))

    # True Leaf Vegetation heuristic:
    # Green to yellow-green / olive (Hue 0.09 to 0.50), with green channel active
    is_veg = (hue >= 0.09) & (hue <= 0.50) & (sat >= 0.08) & (val >= 0.08) & (~is_skin) & ((g >= r * 0.80) | (hue >= 0.13))

    overall_veg_ratio = float(np.mean(is_veg))
    center_veg_ratio = float(np.mean(is_veg[25:75, 25:75]))

    # 3. Overall & Central Coverage Check
    if overall_veg_ratio < 0.07:
        return False, "Image may not contain a crop leaf", f"Image lacks sufficient vegetation content (overall coverage: {overall_veg_ratio:.1%})."
        
    if center_veg_ratio < 0.05:
        return False, "Image may not contain a crop leaf", f"Central region lacks sufficient crop leaf content (center coverage: {center_veg_ratio:.1%})."

    # 4. Human / Person / Non-Crop Dominance Check in Central Subject Region
    if center_skin_ratio > 0.18 and center_skin_ratio > (center_veg_ratio * 0.7):
        return False, "Image may not contain a crop leaf", f"Image appears to contain a person or non-crop subject (center skin/subject: {center_skin_ratio:.1%}, veg: {center_veg_ratio:.1%})."

    # 5. Spatial Quadrant Distribution (Ensure leaf is not just a tiny peripheral corner artifact)
    q1 = float(np.mean(is_veg[:50, :50]))
    q2 = float(np.mean(is_veg[:50, 50:]))
    q3 = float(np.mean(is_veg[50:, :50]))
    q4 = float(np.mean(is_veg[50:, 50:]))
    active_quads = sum([1 for q in [q1, q2, q3, q4] if q >= 0.04])
    
    if active_quads <= 1 and center_veg_ratio < 0.10:
        return False, "Image may not contain a crop leaf", "Vegetation is concentrated in a tiny peripheral region and not the central subject."

    return True, "Valid crop-leaf image", f"Image passed leaf validation (vegetation: {overall_veg_ratio:.1%}, center: {center_veg_ratio:.1%})."


def predict_disease(image_input, soil_type="Unknown", growth_stage="Unknown", severity="Unknown", api_key=None):
    """
    Main prediction pipeline:
    Input: Leaf image, and optional field conditions.
    Output: Dictionary with validation status, predicted crop, disease, confidence, top 3 predictions, and advisory.
    """
    if image_input is None:
        return {
            "success": False,
            "error": "Please upload a crop leaf image."
        }

    # Lightweight Image Validation
    is_valid, validation_status, validation_reason = is_likely_leaf_image(image_input)

    # If image fails basic validation, return early
    if not is_valid:
        return {
            "success": True,
            "is_valid": False,
            "validation_status": validation_status,
            "validation_reason": validation_reason,
            "raw_class": None,
            "crop": "—",
            "disease": "—",
            "confidence_val": 0.0,
            "confidence_pct": "—",
            "is_confident": False,
            "status_text": validation_status,
            "confidence_status": f"⚠️ **Image Validation Warning**",
            "confidence_note": "No pesticide or disease-specific advisory is provided because the uploaded image does not appear to contain a clear crop leaf.",
            "top_3": [],
            "advisory": None
        }

    try:
        model, class_names = load_model_and_classes()
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

    try:
        img_batch = preprocess_image_input(image_input)
    except Exception:
        return {
            "success": False,
            "error": "Unable to process this image. Please upload a valid image."
        }

    # Pass image through trained model (outputs softmax probabilities)
    probabilities = model.predict(img_batch, verbose=0)[0]

    # Find class with highest probability (argmax)
    top_idx = int(np.argmax(probabilities))
    raw_class = class_names[top_idx]
    confidence_val = float(probabilities[top_idx])
    confidence_pct = confidence_val * 100.0

    # Format readable crop and disease names
    crop_name, disease_name = format_readable_label(raw_class)

    # Extract Top 3 predictions
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    top_3_list = []
    for rank, idx in enumerate(top_3_indices, start=1):
        c_name, d_name = format_readable_label(class_names[idx])
        pct = float(probabilities[idx]) * 100.0
        top_3_list.append({
            "rank": rank,
            "raw_class": class_names[idx],
            "display_name": f"{c_name} - {d_name}",
            "percentage": f"{pct:.2f}%",
            "probability": pct / 100.0
        })

    # Check confidence threshold (60.0%)
    is_confident = confidence_val >= CONFIDENCE_THRESHOLD
    if is_confident:
        status_text = "Reasonable confidence"
        confidence_status = "✅ Reasonable confidence"
        confidence_note = f"The model identified this sample with {confidence_pct:.2f}% confidence."
        
        is_healthy = ("Healthy" in disease_name) or ("Normal" in disease_name)
        ai_advisory_text = None
        
        # Only call Gemini if confident AND it is a disease (healthy does not call Gemini)
        if not is_healthy:
            ai_advisory_text = generate_ai_advisory(
                crop=crop_name,
                disease=disease_name,
                confidence=f"{confidence_pct:.2f}%",
                soil_type=soil_type,
                growth_stage=growth_stage,
                severity=severity,
                api_key=api_key
            )
        
        if ai_advisory_text:
            advisory_info = {
                "title": f"✨ AI-Powered Advisory: {crop_name} - {disease_name}",
                "crop": crop_name,
                "disease": disease_name,
                "is_ai": True,
                "ai_text": ai_advisory_text
            }
        else:
            # Format comprehensive fallback advisory
            fallback_text = format_fallback_advisory(
                raw_class=raw_class,
                crop=crop_name,
                disease=disease_name,
                conf_pct=f"{confidence_pct:.2f}%",
                soil_type=soil_type,
                growth_stage=growth_stage,
                severity=severity
            )
            advisory_info = {
                "title": f"📋 {crop_name} - {disease_name} (Reference Advisory)",
                "crop": crop_name,
                "disease": disease_name,
                "is_ai": False,
                "ai_text": fallback_text
            }
    else:
        status_text = "Low confidence / uncertain input"
        confidence_status = "⚠️ Low confidence / uncertain input"
        confidence_note = (
            "No disease-specific pesticide recommendation is provided because the model confidence is below the 60% threshold.\n\n"
            "Please upload a clear close-up image of the crop leaf."
        )
        advisory_info = None

    return {
        "success": True,
        "is_valid": is_valid,
        "validation_status": validation_status,
        "validation_reason": validation_reason,
        "raw_class": raw_class,
        "crop": crop_name,
        "disease": disease_name,
        "confidence_val": confidence_val,
        "confidence_pct": f"{confidence_pct:.2f}%",
        "is_confident": is_confident,
        "status_text": status_text,
        "confidence_status": confidence_status,
        "confidence_note": confidence_note,
        "top_3": top_3_list,
        "advisory": advisory_info
    }


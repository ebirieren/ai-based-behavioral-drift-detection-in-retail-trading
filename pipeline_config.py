from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

DATASET_PATH = DATA_DIR / "behavioraldriftdetectiondataset.xlsx"
RESULTS_PATH = OUTPUT_DIR / "behavioral_drift_results.xlsx"
POST_TRADE_MODEL_PATH = MODEL_DIR / "behavior_model_post_trade.pkl"
LIVE_MODEL_PATH = MODEL_DIR / "behavior_model_live.pkl"
MODEL_BUNDLE_PATH = MODEL_DIR / "behavior_model_bundle.pkl"


BEHAVIOR_LABEL_COLUMNS = [
    "Have you stick to your plan?",
    "Have you move your Stop/TP",
    "Do you think that your entry is okay?",
    "Do you think that your stop-loss is okay?",
    "Do you think that your TP is okay?",
    "Poor Risk/Reward Trade",
    "Entered Too Soon",
    "Entered Too Late",
    "Exited Too Soon",
    "Exited Too Late",
    "Traded not in Trading Plan",
    "Incorrect Stop Placement",
    "Wrong Position Size",
    "Didn't Take Planned Trade",
]


RULE_TARGETS = [
    "Entered Too Soon",
    "Entered Too Late",
    "Exited Too Soon",
    "Exited Too Late",
    "Traded not in Trading Plan",
    "Incorrect Stop Placement",
    "Didn't Take Planned Trade",
]


TEXT_RULES = {
    "Entered Too Soon": [
        r"\bfomo\b",
        r"\bisleme atlamak\b",
        r"\bhicbir sey yokken\b",
        r"\bhicbir yapi yokken\b",
        r"\bbir yapi yok\b",
        r"\bhicbir sey yok\b",
        r"\byapisi yok\b",
        r"\byapisi olusmamis\b",
        r"\bdirekt girilmesi yanlis\b",
        r"\basilmadan\b",
    ],
    "Entered Too Late": [
        r"\bgec kalinmis\b",
        r"\bgec giris\b",
        r"\bis bitmesine ragmen\b",
        r"\bcok gec\b",
    ],
    "Exited Too Soon": [
        r"\bhemen kapatilmis\b",
        r"\bpanik.*kapat\b",
        r"\bislemde kalamiyorsun\b",
        r"\boynamasaydim hedef.*gidecekti\b",
        r"\bhedefine kadar tutulmaliydi\b",
    ],
    "Exited Too Late": [
        r"\bcok gec kapat\b",
        r"\bgec kapat\b",
        r"\bgec cikis\b",
        r"\bcikis gec\b",
    ],
    "Traded not in Trading Plan": [
        r"\bdurtusel\b",
        r"\bfiyat okunmamis\b",
        r"\byanlis islem\b",
        r"\bbiasima uymayip\b",
        r"bias.*tersine",
        r"\bplansiz\b",
        r"\bplan disi\b",
    ],
    "Incorrect Stop Placement": [
        r"\bstop yanlis\b",
        r"\byanlis yere stop\b",
        r"\bstopunla oynuyor\b",
        r"\bstop placement yanlis\b",
        r"\bstop kotu\b",
    ],
    "Didn't Take Planned Trade": [
        r"\bplanima sadik kalmadim\b",
        r"\bhedefime gidecekti\b",
        r"\bkacirdim\b",
        r"\balmadim\b",
        r"\bislemi almadim\b",
        r"\bplanned trade almadim\b",
    ],
}

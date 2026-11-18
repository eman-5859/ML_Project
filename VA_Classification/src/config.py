from pathlib import Path

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
FIG_DIR = ROOT_DIR / "figures"
OUTPUT_DIR = ROOT_DIR / "outputs"

RAW_DATA_PATH = DATA_DIR / "AMD_longitudinal_MEH_v4.csv"
TRAIN_PATH = DATA_DIR / "Train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# ---- Constants ----
TARGET_COL = "grouped_VA"
GROUP_COL = "ID"                  # patient identifier: used for splitting only, never a feature
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---- Columns ----
BIOMARKER_COLS = [
    "Neurosensory", "RPE", "IRF", "SRF", "SHRM",
    "Drusenoid_PED", "Serous_PED", "Fibrovascular_PED",
]
CATEGORICAL_COLS = ["Eye", "FirstTreatedEye", "Gender", "grouped_ethnicity"]
ORDINAL_COLS = ["grouped_age"]
TIME_COL = "DaysSinceBaseline"    # ablation candidate; excluded by default
USE_TIME_FEATURE = False

# ---- Ordered categories (worst -> best vision / youngest -> oldest) ----
VA_CLASSES = ["0-35", "36-52", "53-69", "ge70"]
AGE_ORDER = ["50-59", "60-69", "70-79", "ge80"]
ETHNICITY_VALUES = ["White", "Asian", "Black", "Other or unknown"]
EYE_VALUES = ["Left", "Right"]
FIRST_TREATED_VALUES = ["Left", "Right", "Both"]
GENDER_VALUES = ["Female", "Male"]

# Make sure output folders exist
for _d in (DATA_DIR, MODEL_DIR, FIG_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
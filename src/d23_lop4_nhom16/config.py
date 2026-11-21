from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
RAW_DATASET_DIR = PROJECT_ROOT / "data_collection"  
METADATA_CSV_PATH = PROJECT_ROOT / "src" / "d23_lop4_nhom16" / "features" / "metadata.csv"
HSV_FEATURE_PATH = PROJECT_ROOT / "src" / "d23_lop4_nhom16" / "features" / "hsv_features.pkl" 
GLCM_FEATURE_PATH = PROJECT_ROOT / "src" / "d23_lop4_nhom16" / "features" / "glcm_features.pkl"
CHUNKSIZE=100 
TABLE_VIDEO = "video"
DB_CONFIG = {
    "host": "localhost", 
    "port": 5432, 
    "dbname": "dog_video_db", 
    "user": "ninh_dao", 
    "password": "123" 
}

DEFAULT_WEIGHTS = {
    "hsv": 0.15, 
    "glcm": 0.10, 
    "hog": 0.20, 
    "flow": 0.25, 
    "mhi": 0.3 
}

POSTGRES_DIR = (
    Path("/home/ninhdao/D23_Lop4_Nhom16/src/d23_lop4_nhom16/postgres")
    if Path(
        "/home/ninhdao/D23_Lop4_Nhom16/src/d23_lop4_nhom16/postgres"
    ).is_dir()
    else Path("./postgres")
)
def get_db_url(): 
    c = DB_CONFIG 
    return f"postgresql+psycopg2://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['dbname']}"






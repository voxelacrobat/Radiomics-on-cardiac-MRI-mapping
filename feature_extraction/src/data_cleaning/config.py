from src.configuration import Configuration

cfg = Configuration()
BASE_DIR = cfg.root_dir / 'Radiomics'
DATA_DIR = cfg.img_dir
DATA_OUT_DIR = BASE_DIR / "data"
TABLE_DIR = BASE_DIR / "tables"
RESULT_DIR = BASE_DIR / "results"
KERNEL_RADIUS = 0
KERNEL_RADIUSES = range(1, 6)
CNN_DEPTH = 3
CNN_DEPTHS = range(0, 5)

# mkdirs if not already there
for dir in [BASE_DIR, DATA_DIR, TABLE_DIR, RESULT_DIR]:
    dir.mkdir(parents=True, exist_ok=True)


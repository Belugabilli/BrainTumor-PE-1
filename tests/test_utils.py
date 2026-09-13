from utils.config import Config
from utils.seed import set_seed

cfg = Config("configs/classification.yaml")

print(cfg["model"]["backbone"])

set_seed(42)

print("Utilities Working")
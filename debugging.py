import json
from crossformer.model.crossformer import CrossFormer
from crossformer.data_tools.data_interface import DataInterface
import pandas as pd
# debugging test entry
if __name__ == "__main__":
    json_path = "/Users/paulwu/Documents/GitHub/Surrey_AI/cfg.json"
    with open(json_path) as f:
        cfg = json.load(f)
    print(cfg)
    df = pd.read_csv(cfg["csv_path"])
    data = DataInterface(df, **cfg)
    data.setup()
    model = CrossFormer(cfg=cfg)
    
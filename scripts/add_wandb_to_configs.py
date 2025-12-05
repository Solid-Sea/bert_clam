import json
from pathlib import Path

configs_dir = Path("configs")
for config_file in configs_dir.glob("*.json"):
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    if "use_wandb" not in config:
        config["use_wandb"] = True
        config["wandb_project"] = "bert-clam"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Updated: {config_file.name}")
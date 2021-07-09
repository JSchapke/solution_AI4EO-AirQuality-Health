import yaml

def save_config(config, path):
    with open(path, "w") as f:
        yaml.dump(f)
        
def load_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config
        

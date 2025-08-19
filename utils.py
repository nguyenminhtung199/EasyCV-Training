import importlib
import os.path as osp
import os
import logging
import yaml
import json

def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    # config = importlib.import_module("configs.base")
    # cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config

    return job_cfg

def seconds_to_text(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours} hours {minutes} minutes {secs} seconds"
    elif minutes > 0:
        return f"{minutes} minutes {secs} seconds"
    else:
        return f"{secs} seconds"
    

def setup_logging(save_path):
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(save_path, exist_ok=True)
    
    # Đường dẫn lưu tệp log
    log_file_path = os.path.join(save_path, "training.log")

    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,  
        format='%(asctime)s - %(levelname)s: %(message)s', 
        handlers=[
            logging.StreamHandler(),  
            logging.FileHandler(log_file_path) 
        ]
    )

    logger = logging.getLogger(__name__)  
    return logger

def config2yaml(cfg):
    class yamlDumper(yaml.Dumper):
        def represent_list(self, data):
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

        def represent_dict(self, data):
            return self.represent_mapping('tag:yaml.org,2002:map', data)

    yamlDumper.add_representer(list, yamlDumper.represent_list)
    yamlDumper.add_representer(dict, yamlDumper.represent_dict)

    json_str = json.dumps(cfg, indent=2, ensure_ascii=False)

    cfg_clean = json.loads(json_str)

    return yaml.dump(cfg_clean, Dumper=yamlDumper, sort_keys=False, default_flow_style=False, allow_unicode=True)
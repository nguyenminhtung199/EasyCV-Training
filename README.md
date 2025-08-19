# **Model Training for Classification**

This repository provides a framework for training deep learning classification models using PyTorch. The codebase supports single-task and multi-task classification, with configurable options for models, datasets, training parameters.
---

## **Features**

- Supports **single-task** and **multi-task** classification.
- Random 6 types of augmentation data.
- Modular design for easy extension.
- Convert and inference ONNX model
---

## **Requirements**

- Python 3.8+
- PyTorch 1.12+
- torchvision
- NumPy
- Pillow
- onnxruntime

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/classification-trainer.git
   cd classification-trainer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## **Configuration**

The training process is fully configurable through a configuration file (`config.py`). Below is an example configuration:

```py
dataset_path: "../data/"
save_path: "weights/model_v1"
name: "finetuned_model_mobile_netv2_"
model: "mbf"  # Options: "mbf", "resnet18", etc.
multitask: false
model_type: "person_body_walking"
batch_size: 64
phases: ["train", "validation"]
criterion: "CrossEntropyLoss"
optimizer: "SGD"
lr: 0.001
momentum: 0.9
step_size: 10
gamma: 0.1
num_epochs: 50
num_output: 3
```

### Key Configuration Options:
- **`dataset_path`**: Path to your dataset directory.
- **`save_path`**: Folder to save the model weights and logs.
- **`model`**: Model architecture (e.g., `mbn` for MobileNetV2).
- **`multitask`**: Whether the model performs multitask classification.
- **`num_epochs`**: Number of training epochs.
- Model with save in `save_path/name+model_type.pt`
---

## **Usage**

### **1. Prepare Dataset**
Organize your dataset in the following structure: 
- 1 task:
```
data/
├── model_type/
│   ├── train/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   └── ...
│   ├── validation/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   └── ...
```
- Multiple task: example in `data/multitask_example`
```
data/
├── model_type/
│   ├── train.txt
│   ├── validation.txt
```
### **2. Update Configuration**
Edit `configs/config.py` to set your dataset path, model, and training parameters.

### **3. Start Training**
Run the training script:
```bash
python train.py -c configs/config.py
```

### **4. Export ONNX**
Run the convert ONNX: 
```bash 
python onnx_convert -f model_path -m False
```

### **5. Inference ONNX**
Add *`model_path`* and *`image_path`* in `inference.py` and run: 
```bash
python inference.py
```
---
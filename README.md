# VQA V1 PyTorch Implementation

This repository contains the code and data necessary to replicate the experiments from the paper VQA: Visual Question Answering. The aim is to develop a model capable of answering questions about images, similar to the system presented in the paper.
#### **Model architecture**
![model](./setup/vqa_architecture.png)
## Steps to Run

1. **Download Data**
   ```bash
   python download_data.py
2. **Preprocess Images**
    ```bash
    python preprocess_image.py
3. **Create Vocabulary**
    ```bash
    python make_vocabulary.py
4. **Prepare VQA Inputs**
    ```bash
    python make_vqa_inputs.py
    
### This is how the datasets firectory should like after finishing the final preprocessing step
datasets\
├── Annotations\
├── Images\
├── Questions\
├── Resized_Images\
├── test-dev.npy\
├── test.npy\
├── train_valid.npy\
├── train.npy\
├── valid.npy\
├── vocab_answers.txt\
├── vocab_questions.txt\

### **Train the Model**
    ```bash
    python train.py

## **Acknowledgements**

- COCO Dataset: [http://cocodataset.org/](http://cocodataset.org/)
- Paper: [VQA: Visual Question Answering](https://arxiv.org/pdf/1505.00468)





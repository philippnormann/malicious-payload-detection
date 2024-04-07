# Models Download Guide

Due to the substantial size of the models, they cannot be hosted directly in the repository. Instead, they are hosted on Google Drive and can be downloaded from there. The following sections provide a step-by-step guide on how to download the models.

## 📥 Model Download Steps

1. **Access the Model File:**
   - Navigate to the following link or paste it into your browser:
     [Download Models](https://drive.google.com/file/d/1Qfnq2FlyK3675niMK_vPHQ_hntiFrCrx/view?usp=sharing)

2. **Download the Archive:**
   - Click the download button to obtain the `models.zip` file on your local machine.

3. **Extraction & Setup:**
   - Extract the contents of `models.zip`.
   - Make sure that the extracted files are placed in the `models/` directory within the project structure.

## ✅ Verification of the Setup

Your local project directory should mirror the following structure after extraction:

```bash
📁 models                   
│   📁 rf                   # Random Forest model artifacts
│   📁 roberta              # RoBERTa model artifacts
│   📁 codebert             # CodeBERT model artifacts
```

## 📊 Evaluation Notebooks:

To ensure the models are set up correctly, run the following notebooks:

- **Random Forest Model:**
    - Open the `evaluate_rf.ipynb` notebook in the `notebooks/` directory.
    - Execute all cells and ensure that the model’s predictions are displayed in the last section.

- **RoBERTa Model:**
    - Navigate to `evaluate_bert.ipynb` within the `notebooks/` directory.
    - Run all the cells, ensuring that the model’s evaluations are outputted without error.

Executing these notebooks will load the pre-trained models to conduct evaluations, ensuring the models are downloaded and set up correctly. In case of any errors, please reach out to me by creating an issue in the repository.


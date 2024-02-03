

## Packages

To successfully run the project, the following Python packages need to be installed:

```
captum==0.6.0
libclang==16.0.6
numpy==1.26.3
pandas==2.2.0
scikit_learn==1.3.2
tokenizers==0.15.1
torch==2.1.0
tqdm==4.66.1
transformers==4.34.0
```



### Datasets

The dataset (big_vul) is provided by [Fan et al.](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset.git) We focus on the following 3 columns to conduct our experiments:

1. `func_before` (str): The original function written in C/C++.
2. `target` (int): The function-level label that determines whether a function is vulnerable or not.
3. `flaw_line_index` (str): The labeled index of the vulnerable statement. 

| func_before | target | flaw_line_index |
| ----------- | ------ | --------------- |
| ...         | ...    | ...             |



Training, evaluation, and testing datasets can be downloaded from:

```
https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw
https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ
https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V
```

The entire dataset without splitting can be downloaded from:

```
https://drive.google.com/uc?id=1WqvMoALIbL3V1KNQpGvvTIuc3TL5v5Q8
```

After downloading, place all data into `./resource/dataset`.

(Note that these dataset links are provided by the author of [LineVul](https://github.com/awsm-research/LineVul.git), which is an important transformer-based vulnerability detection method and also serves as a crucial baseline for our research.)

## Pre-trained Model

The pre-trained model, using the proposed Masked Statement Prediction (MSP) method, can be downloaded from the following link:

```
https://drive.google.com/file/d/1frZLAmB2F0z1LLEwjVmoAtqKlPMg13uR/view?usp=sharing
```

After downloading, put the model into `./resource/staged-models`.



## Fine-tuned Model

The pre-trained model fine-tuned on the BigVul dataset can be downloaded from the links below:

- **Coarse-grained model:**

```
https://drive.google.com/file/d/1Q4_jjXQydP5oyHsLAWnkhlvEcFjVu4fj/view?usp=sharing
```

- **Fine-grained model:**

```
https://drive.google.com/file/d/1jidokf-f8KOLPleWJSd6qLY_TeqmGPKP/view?usp=sharing
```



## How to run the model

`Entry/StagedBert_vul.py` and `Entry/StagedBert_line_vul.py` are the entry files for coarse-grained and fine-grained training, respectively. Please run these files to reproduce the results presented in our reports.
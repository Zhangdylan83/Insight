## 1. Getting the Dataset

This project uses the following datasets:

- **MosMedData**: [Download here](https://academictorrents.com/details/f2175c4676e041ea65568bb70c2bcd15c7325fd2)
- **Camelyon16**: [Download here](https://huggingface.co/datasets/osbm/camelyon16/tree/main)
- **BRACS**: [Download here](https://www.bracs.icar.cnr.it/)

Please download and organize the datasets appropriately before proceeding to the preprocessing steps.

## 2. Preprocessing and Feature Extraction

### A. CT Volumes

#### 1. Generate Lung Masks

Use the **[LungMask](https://github.com/JoHof/lungmask)** tool to generate lung masks for each CT scan in the dataset.

#### 2. Crop Lung Regions

Run the following command to crop the lung regions using the generated masks:

```bash
python3 save_crop.py 
  --original_folder /path/to/original/dataset/ 
  --mask_folder /path/to/lung/masks/ 
  --output_folder /path/to/save/lung/crops/
```

#### 3. Extract Feature Embeddings

Extract and store feature embeddings by executing:

```bash
python3 store_embedding.py --file_paths /path/to/lung/crops/ 
                           --save_folder /path/to/save/feature/embeddings/
```

### B. Whole Slide Images (WSIs)

#### 1. Spatial Embedding Extraction

Utilize the **[CLAM](https://github.com/mahmoodlab/CLAM)** tool pipeline to process WSIs. Remove the final `avgpool` layer to preserve spatial embeddings.

#### 2. Feature Extraction with UNI

- Use the **UNI** pretrained model (requires prior access approval).
- Register a hook before the final layer to extract all token information.
- Skip the CLS token and reshape the remaining patch tokens into continuous spatial embeddings.

## 3. Training

To train the model, run the following commands with your custom configuration files:

- For WSI training:

  ```bash
  python3 train_pathology.py --config /path/to/your/config/file/
  ```

- For CT training:

  ```bash
  python3 train_medical.py --config /path/to/your/config/file/
  ```

Example training configuration files are provided in:

- `config/train_WSI.yaml`
- `config/train_CT.yaml`

## 4. Evaluation

Evaluate the model using:

```bash
python3 evaluation.py --config /path/to/your/config/file/
```

An example evaluation configuration file is available at:

- `config/evaluation.yaml`

The evaluation step supports both **AUC** (for classification) and **Dice scores** (for segmentation). Ensure your configuration file specifies:

- The evaluation type (classification or segmentation)
- Required thresholds

Results will be saved in the specified output directory.

## 5. Additional Instructions

- **Environment Setup**: Use the provided `environment.yml` file to set up the environment. Run:
  ```bash
  conda env create -f environment.yml
  ```
  Then activate the environment:
    ```bash
    conda activate <environment_name>
    ```
- **Access Approvals**: Obtain prior access approval for:
  - **UNI** pretrained model
  - **BRACS** dataset

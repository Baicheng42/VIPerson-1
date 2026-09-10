# README
## Overview
This document describes the data preparation and model execution workflow for identity generator.

## Data Preparation
Prepare two types of resources as follows:
- Image dataset with fixed unified poses
- Corresponding descriptions matching the image dataset

    - The descriptions are stored in a JSON file with the following format:
```json
[
  {
    "image": "1.jpg", 
    "attri": 
        {"basic": "man", 
        "updress": "blue polo shirt", 
        "downdress": "tan shorts", 
        "shoes": "brown sandals", 
        "hat": null, 
        "glasses": "no", 
        "bag": null}, 
    "oricap": "A man wearing blue polo shirt and tan shorts and brown sandals", 
    "caption": "A man wearing blue polo shirt and tan shorts and brown sandals"
  }, 
  ...
]
```



## Step 1: Extract Text Embeddings for Training Images
Fill in the script parameters in `./extract_text_embeddings_from_dir_clip.py`:
- `--data_dir`: File path of descriptions matched with the image dataset
- `--out_dir`: Target path to save output text embeddings

## Step 2: Train Identity Generator
1. Modify `./configs/train_config.yaml`:
    - Set `training.checkpoint.restore` to `false` for training from scratch
    - Set `training.checkpoint.restore` to `true` for resume training, and specify `training.checkpoint.path` for checkpoint weights
2. (Optional) Update `data_root` in `./configs/paths/gpu_cluster.yaml` to the root directory of your image dataset
3. Update dataset paths in `./configs/dataset/person_folder.yaml`:
    - `dataset.samples_root`: Path of training image dataset
    - `dataset.embedding_root`: Path of text embeddings generated in Step 1
4. Launch training script:
```bash
python personmain.py
```

## Step 3: Generate New Identities
1. Edit configurations in `./ID_generator/configs/person_config.yaml`:
    -  `gener_path`: Storage path for images of newly generated identities
    - `gener_context_path`: Storage path for corresponding random vectors
    - `training.checkpoint.path`: Model weight checkpoint trained in Step 2

2. Run the identity generation script:
```bash
python persongener_uncondi_new.py
```

## Step 4: Generate Hard Identities
1. Generate descriptions and matching text embeddings for each new identity obtained in Step 3
2. Configure parameters in `./ID_generator/configs/person_config.yaml`:
    - `gener_path_text_emb`: Path of text embeddings of generated new identity images
    - `gener_hard_path`: Output storage path for hard identity results
3. Execute the generation script:
```bash
python persongener_part_dropout.py
```
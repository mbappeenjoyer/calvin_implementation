# CALVIN Implementation on Google Colab

This guide walks you through running and training CALVIN (Composing Actions from Language and Vision), an open-source simulated benchmark to learn long-horizon language-conditioned, directly on Google Colab. Follow these steps carefully to set up the environment, train the MCIL policy, and evaluate your trained agent.

---

## Step 1: Clone the Notebook and Connect GPU

1. Open your Google Drive and upload the file  
   `CALVIN_code_colab.ipynb`  

2. Open the notebook in Google Colab.

3. Go to the menu bar →  
   **Runtime > Change runtime type > Hardware accelerator > GPU**
---

## Step 2: Clone the Implementation Repository

Run the following command in a new code cell:

```bash
!git clone --recurse-submodules https://github.com/mbappeenjoyer/calvin_implementation.git
```
--- 

## Step 3: Environment Setup

Inside the notebook, locate the **Setup** section and run all its cells sequentially in order to install dependencies.
> [!TIP]
> Restarting the session after installing some dependencies caused the session to crash entirely, so one should avoid it.

---

## Step 4: Download the Dataset (Debug Split)

Navigate to the dataset directory and download the debug split:

```bash
%cd /content/calvin/dataset
!sh download_data.sh debug
```
We use the debug split owing to compute constraints, on the free tier of Google Colab.

---

## Step 5: Training the MCIL Policy

Before training, move to the agent directory:

```bash
%cd /content/calvin_implementation/calvin_models/calvin_agent
```

Then run the training script:

```bash
!python training.py \
  datamodule.root_data_dir=/content/calvin_implementation/dataset/calvin_debug_dataset \
  ~callbacks.rollout \
  ~callbacks.rollout_lh \
  datamodule/datasets=vision_lang \
  datamodule.datasets.vision_dataset.num_workers=1 \
  datamodule.datasets.lang_dataset.num_workers=1 \
  datamodule.datasets.vision_dataset.batch_size=8 \
  datamodule.datasets.lang_dataset.batch_size=8 
```

> [!WARNING]  
> Adjust `num_workers` and `batch_size` based on your Colab GPU memory.  

> [!TIP]  
> Training logs and checkpoints are automatically saved under:

```
/content/calvin_implementation/calvin_models/runs/<timestamp>/
```

---

## Step 6: Inference / Evaluation

After training, you can test your policy using:

```bash
!python evaluation/evaluate_policy.py   --dataset_path=/content/calvin_implementation/dataset/calvin_debug_dataset   --train_folder=/content/calvin_implementation/calvin_models/runs/<your-run-folder-here>
```

Replace the path after `--train_folder` with your actual training run directory name (the latest timestamped folder inside `runs/`).

---

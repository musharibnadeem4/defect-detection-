"""Dataset loading and group-aware train/val/test splitting (Step 2).

Reads data/raw/<class_name>/*.jpg|png using class names from configs/config.yaml and
the group ids in data/processed/manifest.csv so duplicates never straddle splits.
"""

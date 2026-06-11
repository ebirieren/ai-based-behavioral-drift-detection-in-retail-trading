# Current Version Structure

The repository is divided into two main folders:

- `current_version/`: active project files used for the current thesis/demo version.
- `old_versions/`: old datasets, older notebooks, backups, and previous outputs/models.

This file explains the active `current_version/` folder.

## Main Python Files

- `userInterface.py`: Streamlit dashboard application.
- `behavioral_drift_pipeline.py`: Main batch pipeline runner.
- `pipeline_config.py`: Shared file paths and constant lists.
- `data_preprocessing.py`: Dataset cleaning and proxy creation.
- `text_label_rules.py`: Rule-based journal label support.
- `model_training.py`: Isolation Forest feature selection, training, and scoring.
- `recommendations.py`: Alert levels, behavior patterns, and recommendation text.
- `validation_outputs.py`: Thesis validation tables and Excel export.

## Folders

- `data/`: raw and uploaded/supporting datasets, plus the local UI database.
- `outputs/`: generated Excel result files.
- `models/`: saved model files.
- `notebooks/`: current development notebook.
- `docs/`: methodology and pipeline documentation.

## Default Run Commands

Run the batch pipeline:

```bash
python3 behavioral_drift_pipeline.py
```

Run the dashboard:

```bash
streamlit run userInterface.py
```

The default raw dataset is:

```text
data/behavioraldriftdetectiondataset.xlsx
```

The default result workbook is:

```text
outputs/behavioral_drift_results.xlsx
```

# contextual-redundancy

Built off of https://github.com/lu-wo/quantifying-redundancy

To generate conditional entropy values for each feature run:
- **Absolute Prominence**: `python src/train.py experiment=emnlp/finetuning/prominence_regression_absolute_bert_large_mle_cm.yaml`
- **Relative Prominence**: `python src/train.py experiment=emnlp/finetuning/prominence_regression_relative_bert_large_cm.yaml`
- **Energy/Loudness**: `python src/train.py experiment=emnlp/finetuning/energy_regression_bert_cm.yaml`
- **Pitch/F0**: `python src/train.py experiment=emnlp/finetuning/duration_regression_syll_bert_cm.yaml`
- **Pause**: `python src/train.py experiment=emnlp/finetuning/pause_regression_after_roberta_cm.yaml`
- **Duration**: `python src/train.py experiment=emnlp/finetuning/duration_regression_syll_bert

Then use the notebook to produce the plots.
- `name` is the relevant folder in the folder `losses_cm`
- `feature` is the name the of the feature (one of the keys in the `featplotdict` dictionary)

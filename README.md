# contextual-redundancy

Built off of https://github.com/lu-wo/quantifying-redundancy

To generate conditional entropy values for each feature run:
- Absolute Prominence: python src/train.py experiment=emnlp/finetuning/prominence_regression_absolute_bert_large_mle_cm.yaml
- Relative Prominence: python src/train.py experiment=emnlp/finetuning/prominence_regression_relative_bert_large_cm.yaml
- Energy: `python src/train.py experiment=emnlp/finetuning/energy_regression_bert_cm.yaml`

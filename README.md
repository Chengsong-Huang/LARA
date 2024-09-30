# LARA
codes for Divide, Reweight, and Conquer: A Logit\\ Arithmetic Approach for In-Context Learning

Most codes comes from https://github.com/eth-sri/language-model-arithmetic

Main experiment can be run like 
```
python language-model-arithmetic/hyper_search_loss.py --model meta-llama/Meta-Llama-3.1-8B --output_path llama31 --dataset_name BBH --subdataset_name date_understanding --train_length 32
```

The BBH and MMLU data come from `Chain-of-Thought Hub`
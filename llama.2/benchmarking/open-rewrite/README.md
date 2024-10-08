# Llama 3.2 (1B) Open-rewrite Benchmark

This repository is an implementation of Open-rewrite benchmark on [llama 3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) (1B). This repository is intended to reproduce the reported benchmark scores under the [Instruction Tuned Models (Re-rewriting)](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md#instruction-tuned-models) which is around which is around `41.6 %` for `Llama 3.2 1B`

# Setup
## Dataset, Models and Metrics
1. We used a downloaded version of the [Open-rewrite Evaluation Datasets](https://github.com/google-research/google-research/blob/master/rewritelm/openrewriteeval_aaai24.zip) from google-research's **rewritelm** existing dataset.
2. A checkpoint version of `Llama 3.2 1B` was obtained and used for evaluation
3. To calculate the effectiveness of the model on Re-writing, [Rouge-L score](https://pypi.org/project/rouge-score/) was calculated for each of the predicted response against the dataset reference and the micro-average was computed.

## Environment
1. An environment with 1 GPU is used for evaluation (as llama-models api by default use GPU)
2. Libraries includes: `general: pandas, numpy, torch, etc`, `configuration: hydra-core`, `Model specific: llama-models`
    ```bash
    pip install -r requirements.txt
    ```

## Experiment Configuration
Experiment configuration can be found in `llama.2/benchmarking/open-rewrite/config/main.yaml` with sample configuration below.
```yaml
root_dir:  some/root/directory
experiment_base_dir: ${root_dir}/experiments/llama.2/benchmarking/open-rewrite
llama3_open_rewrite_benchmark:
  _target_: modules.benchmark_modules.LlamaOpenRewriteBenchmark
  data_path: ${..experiment_base_dir}/datasets/openrewriteeval_aaai24.csv
  model_ckpt_dir: /data/llm/llama/Llama-3.2-1B-Instruct/original
  tokenizer_path: ${.model_ckpt_dir}/tokenizer.model
  model_args:
    build:
      max_seq_len: 512
      max_batch_size: 1
      model_parallel_size: 1
    generation:     
      temperature: 0.6
      top_p: 0.9
      max_gen_len: 512

# configuration for benchmarks for ease of use use when testing
predict:
  mode: full #full
  prompt_engineered: true
  n_rows: 1
  frac: 0.1
```

## Scripts
1. Default benchmark
    ```bash
    python benchmark.py
    ``` 
2. Benchmark w/o prompt engineering
    ```
    python benchmark.py predict.prompt_engineered=false
    ```
> Please see configuration above and `llama.2/benchmarking/open-rewrite/modules/benchmark_modules.py` code if you wish another configurations. 


# Results
| Setup       | Rouge-L (micro-avg) |
|-----------------|-------|
| Meta (Reported)     | .416   |
| Ours (with prompt engineering similar to meta)           | 0.401   |
| Ours (with no prompt engineering)  | 0.319   |
# Quantifying Exposure Bias
Accompanying repository for the paper: Why Exposure Bias Matters: An Imitation Learning Perspective of Error Accumulation in Language Generation

### Installation Instruction
```bash
 python -m venv ${HOME}/envs/exp_bias
 source ${HOME}/envs/exp_bias/bin/activate

 pip install -r requirements.txt
```

### Command to train the oracle model.
```python
 python run_clm.py     \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
     --dataset_config_name wikitext-103-raw-v1 \
     --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --block_size 512 \
    --output_dir <oracle dir>
 ```

### Command to evaluate the exposue bias of GPT-2 model
```python
python decoding_experiments.py \
   --oracle-model <path to oracle model> \
   --eval-model <path or name of the eval model.> \
   --context-dataset wikitext-103 \
   --context-len 50 \
   --top-ks 10,50,100,5,500 \
   --top-ps 0.9,0.8,0.6,0.5 \
   --sampling-temperatures 0.5 1,1.2,1.5,2
   --beams 2,5,10,30
```

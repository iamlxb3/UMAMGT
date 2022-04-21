# Unraveling the Mystery of Artifacts in Machine Generated Texts
This paper has been accpected to LREC'22. Paper link: 

### Video Presentation
Presenter: JiaShu Pu

### Install
```
conda create --name myenv python==3.6.8
source activate myenv
pip install -r requirements.txt
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```
install Torch with cuda 11.4:
```pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html```


### Prerequisites & Datasets
- We have open-sourced our datasets, which can be downloaded from https://drive.google.com/file/d/1xA9TtDYJE9BEwecL8QJ5d0LTytn5hhBr.
- Before running the experiment, you have to download the dataset and unzip all data to ```data/``` folder
- Label-1: The excerpt is written by Human. Label-0: The excerpt is written by Machine.

### Run experiments:

---
#### Run Corruption/Replacement Operation Experiments
run ```app/train_on_en.sh``` for English Dataset and ```app/train_on_cn.sh``` for Chinese Dataset.

Example: ```bash train_on_en.sh en_grover '0' 1 1 0 1 1 'char_deduplicate'```

Arguments for ```train_on_en.sh``` and ```train_on_cn.sh```:
- dataset_name: Valid choices: ('en_grover', 'cn_novel_5billion', 'en_writing_prompt')
- char_freq_ranges: Use top-k characters from the Vocabulary
- is_change_apply_to_test: Whether apply Corruption/Replacement Operation to test set 
- is_debug: if the training is in debug mode
- re_init_weights: Whether to use the pre-trained LM or initialize LM with random parameters
- repeat: The number of times experiment repeat for each Corruption/Replacement Operation setting.
- is_change_apply_to_train: Whether apply Corruption/Replacement Operation to training set
- semantic_change: Choices of Corruption/Replacement Operations

Valid choices of applying Corruption/Replacement Operation:
- reorder_shuffle: shuffle all tokens in an excerpt
- reorder_freq_high2low: sort tokens according to frequency (descending order)
- reorder_freq_low2high: sort tokens according to frequency (ascending order)
- char_deduplicate: remove duplicate tokens in an excerpt
- None: keep the original text
- likelihood_rank: replace text with Text Generation Models' Likelihood ranks
- pos: replace text with part-of-speech
- dep: replace text with dependency parsing tree
- constit: replace text with constituent parsing tree
- ner: replace text with Named Entities
- use stopword: keep only stop words
- non_use_stopword: keep only context words
---
#### Run Experiments of constraining tokens in a certain region
run ```app/restrict_char_freq_region.sh```

Example: ```bash restrict_char_freq_region.sh '10 20' 1 15 'rm_chars_in_freq'  'en_writing_prompt' 'en_roberta'```

Arguments for ```restrict_char_freq_region.sh```:
- char_freq_ranges: The dividing values for constraining tokens of certain frequencies, eg. (256, 512)
- is_debug: if the training is in debug mode
- repeat: The number of times experiment repeat for each frequency region
- semantic_change: Valid choice: ('rm_chars_in_freq', 'rm_chars_out_freq')
- dataset_name: Valid choices: ('en_grover', 'cn_novel_5billion', 'en_writing_prompt')
- classifier_name: If running for English Dataset, specify 'cn_roberta', if running for Chinese Dataset, specify 'cn_roberta'

---
#### Do Explainable Analysis (Integrated Gradients)
run ```app/run_story_interpret.sh```

Example: ```bash run_story_interpret.sh 50 1 256 en_grover interpret_en_grover_en_roberta_debug_1 roberta 50 1```

Arguments for ```run_story_interpret.sh```:
- debug_N: Number of Training sample, used for quick code tuning
- batch_size: Batch size for parameters update (Choose the one that fits your GPU memory)
- max_text_length: The maximum text length
- data_name: Valid choices: ('en_grover', 'cn_novel_5billion', 'en_writing_prompt')
- model_name: To interpret which model. Model should be trained with parameters saved in model_ckpts/$model_name. In this folder, 'config.json' and 'pytorch_model.bin' should be included.
- model_type: The model name, it could be roberta, bert, etc. It should corresponds to the model being trained.
- ig_n_steps: The number of steps used by the approximation method for Integrated Gradients.
- use_pad_baseline: Bool value. 1: use all [PAD] baseline for Integrated Gradients. 0: Use all zero baseline.

### License:
Our code is under Apache License 2.0.

### Cite our paper
If you found our paper/code useful, please cite:


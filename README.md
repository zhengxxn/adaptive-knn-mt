# Adaptive kNN-MT

Code for our ACL 2021 paper "Adaptive Nearest Neighbor Machine Translation". 
Please cite our paper if you find this repository helpful in your research:

```
@inproceedings{zheng-etal-2021-adaptive,
    title = "Adaptive Nearest Neighbor Machine Translation",
    author = "Zheng, Xin  and Zhang, Zhirui  and Guo, Junliang  and Huang, Shujian  and Chen, Boxing  and Luo, Weihua  and Chen, Jiajun",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.47",
    doi = "10.18653/v1/2021.acl-short.47",
    pages = "368--374",
    abstract = "kNN-MT, recently proposed by Khandelwal et al. (2020a), successfully combines pre-trained neural machine translation (NMT) model with token-level k-nearest-neighbor (kNN) retrieval to improve the translation accuracy. However, the traditional kNN algorithm used in kNN-MT simply retrieves a same number of nearest neighbors for each target token, which may cause prediction errors when the retrieved neighbors include noises. In this paper, we propose Adaptive kNN-MT to dynamically determine the number of k for each target token. We achieve this by introducing a light-weight Meta-k Network, which can be efficiently trained with only a few training samples. On four benchmark machine translation datasets, we demonstrate that the proposed method is able to effectively filter out the noises in retrieval results and significantly outperforms the vanilla kNN-MT model. Even more noteworthy is that the Meta-k Network learned on one domain could be directly applied to other domains and obtain consistent improvements, illustrating the generality of our method. Our implementation is open-sourced at https://github.com/zhengxxn/adaptive-knn-mt.",
}
```

This project implements our Adaptive kNN-MT as well as Vanilla kNN-MT.
The implementation is build upon [fairseq](https://github.com/pytorch/fairseq), and heavily inspired by [knn-lm](https://github.com/urvashik/knnlm), many thanks to the authors for making their code avaliable.

The [NJUNLP/knn-box](https://github.com/NJUNLP/knn-box) toolkit also implements Adaptive kNN-MT and other kNN models. We highly recommend to use NJUNLP/knn-box instead of this repo in the future, it can be more clear and easy to use, and can do some visualization.

## Requirements and Installation

* pytorch version >= 1.5.0
* python version >= 3.6
* faiss-gpu >= 1.6.5
* pytorch_scatter = 2.0.5
* 1.19.0 <= numpy < 1.20.0

You can install this project by
```
pip install --editable ./
```

## Attention！
**2022/12/26** In our earlier implementations (before [commit 7997d11](https://github.com/zhengxxn/adaptive-knn-mt/commit/7997d11907eb03b37e22cf9e6342f0085f9eef06)), kNN-MT / adaptive kNN-MT were much slower than NMT for implementation-specific reasons. We repaired this problem and now kNN-MT / adaptive kNN-MT are only a little slower than NMT. To get high inference speed, if your code version is older than [commit 7997d11](https://github.com/zhengxxn/adaptive-knn-mt/commit/7997d11907eb03b37e22cf9e6342f0085f9eef06), please pull the latest code, or refer to [Pull Request #9](https://github.com/zhengxxn/adaptive-knn-mt/pull/9) to modify your code.

## Instructions

We use an example to show how to use our codes.

### Pre-trained Model and Data

The pre-trained translation model can be downloaded from [this site](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md).
We use the De->En Single Model for all experiments.

The raw data can be downloaded in [this site](https://github.com/roeeaharoni/unsupervised-domain-clusters), and 
you should preprocess them with moses toolkits and the bpe-codes provided by pre-trained model. 
For convenience, We also provide pre-processed [data](https://drive.google.com/file/d/18TXCWzoKuxWKHAaCRgddd6Ub64klrVhV/view?usp=sharing).

### Create Datastore

This script will create datastore (includes key.npy and val.npy) for the data.

```
DSTORE_SIZE=3613350
MODEL_PATH=/path/to/pretrained_model_path
DATA_PATH=/path/to/fairseq_preprocessed_data_path
DATASTORE_PATH=/path/to/save_datastore
PROJECT_PATH=/path/to/ada_knnmt

mkdir -p $DATASTORE_PATH

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/save_datastore.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation \
    --valid-subset train \
    --path $MODEL_PATH \
    --max-tokens 4096 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 1024 --dstore-fp16 --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH
 
# 4096 and 1024 depend on your device and model separately
```

The DSTORE_SIZE depends on the num of tokens of target language train data. You can get it by two ways:

- find it in preprocess.log file, which is created by fairseq-process and in data binary folder.
- calculate wc -l + wc -w of raw data file.

The datastore sizes we used in our paper are listed as below:

| IT      | Medical | koran  | Law      |
|---------|---------|--------|----------|
| 3613350 | 6903320 | 524400 | 19070000 |

### Build Faiss Index

This script will build faiss index for keys, which is used for fast knn search. when the knn_index is build, you can
remove keys.npy to save the hard disk space.

```
PROJECT_PATH=/path/to/ada_knnmt
DSTORE_PATH=/path/to/saved_datastore
DSTORE_SIZE=3613350

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/train_datastore_gpu.py \
  --dstore_mmap $DSTORE_PATH \
  --dstore_size $DSTORE_SIZE \
  --dstore_fp16 \
  --faiss_index ${DSTORE_PATH}/knn_index \
  --ncentroids 4096 \
  --probe 32 \
  --dimension 1024
```

### Train Adaptive kNN-MT Model
```

DSTORE_SIZE=3613350
DATA_PATH=/path/to/fairseq_preprocessed_data_path
PROJECT_PATH=/path/to/ada_knnmt
MODEL_PATH=/path/to/pretrained_model_path
DATASTORE_PATH=/path/to/saved_datastore

max_k_grid=(4 8 16 32)
batch_size_grid=(32 32 32 32)
update_freq_grid=(1 1 1 1)
valid_batch_size_grid=(32 32 32 32)

for idx in ${!max_k_grid[*]}
do

  MODEL_RECORD_PATH=/path/to/save/model/train-hid32-maxk${max_k_grid[$idx]}
  TRAINING_RECORD_PATH=/path/to/save/tensorboard/train-hid32-maxk${max_k_grid[$idx]}
  mkdir -p "$TRAINING_RECORD_PATH"

  CUDA_VISIBLE_DEVICES=0 python \
  $PROJECT_PATH/fairseq_cli/train.py \
  $DATA_PATH \
  --log-interval 100 --log-format simple \
  --arch transformer_wmt19_de_en_with_datastore \
  --tensorboard-logdir "$TRAINING_RECORD_PATH" \
  --save-dir "$MODEL_RECORD_PATH" --restore-file "$MODEL_PATH" \
  --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
  --validate-interval-updates 100 --save-interval-updates 100 --keep-interval-updates 1 --max-update 5000 --validate-after-updates 1000 \
  --save-interval 10000 --validate-interval 100 \
  --keep-best-checkpoints 1 --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
  --train-subset valid --valid-subset valid --source-lang de --target-lang en \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.001 \
  --max-source-positions 1024 --max-target-positions 1024 \
  --batch-size "${batch_size_grid[$idx]}" --update-freq "${update_freq_grid[$idx]}" --batch-size-valid "${valid_batch_size_grid[$idx]}" \
  --task translation \
  --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 --min-lr 3e-05 --lr 0.0003 --clip-norm 1.0 \
  --lr-scheduler reduce_lr_on_plateau --lr-patience 5 --lr-shrink 0.5 \
  --patience 30 --max-epoch 500 \
  --load-knn-datastore --dstore-filename $DATASTORE_PATH --use-knn-datastore \
  --dstore-fp16 --dstore-size $DSTORE_SIZE --probe 32 \
  --knn-sim-func do_not_recomp_l2 \
  --use-gpu-to-search --move-dstore-to-mem --no-load-keys \
  --knn-lambda-type trainable --knn-temperature-type fix --knn-temperature-value 10 --only-train-knn-parameter \
  --knn-k-type trainable --k-lambda-net-hid-size 32 --k-lambda-net-dropout-rate 0.0 --max-k "${max_k_grid[$idx]}" --k "${max_k_grid[$idx]}" \
  --label-count-as-feature
done
```

The batch size and update-freq should be adjust by yourself depends on your gpu.

### Inference with Adaptive kNN-MT

```
DSTORE_SIZE=3613350
MODEL_PATH=/path/to/trained_model

DATASTORE_PATH=/path/to/datastore
DATA_PATH=/path/to/data
PROJECT_PATH=/path/to/ada_knnmt

OUTPUT_PATH=/path/to/save_output_result

mkdir -p "$OUTPUT_PATH"

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/experimental_generate.py $DATA_PATH \
    --gen-subset test\
    --path "$MODEL_PATH" --arch transformer_wmt19_de_en_with_datastore \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu \
    --batch-size 32 \
    --tokenizer moses --remove-bpe \
    --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True,
    'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'probe': 32,
    'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
    'knn_temperature_type': 'fix', 'knn_temperature_value': 10,}" \
    | tee "$OUTPUT_PATH"/generate.txt

grep ^S "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/src
grep ^T "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/ref
grep ^H "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp
grep ^D "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp.detok
```

### base NMT inference

We also provide scripts to do NMT and vanilla kNN-MT inference

```
MODEL_PATH=/path/to/pretrained_model_path/
DATA_PATH=/path/to/fairseq_preprocessed_path/
DATASTORE_PATH=/path/to/saved_datastore/
PROJECT_PATH=/path/to/knnmt/

mkdir -p $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/fairseq_cli/generate.py $DATA_PATH\
    --gen-subset test \
    --path $MODEL_PATH \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu \
    --max-tokens 4096 \
    --tokenizer moses --remove-bpe | tee $OUTPUT_PATH/generate.txt

grep ^S "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/src
grep ^T "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/ref
grep ^H "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp
grep ^D "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp.detok
```

### Vanilla kNN-MT inference

```
DSTORE_SIZE=3613350
MODEL_PATH=/path/to/pre_trained_model

DATASTORE_PATH=/path/to/datastore
DATA_PATH=/path/to/data
PROJECT_PATH=/path/to/ada_knnmt

OUTPUT_PATH=/path/to/save_output_result

mkdir -p "$OUTPUT_PATH"

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/experimental_generate.py $DATA_PATH \
    --gen-subset test\
    --path $MODEL_PATH --arch transformer_wmt19_de_en_with_datastore \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu \
    --batch-size 32 \
    --tokenizer moses --remove-bpe \
    --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True,
    'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'k': 8, 'probe': 32,
    'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
    'knn_lambda_type': 'fix', 'knn_lambda_value': 0.7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10,
     }" \
    | tee "$OUTPUT_PATH"/generate.txt

grep ^S "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/src
grep ^T "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/ref
grep ^H "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp
grep ^D "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp.detok
```

We recommend you to use below hyper-parameters to replicate the good vanilla knn-mt results.
And note that for our adaptive-knn-mt, we set the temperature as same as below.

|             |  IT | Medical | Law | Koran |
|:-----------:|:---:|:-------:|:---:|:-----:|
|      k      |  8  |    4    |  4  |   16  |
|    lambda   | 0.7 |   0.8   | 0.8 |  0.8  |
| temperature |  10 |    10   |  10 |  100  |

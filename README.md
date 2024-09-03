# Northern Sámi wav2vec2 ASR
Scripts for adapting large speech foundation models for Northern Sámi ASR

## Pre-trained and fine-tuned models

The pre-trained and fine-tuned models are available at [Huggingface Hub](https://huggingface.co/collections/GetmanY1/sami-parliament-wav2vec2-asr-66699110493b618b9ee2bf21).

More details on the models are available in the [paper](TODO).

## Pre-training the models

The scripts shared in this repository are adapted to the AMD hardware of the [LUMI supercomputer](https://www.lumi-supercomputer.eu/). To train a wav2vec 2.0 Base model with continued pre-training, run

```
sbatch /scripts/pretraining/fairseq_train_multinode_w2v2_B_512gpus.sh
```

Note: you can simulate 512 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 512/k

## Fine-tuning the models with CTC using 🤗Transformers

To fine-tune a wav2vec 2.0 Base model using Huggingface Transformers, run

```
sbatch scripts/finetuning/huggingface_finetune_multinode_w2v2_B_8gpus_full.sh
```

For extended fine-tuning (see Section 3.3 in the [paper](TODO)), set `EXTENDED_FINETUNING` to True in [`scripts/finetuning/huggingface_run_speech_recognition_ctc_multigpu.py`](https://github.com/aalto-speech/northern-sami-asr/blob/38580c7ec9337da5cee8ea7aa31d085b7e63363b/scripts/finetuning/huggingface_run_speech_recognition_ctc_multigpu.py#L620) (line 620)

## Citation

If you use our models or scripts, please cite our article as:

```bibtex
@inproceedings{getman24b_interspeech,
  title     = {Exploring adaptation techniques of large speech foundation models for low-resource ASR: a case study on Northern Sámi},
  author    = {Yaroslav Getman and Tamas Grosz and Katri Hiovain-Asikainen and Mikko Kurimo},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {2539--2543},
  doi       = {10.21437/Interspeech.2024-479},
}
```

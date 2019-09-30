# OCR

## Step by step

### Training

1. Prepare training datasets

` python tools/create_crnn_ctc_tfrecord.py --image_dir YOUR/TRAINING/IMAGE --anno_file YOUR/LABEL/DIR --data_dir TFRECORD/SAVE/DIR`

2. Start training 

train_densenetocr_ctc_multigpu.py support multi-GPU training

### Inference

`python tools/inference_densenet_ctc.py --image_dir YOUR/TEST/IMAGE/DIR --model_dir ckpt/DIR`

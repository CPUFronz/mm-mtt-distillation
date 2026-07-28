# Multimodal Dataset Distillation by Matching Training Trajectories

Forked from https://github.com/GeorgeCazenavette/mtt-distillation

## Create Conda Environmen
```
conda env create -f requirements.yml
```

## Launch Example for ActionSense (Multimodal)

Put ActionSense data into `data/` folder.

#### 1. Train Buffers
```
python buffer.py \
    --dataset=ActionSense \
    --model=MMSConvB \
    --num_experts=100 \
    --data_path=data/ \
    --lr=0.05 \
    --train_epochs=100 \
    --batch_size=128 \
    --buffer_path=buffers/mm \
    --name=ActionSense_MM \
    --unimodal=''
```

#### 2. Distill Dataset
```
python distill.py \
    --data_path=data/ \
    --lr_img=1000 \
    --lr_sens=0.01 \
    --lr=0.01 \
    --lr_lr=1e-05 \
    --dataset=ActionSense \
    --model=MMSConvB \
    --expert_epochs=2 \
    --syn_steps=30 \
    --max_start_epoch=40 \
    --buffer_path=buffers/mm/ \
    --spc=10 \
    --name=ActionSense_SPC10_MM \
    --unimodal=''
```
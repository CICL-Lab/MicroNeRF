set -e

directory=dir/to/data
for file in "$directory"/*
do
  DATA_NAME=$(basename "$file")
  DATA_NAME="${DATA_NAME%%.*}"
  python -u MicroNeRF/main.py \
  --save_dir Tem \
  --is_train \
  --img_file data/gt/ratios/${DATA_NAME}.nii.gz \
  --step 20000 \
  --model_file Tem/${DATA_NAME}_stepBest.pth

  python -u MicroNeRF/main.py \
  --save_dir Tem/results \
  --img_file data/gt/ratios/${DATA_NAME}.nii.gz \
  --step 20000 \
  --model_file Tem/${DATA_NAME}_stepBest.pth
done

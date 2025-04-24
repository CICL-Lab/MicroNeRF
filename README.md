## MicroNeRF
This is the official implementation of "Reconstruct dense live-cell microscopy images via learning continuous fluorescence field"
(ISBI, 2025), which reconstructs dense 3D fluorescent stack from sparse slices.

### Environment
To create the environment, run
```bash
conda env create -f environment.yml
conda create MicroNeRF
```

### Train and test
MicroNeRF consists of two stages for constructing dense volume: 1) It learns the distribution of fluorescent signal from 
sparse slices; 2) After the training process is finished, the trained model can be used to generate dense stack according
to pixel's location.

To test MicroNeRF, pls excute the code as follows:
```bash
  # Training stage
  FILE_NAME=Data/test.nii.gz  # Replace this with your own data
  
  BASE_NAME=$(basename "$file")
  BASE_NAME="${DATA_NAME%%.*}"
  python -u main.py \
  --save_dir Expertments \
  --is_train \
  --img_file ${FILE_NAME} \
  --step 20000 \
  --model_file Expertments/${BASE_NAME}_stepBest.pth
  
  # Test stage
  python -u main.py \
  --save_dir Expertments \
  --img_file ${FILE_NAME} \
  --step 20000 \
  --model_file Tem/${BASE_NAME}_stepBest.pth
```
 Remember to save your own data according to the format of `Data/test.nii.gz`, including the image header.


### Acknowledgement
Part of the code is referred to [nerf](https://github.com/bmild/nerf).

### Citation
TODO

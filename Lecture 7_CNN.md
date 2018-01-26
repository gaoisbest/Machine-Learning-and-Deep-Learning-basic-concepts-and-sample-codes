# Basic concepts
## CNNs architecture
- CONV
  - filter size F: 3 by 3, 5 by 5
  - number of filters K: **power of 2**, e.g., 32, 64, 128
  - stride S: 1, 2
  - zero padding P: 0, 1, 2
  
- POOLING
  - filter size F: 2 by 2
  - stride S: 2
  - **DOES not change the number of depth**
  - **Not common to use zero-padding**
- FC
  
## CNNs properties
- Local connectivity: the depth of filter is equals to the depth of the input.
- Parameter sharing.

# Example
Please refer to [tf_5_CNN_text_classification.ipynb](https://github.com/gaoisbest/Tensorflow_notes_and_projects/blob/master/README.md) for more details about CNNs in text classification.

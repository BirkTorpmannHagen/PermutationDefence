# PermutationDefence
Defence against neural network stegomalware.

For a proof of concept, run `bash python main.py`. This tests the models for functional equality - i.e. that they have the same outputs (short of floating point errors) and tests if the payload can be extracted.
Example output: 
```
Outputs are the same!
Hiding secret message: "Hello, World! This could be malware! But it is not :)" in model
Layer Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) is processed, last index modified: 105
Layer Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) is processed, bits are extracted
Hello, World! This could be malware! But it is not :)
Layer Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) is processed, bits are extracted
ïMD^o*ReIü¬¨µt{@¶YÂW~ã~oxV£mFñÿ-V£¨JÈ$B_d

```

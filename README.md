# PermutationDefence
Defence against neural network stegomalware based on random permutation of the network weights. We utilize MaleficNet as our baseline, the implementation of which was forked from [here](https://github.com/pagiux/maleficnet/tree/master). Some modifications have been made to improve execution time and simplify the code.

For a proof of concept, run `python concept.py`. This tests the models for functional equality - i.e. that they have the same outputs (short of floating point errors) and tests if the payload can be extracted.
Example output: 
```
Injecting: 100%|██████████| 1736/1736 [00:02<00:00, 748.99it/s]
Extracting: 100%|██████████| 1736/1736 [00:01<00:00, 1131.77it/s]
Hello, World! This could be malware! But it is not :)
Extracting: 100%|██████████| 1736/1736 [00:01<00:00, 1091.60it/s]
xg­&;,T¶6+IEã2i´ô!;QÉbádd­^ðØð^Ô£}Yï»Ð

Vanilla Payload Recovered: True
Permuted model Payload Recovered: False

```
To reproduce our results, run `python experiments.py`. Note that this may take a while, since it requires training several ResNets and encoding/decoding large ldpc data. The experimental results are available in `data/`.

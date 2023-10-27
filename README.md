# PermutationDefence
Defence against neural network stegomalware.

For a proof of concept, run `bash python main.py`. This tests the models for functional equality - i.e. that they have the same outputs (short of floating point errors) and tests if the payload can be extracted.
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

The maleficnet implementation was forked from (here)[https://github.com/pagiux/maleficnet/tree/master]. Note the experiments defined in `maleficnet.py` do not work and are undergoing refactoring.

import os

import numpy as np

import torch.nn.utils.prune as prune
import pytorch_lightning as pl
from modelmodules import Classifier
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import loggers as pl_loggers
import torch.nn as nn
from channelpermutation import StegoSafeModel
from copy import deepcopy
import torch
import datasets
import pandas as pd
from itertools import product

from  channelpermutation import StegoSafeModel
from copy import deepcopy
from maleficnet.injector import Injector
from maleficnet.extractor import Extractor
import logging

import torch

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

def compare_model_weights(model1, model2):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    if state_dict1.keys() != state_dict2.keys():
        print(state_dict1.keys())
        print()
        print(state_dict2.keys())
        print("Models have different architecture")
        return False

    for key in state_dict1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"Layer {key} has different weights")
            return False

    print("Models have the same weights")
    return True

class MalwareTestBed:
    def __init__(self, data_module, model, model_name, dataset_name, max_epochs=200):
        self.model=model
        print("Training initial model")
        try:
            os.mkdir(f"checkpoints/{model_name}")
        except:
            pass
        callbacks = [pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max', dirpath=f"checkpoints/{model_name}", filename="best")]
        self.trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, callbacks=callbacks, logger=False, enable_progress_bar=True)
        self.data_module = data_module
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
        self.model.train()
        self.model_name = model_name
        print(self.model)
        try:
            print("loading")
            self.reset_model()
        except:
            self.module = Classifier(model, num_classes=data_module.num_classes)
            self.trainer.fit(model=self.module, datamodule=data_module)
            self.module = Classifier.load_from_checkpoint(f"checkpoints/{model_name}/best.ckpt", model=model,
                                                          num_classes=10)
        self.vanilla_performance = self.eval_model(self.module.model)
        print("Performance before payload", self.vanilla_performance)
        self.dataset_name = dataset_name
        self.model_name = model_name

    def eval_model(self, model):
        model = model.cuda()
        module = Classifier(model, num_classes=10)
        validation_results = self.trainer.validate(model=module, datamodule=self.data_module, verbose=False)
        val_acc_values = [result['val_acc'] for result in validation_results if 'val_acc' in result]
        overall_val_acc = sum(val_acc_values) / len(val_acc_values) if val_acc_values else None
        return overall_val_acc

    def retrain_model(self,size, num_epochs=2):
        retrained_module = Classifier.load_from_checkpoint(f"checkpoints/{self.dataset_name}_{self.model_name}_infected_{size}.ckpt", model=self.model, num_classes=self.data_module.num_classes)
        self.trainer = pl.Trainer(accelerator="gpu", max_epochs=num_epochs, enable_checkpointing=False, logger=False, enable_progress_bar=False)
        self.trainer.fit(model=retrained_module,datamodule=self.data_module)
        return retrained_module.model


    def reset_model(self):
        self.module = Classifier.load_from_checkpoint(f"checkpoints/{self.model_name}/best.ckpt", model=self.model,
                                                      num_classes=10)

    def prune_model(self, model, amount=0.01):
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        #this method adds new modules and hooks to the model. For simplicity, we do not use this method.
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        for module, name in parameters_to_prune:
            prune.remove(module, name)


    def integrity(self,payload, extracted):
        return np.mean([extracted[i]==payload[i] for i in range(len(payload))])

    def collect_data(self, size=1000):
        self.reset_model()
        print(size)
        payload = "".join(list(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size)))
        print("Payload:", payload)
        start_accuracy = self.eval_model(self.module.model)
        injector = Injector(seed=42,
                            device="cuda",
                            malware_payload=payload,
                            chunk_factor=16)

        # Infect the system 🦠
        extractor = Extractor(seed=42,
                              device="cuda",
                              malware_length=len(injector.payload),
                              hash_length=len(injector.hash),
                              chunk_factor=16)
        new_model_sd, message_length, _, _ = injector.inject(self.module.model, 0.0009)
        self.module.model.load_state_dict(new_model_sd)
        # self.trainer = pl.Trainer(accelerator="gpu", max_epochs=5, enable_checkpointing=False, logger=False, enable_progress_bar=False)  #retrain to improve performance... needed, but not in the paper
        # self.trainer.fit(model=self.module,datamodule=self.data_module)
        self.trainer.save_checkpoint(f"checkpoints/{self.dataset_name}_{self.model_name}_infected_{size}.ckpt")
        extract = extractor.extract(self.module.model, message_length)
        print(extract)
        assert extract==payload, f"Extracted Payload != Embedded Payload!"
        infected_performance = self.eval_model(self.module.model)
        safe_model = StegoSafeModel(deepcopy(self.module.model).cuda()).cuda()
        safe_extracted = extractor.extract(safe_model, message_length)
        # print("Channel shuffling extracted:", safe_extracted)
        cs = self.integrity(payload, safe_extracted)
        cs_performance = self.eval_model(self.module.model)
        # print()
        print("Model performance before infection:", start_accuracy)
        print("Model performance after infection:", infected_performance)
        print("Model performance after infection and shuffling:", cs_performance)
        print("Integrity after channel shuffling:", cs)


        data = [
            {"Defense":"Channel Shuffling", "Integrity": cs, "Performance": cs_performance},
        ]


        pruning = []
        pruning_performance = []
        for amount in [0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
            print("Pruning:", amount)
            pruned_model = Classifier.load_from_checkpoint(
                f"checkpoints/{self.dataset_name}_{self.model_name}_infected_{size}.ckpt", model=deepcopy(self.model),
                num_classes=self.data_module.num_classes)
            self.prune_model(pruned_model.model, amount=amount)
            extracted = extractor.extract(pruned_model.model, message_length)
            pruning.append(self.integrity(payload, extracted))
            pruning_performance.append(self.eval_model(pruned_model.model))
            data.append({"Defense": f"Pruning_{amount}", "Integrity": pruning[-1], "Performance": pruning_performance[-1]})

        retraining = []
        retraining_performance = []
        for i in [1, 5, 25, 50]:
            retrained_model = self.retrain_model(num_epochs=i, size=size)
            retraining.append(self.integrity(payload, extractor.extract(retrained_model, message_length)))
            retraining_performance.append(self.eval_model(retrained_model))
            data.append({"Defense": f"Retraining_{i}", "Integrity": retraining[-1], "Performance": retraining_performance[-1]})
        data.append({"Defense": "None", "Integrity": 1, "Performance": self.vanilla_performance})
        print("Channel Shuffling: ", cs)
        print("Pruning:", pruning, "performance:", pruning_performance)
        print("Retraining:", retraining,"performance:", retraining_performance)
        pd.DataFrame(data).to_csv(f"{self.dataset_name}_{self.model_name}_{len(payload)}.csv")


if __name__ == '__main__':

    for model_size, model_constructor in list(zip([18, 34,50,101,152], [resnet18, resnet34, resnet50, resnet101, resnet152])):
        data_module = CIFAR10DataModule("../../Datasets/CIFAR10", batch_size=64, val_split=0.2)

        data_module.setup()
        model_instance = model_constructor()
        tb = MalwareTestBed(data_module, model_instance, model_name=f"resnet{model_size}", dataset_name="cifar10")
        for payload_size in [1000, 5000, 10000, 20000]:
            tb.collect_data(payload_size)

    # datasets.load_dataset("glue", "ex")



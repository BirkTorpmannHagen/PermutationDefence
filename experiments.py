import numpy as np
from torchvision.datasets import CIFAR10
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.models.deepspeech import DeepSpeech
from torchtext.datasets import AG_NEWS
import torch.nn.utils.prune as prune
import pytorch_lightning as pl
from modelmodules import Classifier
from torchvision.models import resnet18, VGG, densenet121
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import loggers as pl_loggers
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
    def __init__(self, data_module, model, model_name, dataset_name, max_epochs=1,size=10000):
        self.model=model
        self.model.eval()
        self.size = size
        # checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints", filename=f"{dataset_name}_{model_name}")

        self.trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, callbacks=[],enable_checkpointing=False, logger=False, enable_progress_bar=False)
        self.data_module = data_module
        self.module = Classifier.load_from_checkpoint(f"checkpoints/{dataset_name}_{model_name}.ckpt", model=model, num_classes=data_module.num_classes)
        # self.module = Classifier(model, num_classes=data_module.num_classes)
        # self.trainer.fit(model=self.module, datamodule=data_module)
        self.vanilla_performance = self.eval_model(self.module.model)
        print("Performance before payload", self.vanilla_performance)
        self.dataset_name = dataset_name
        self.model_name = model_name

    def eval_model(self, model):
        model = model.cuda()
        module = Classifier(model, num_classes=10)
        validation_results = self.trainer.validate(model=module, datamodule=self.data_module, verbose=True)
        val_acc_values = [result['val_acc'] for result in validation_results if 'val_acc' in result]
        overall_val_acc = sum(val_acc_values) / len(val_acc_values) if val_acc_values else None
        return overall_val_acc

    def retrain_model(self, num_epochs=2):
        retrained_module = Classifier.load_from_checkpoint(f"checkpoints/{self.dataset_name}_{self.model_name}_infected_{self.size}.ckpt", model=self.model, num_classes=self.data_module.num_classes)
        self.trainer = pl.Trainer(accelerator="gpu", max_epochs=num_epochs, enable_checkpointing=False, logger=False, enable_progress_bar=False)
        self.trainer.fit(model=retrained_module,datamodule=self.data_module)
        return retrained_module.model


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

    def collect_data(self):
        size=self.size
        payload = "".join(list(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size)))
        print("Payload:", payload)
        injector = Injector(seed=42,
                            device="cpu",
                            malware_payload=payload,
                            chunk_factor=6)

        # Infect the system 🦠
        extractor = Extractor(seed=42,
                              device="cpu",
                              malware_length=len(injector.payload),
                              hash_length=len(injector.hash),
                              chunk_factor=6)
        new_model_sd, message_length, _, _ = injector.inject(self.module.model, 0.0009)
        self.module.model.load_state_dict(new_model_sd)
        self.trainer.save_checkpoint(f"checkpoints/{self.dataset_name}_{self.model_name}_infected_{self.size}.ckpt")
        assert extractor.extract(self.module.model, message_length)==payload, f"Extracted Payload != Embedded Payload!"

        safe_model = StegoSafeModel(deepcopy(self.module.model).cuda()).cuda()
        safe_extracted = extractor.extract(safe_model, message_length)
        print("Channel shuffling extracted:", safe_extracted)
        cs = self.integrity(payload, safe_extracted)
        cs_performance = self.eval_model(self.module.model)
        print()
        print("Model performance after infection:", self.eval_model(safe_model))


        data = [
            {"Defense":"Channel Shuffling", "Integrity": cs, "Performance": cs_performance},
        ]


        pruning = []
        pruning_performance = []
        for amount in [0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
            pruned_model = Classifier.load_from_checkpoint(
                f"checkpoints/{self.dataset_name}_{self.model_name}_infected_{self.size}.ckpt", model=deepcopy(self.model),
                num_classes=self.data_module.num_classes)
            self.prune_model(pruned_model.model, amount=amount)
            extracted = extractor.extract(pruned_model.model, message_length)
            pruning.append(self.integrity(payload, extracted))
            pruning_performance.append(self.eval_model(pruned_model.model))
            data.append({"Defense": f"Pruning_{amount}", "Integrity": pruning[-1], "Performance": pruning_performance[-1]})

        retraining = []
        retraining_performance = []
        for i in [1, 5, 25, 50]:
            retrained_model = self.retrain_model(num_epochs=i)
            retraining.append(self.integrity(payload, extractor.extract(retrained_model, message_length)))
            retraining_performance.append(self.eval_model(retrained_model))
            data.append({"Defense": f"Retraining_{i}", "Integrity": retraining[-1], "Performance": retraining_performance[-1]})
        print("Channel Shuffling: ", cs)
        print("Pruning:", pruning, "performance:", pruning_performance)
        print("Retraining:", retraining,"performance:", retraining_performance)
        pd.DataFrame(data).to_csv(f"{self.dataset_name}_{self.model_name}_{len(payload)}.csv")


if __name__ == '__main__':

    for size in [5000, 1000, 500, 100, 10][::-1]:
        cifar10 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to("cuda")
        module = CIFAR10DataModule("../../Datasets/CIFAR10", batch_size=128, val_split=0.2)
        module.setup()
        tb = MalwareTestBed(module, cifar10, model_name="resnet32", dataset_name="cifar10", size=size)
        tb.collect_data()

    datasets.load_dataset("glue", "ex")



import numpy as np
from torchvision.datasets import CIFAR10
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.models.deepspeech import DeepSpeech
from torchtext.datasets import AG_NEWS
import torch.nn.utils.prune as prune
import pytorch_lightning as pl
from classifier import Classifier
from torchvision.models import resnet18, VGG, densenet121
from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule, ImagenetDataModule
from pytorch_lightning import loggers as pl_loggers
from channelpermutation import StegoSafeModel
from copy import deepcopy
import torch

from  channelpermutation import StegoSafeModel
from copy import deepcopy
from maleficnet.injector import Injector
from maleficnet.extractor import Extractor

import torch


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

class TestBed:
    def __init__(self, data_module, model, max_epochs=5):
        model.eval()
        logger = pl_loggers.TensorBoardLogger(save_dir=f"{data_module.__class__.__name__}_{model.__class__.__name__}",
                                              version=None)
        self.trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, logger=logger)
        self.data_module = data_module
        try:
            self.module = Classifier.load_from_checkpoint(f"lighting_logs/{data_module.__class__.__name__}_{model.__class__.__name__}.ckpt", model=model, num_classes=data_module.num_classes)
        except FileNotFoundError:
            self.module = Classifier(model, num_classes=data_module.num_classes)
            self.trainer.fit(datamodule=data_module, model=self.module)

    def eval_model(self, model):
        trainer = pl.Trainer(accelerator="gpu", max_epochs=1)
        module = Classifier(model, num_classes=10)
        trainer.validate(datamodule=self.data_module, model=module, verbose=False)
        return trainer.callback_metrics["val_acc"]

    def retrain_model(self, num_epochs=2):
        self.trainer = pl.Trainer(accelerator="gpu", max_epochs=1)
        self.trainer.fit(model=self.module, datamodule=self.data_module)


    def prune_model(self, model, amount=0.01):
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
                prune.l1_unstructured(module, name='weight', amount=amount)
                torch.nn.utils.prune.remove(module, 'weight')
        #this method adds new modules and hooks to the model. For simplicity, we do not use this method.
        # prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)

    def integrity(self,payload, extracted):
        return np.mean([extracted[i]==payload[i] for i in range(len(payload))])

    def collect_data(self):
        payload = "".join(list(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 10000)))
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
        print("Model infected w/: ", extractor.extract(self.module.model, message_length))
        # self.safe_model = StegoSafeModel(deepcopy(self.module.model))
        # safe_extracted = extractor.extract(self.safe_model, message_length)
        # print("Channel shuffling extracted:", safe_extracted)
        # cs = self.integrity(payload, safe_extracted)
        cs = 0


        pruning = []
        pruning_performance = []
        for amount in np.linspace(0, 0.9, 10):
            pruned_model = deepcopy(self.module.model)
            self.prune_model(pruned_model, amount=amount)
            extracted = extractor.extract(pruned_model, message_length)
            print(extracted)
            pruning.append(self.integrity(payload, extracted))
            print(pruning[-1])
            pruning_performance.append(self.eval_model(pruned_model))
        retraining = []
        for _ in range(5):
            self.retrain_model()
            retraining.append(self.integrity(payload, extractor.extract(self.module.model, message_length)))
        print("Channel Shuffling: ", cs)
        print("Pruing:", pruning, "performance:", pruning_performance)
        print("Retraining:", retraining)

if __name__ == '__main__':
    cifar10 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to("cuda")

    tb = TestBed(CIFAR10DataModule("../../Datasets/CIFAR10", batch_size=128), cifar10)
    new = deepcopy(cifar10)
    tb.prune_model(new, amount=0.0)
    # compare_model_weights(cifar10, new)
    tb.collect_data()


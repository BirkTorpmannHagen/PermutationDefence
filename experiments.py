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
class TestBed:
    def __init__(self, data_module, model):
        self.module = Classifier(model)
        self.trainer = pl.Trainer(accelerator="gpu", max_epochs=200)
        try:
            self.module.load_from_checkpoint(f"lighting_logs/{data_module.__class__.__name__}_{model.__class__.__name__}.ckpt")
        except FileNotFoundError:
            logger = pl_loggers.TensorBoardLogger("lighting_logs", name=f"{data_module.__class__.__name__}_{model.__class__.__name__}",)
            self.trainer.fit(datamodule=data_module, model=self.module)
        self

    def retrain_model(self, num_epochs=2):
        self.trainer = pl.Trainer(accelerator="gpu", max_epochs=1)
        self.trainer.fit(self.module)


    def prune_model(self, model, amount=0.2):
        prune.global_unstructured(model.parameters(), pruning_method=prune.L1Unstructured, amount=amount)

    def integrity(self,payload, extracted):
        return np.mean([extracted[i]==payload[i] for i in range(len(payload))])
    def collect_data(self):
        payloads = ["".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), i) for i in np.geomspace(10, 10000,4))]
        data = []
        for payload in payloads:
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

            self.safe_model = StegoSafeModel(deepcopy(self.module.model))
            safe_extracted = extractor.extract(self.module.model, message_length)

            cs = self.integrity(payload, safe_extracted)

            pruning = []
            for amount in np.linspace(0, 0.5, 10):
                pruned_model = deepcopy(self.module.model)
                self.prune_model(pruned_model, amount=amount)
                pruning.append(self.integrity(payload, extractor.extract(pruned_model, message_length)))
            retraining = []
            for _ in range(5):
                self.retrain_model()
                retraining.append(self.integrity(payload, extractor.extract(self.module.model, message_length)))


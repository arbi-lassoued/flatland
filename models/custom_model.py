import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override


class FlatlandModel(TorchModelV2, nn.Module):
    """
    Shared policy network for all Flatland agents.

    Architecture:
        Input(231) → FC(256) → ReLU → FC(256) → ReLU → FC(128) → ReLU
        ├── Policy head: Linear(128, num_outputs)
        └── Value head:  Linear(128, 1)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        input_size = int(np.prod(obs_space.shape))

        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, num_outputs)
        self.value_head = nn.Linear(128, 1)

        self._last_features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        features = self.shared(obs)
        self._last_features = features
        logits = self.policy_head(features)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._last_features is not None, "forward() must be called before value_function()"
        return self.value_head(self._last_features).squeeze(1)


# Register globally so train.py / configs can reference it by name
ModelCatalog.register_custom_model("flatland_model", FlatlandModel)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from offlinerlkit.modules.dynamics_module import Swish
from offlinerlkit.nets import EnsembleLinear


def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class EnsembleRewardModel(nn.Module):
    """Ensemble of reward models. Uses EnsembleLinear layers provided in repository."""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        with_action: bool = True,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        dropout_probs: Union[List[float], Tuple[float]] = None,
        reward_final_activation: str = 'none',
        device: str = "cpu"
    ) -> None:
        super().__init__()
        
        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_action = with_action
        self.device = torch.device(device)
        self.activation = activation()
        
        if weight_decays is not None:
            assert len(weight_decays) == (len(hidden_dims) + 1)
        
        # create layers (here we default have no weight decay, just like MILO and stuff)
        module_list = []
        dims = [obs_dim + (action_dim if with_action else 0)] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
            
        if dropout_probs is not None:
            assert len(dropout_probs) == len(hidden_dims)
        else:
            dropout_probs = [0.0] * len(hidden_dims)
        self.dropout_probs = dropout_probs
        
        out_dims = []
        for in_dim, out_dim, weight_decay in zip(dims[:-1], dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
            out_dims.append(out_dim)
        
        self.out_dims = out_dims
        self.backbones = nn.ModuleList(module_list)
        
        # this is binary classification trained with MLE, so 1 output for the positive logit (no dropout on final output)
        self.output_layer = EnsembleLinear(
            dims[-1],
            1,
            num_ensemble,
            weight_decays[-1]
        )
        self.reward_final_activation = reward_final_activation
        
        # min and max logvar only matter in obs dim case (Gaussian)
        self.register_parameter(
            "max_reward",
            nn.Parameter(torch.ones(1) * 1.0, requires_grad=True)
        )
        self.register_parameter(
            "min_reward",
            nn.Parameter(torch.ones(1) * -1.0, requires_grad=True)
        )
        
        # register elite parameters and move to device
        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )
        self.to(self.device)
        
    def forward(self, obs: Union[np.ndarray, torch.Tensor], action: Union[np.ndarray, torch.Tensor], masks: Optional[List[float]] = None, train: bool = True) -> torch.Tensor:
        # just assume action is given, we don't necessarily need it
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
            
        available_mask = masks is not None
        if not available_mask:
            masks = self.generate_masks()
            
        output = torch.cat([obs, action], dim=-1) if self._with_action else obs
        for layer, mask in zip(self.backbones, masks):
            mask = mask.to(self.device)
            output = self.activation(layer(output))
            if train:
                output = output * mask
        
        output = self.output_layer(output) # logits
        output = soft_clamp(output, self.min_reward, self.max_reward)

        if not available_mask:
            return output, masks

        return output
    
    def generate_masks(self) -> List[torch.Tensor]:
        masks = []
        for out_dim, dp in zip(self.out_dims, self.dropout_probs):
            mask = (torch.rand(out_dim) > dp).float()
            masks.append(mask)
        
        return masks
    
    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs
        
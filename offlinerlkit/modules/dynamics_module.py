import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from offlinerlkit.nets import EnsembleLinear


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


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


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        dropout_probs: Optional[Union[List[float], Tuple[float]]] = None,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self.device = torch.device(device)

        self.activation = activation()

        if weight_decays is not None:
            assert len(weight_decays) == (len(hidden_dims) + 1)
            
        if dropout_probs is not None:
            assert len(dropout_probs) == len(hidden_dims) # not doing dropout on last layer
        else:
            dropout_probs = [0.0] * len(hidden_dims)

        module_list = []
        hidden_dims = [obs_dim+action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay, dropout_prob in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1], dropout_probs):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
            if dropout_prob:
                module_list.append(nn.Dropout(p=dropout_prob))
        
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * (obs_dim + self._with_reward),
            num_ensemble,
            weight_decays[-1]
        )

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * -10, requires_grad=True)
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )

        self.to(self.device)

    def forward(self, obs_action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        output = obs_action
        for layer in self.backbones:
            output = layer(output)
            if self.use_dropout:
                # only do activation after dropout
                if isinstance(layer, nn.Dropout):
                    output = self.activation(output)
            else:
                output = self.activation(output)
        
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            if not isinstance(layer, nn.Dropout):
                layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            if not isinstance(layer, nn.Dropout):
                layer.update_save(indexes)
        self.output_layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            if not isinstance(layer, nn.Dropout):
                decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs
    
    
class EnsembleDynamicsModelWithSeparateReward(nn.Module):
    """Separate reward head for BCE loss as opposed to Gaussian MLE."""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        reward_weight_decay: Optional[float] = None,
        dropout_probs: Optional[Union[List[float], Tuple[float]]] = None,
        reward_final_activation: str = 'none',
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self.device = torch.device(device)

        self.activation = activation()

        if weight_decays is not None:
            assert len(weight_decays) == (len(hidden_dims) + 1)
            
        if dropout_probs is not None:
            assert len(dropout_probs) == len(hidden_dims) # not doing dropout on last layer
        else:
            dropout_probs = [0.0] * len(hidden_dims)

        module_list = []
        hidden_dims = [obs_dim+action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        
        for in_dim, out_dim, weight_decay, dropout_prob in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1], dropout_probs):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
            if dropout_prob:
                module_list.append(nn.Dropout(p=dropout_prob))
        
        self.backbones = nn.ModuleList(module_list)

        # gaussian parameters
        self.next_state_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * obs_dim,
            num_ensemble,
            weight_decays[-1]
        )
        
        # bce parameters
        rew_wd = reward_weight_decay if reward_weight_decay is not None else weight_decays[-1]
        self.reward_layer = EnsembleLinear(
            hidden_dims[-1],
            1,
            num_ensemble,
            rew_wd
        )
        self.reward_final_activation = reward_final_activation

        # min and max logvar only matter in obs dim case (Gaussian)
        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(obs_dim) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(obs_dim) * -10, requires_grad=True)
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )

        self.to(self.device)

    def forward(self, obs_action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
            
        # next state stuff
        sp_mean, sp_logvar = torch.chunk(self.next_state_layer(output), 2, dim=-1)
        
        # reward stuff
        reward = self.reward_layer(output)
        if self.reward_final_activation == 'sigmoid':
            reward = F.sigmoid(reward)
        elif self.reward_final_activation == 'relu':
            reward = F.relu(reward)
        
        sp_logvar = soft_clamp(sp_logvar, self.min_logvar, self.max_logvar)
        return sp_mean, sp_logvar, reward

    def load_save(self) -> None:
        for layer in self.backbones:
            if not isinstance(layer, nn.Dropout):
                layer.load_save()
        self.next_state_layer.load_save()
        self.reward_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            if not isinstance(layer, nn.Dropout):
                layer.update_save(indexes)
        self.next_state_layer.update_save(indexes)
        self.reward_layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            if not isinstance(layer, nn.Dropout):
                decay_loss += layer.get_decay_loss()
        decay_loss += self.next_state_layer.get_decay_loss()
        decay_loss += self.reward_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs
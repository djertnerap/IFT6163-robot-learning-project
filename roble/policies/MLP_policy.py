import abc
import itertools
from typing import Iterator

import torch
from torch import distributions, nn
from torch.nn.parameter import Parameter

from roble.infrastructure.pytorch_util import build_mlp
from roble.util.class_util import hidden_member_initialize


class MLPPolicy(nn.Module, metaclass=abc.ABCMeta):
    @hidden_member_initialize
    def __init__(self, ac_dim, ob_dim, n_layers, size, deterministic=False, **kwargs):
        super().__init__(**kwargs)

        self._logits_na = None
        self._mean_net = build_mlp(
            input_size=self._ob_dim, output_size=self._ac_dim, n_layers=self._n_layers, size=self._size
        )
        if not deterministic:
            self._logstd = nn.Parameter(torch.zeros(self._ac_dim, dtype=torch.float32))

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self._deterministic:
            return self._mean_net.parameters()
        else:
            return itertools.chain([self._logstd], self._mean_net.parameters())

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        distribution = self.forward(obs)
        return distribution.sample()

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self._deterministic:
            return self._mean_net(observation)
        else:
            batch_mean = self._mean_net(observation)
            scale_tril = torch.diag(torch.exp(self._logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(batch_mean, scale_tril=batch_scale_tril)
            return action_distribution


class ConcatMLP(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """

    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self._dim)
        return super().forward(flat_inputs, **kwargs)


class MLPPolicyStochastic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """

    def __init__(self, entropy_coeff, *args, **kwargs):
        kwargs["deterministic"] = False
        super().__init__(*args, **kwargs)
        self.entropy_coeff = entropy_coeff

    # def get_action(self, obs):
    #     # TODO: sample actions from the gaussian distribrution given by MLPPolicy policy when providing the observations.
    #     # Hint: make sure to use the reparameterization trick to sample from the distribution
    #     return super().get_action(obs)
    #     # return ptu.to_numpy(action)

    def update(self, observations, q_fun, optimizer):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss
        ## Hint: you will have to add the entropy term to the loss using self.entropy_coeff

        # loss = - q_fun.qa_values(ptu.from_numpy(observations), self.get_action(observations))
        action_distribution = self.forward(observations)
        actions = action_distribution.rsample()
        q_values1 = q_fun.q_net(observations, actions)
        q_values2 = q_fun.q_net2(observations, actions)
        loss = torch.mean(
            -(
                torch.squeeze(torch.minimum(q_values1, q_values2))
                - self.entropy_coeff * action_distribution.log_prob(actions)
            )
        )

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # q_fun.q_net.requires_grad_(True)
        return loss.item()

    def get_loss(self, observations, q_fun):
        action_distribution = self.forward(observations)
        actions = action_distribution.rsample()
        q_values1 = q_fun.q_net(observations, actions)
        q_values2 = q_fun.q_net2(observations, actions)
        loss = torch.mean(
            -(
                torch.squeeze(torch.minimum(q_values1, q_values2))
                - self.entropy_coeff * action_distribution.log_prob(actions)
            )
        )
        return loss


#####################################################

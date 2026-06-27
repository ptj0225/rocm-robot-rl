"""SimBa 네트워크 (ICLR 2025, Lee et al.) — brax PPO용 network_factory.

plain MLP 대신 '관측 정규화 + residual feedforward block + LayerNorm' 구조를 써서
파라미터를 키워도 성능이 떨어지지 않고(simplicity bias) 연속제어·휴머노이드 보행에서
샘플효율이 향상되는 것이 검증됨. brax는 feedforward PPO라 torso만 교체하면 된다.

블록 구조: x -> x + Dense(h) -> [LayerNorm -> Dense(4h) -> relu -> Dense(h) -> +residual] * K
          -> LayerNorm -> Dense(out)
"""

from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import linen
from brax.training import distribution, types
from brax.training import networks
from brax.training.agents.ppo.networks import PPONetworks

Initializer = networks.Initializer


class SimbaTorso(linen.Module):
    """SimBa residual torso. 마지막에 out_size로 사상."""

    out_size: int
    num_blocks: int = 2
    hidden: int = 256
    activation: networks.ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = linen.Dense(self.hidden, kernel_init=self.kernel_init, name="proj")(x)
        for i in range(self.num_blocks):
            res = h
            y = linen.LayerNorm(name=f"ln_{i}")(h)
            y = linen.Dense(self.hidden * 4, kernel_init=self.kernel_init, name=f"mlp_{i}_0")(y)
            y = self.activation(y)
            y = linen.Dense(self.hidden, kernel_init=self.kernel_init, name=f"mlp_{i}_1")(y)
            h = res + y
        h = linen.LayerNorm(name="ln_out")(h)
        return linen.Dense(self.out_size, kernel_init=self.kernel_init, name="out")(h)


def _make_ffn(module, obs_size, obs_key, preprocess_observations_fn, squeeze=False):
    """brax FeedForwardNetwork(init, apply) 래퍼 — dict 관측/정규화 처리 포함."""

    def apply(processor_params, params, obs):
        if isinstance(obs, Mapping):
            o = preprocess_observations_fn(
                obs[obs_key], networks.normalizer_select(processor_params, obs_key)
            )
        else:
            o = preprocess_observations_fn(obs, processor_params)
        out = module.apply(params, o)
        return jnp.squeeze(out, axis=-1) if squeeze else out

    size = networks._get_obs_state_size(obs_size, obs_key)
    dummy = jnp.zeros((1, size))
    return networks.FeedForwardNetwork(init=lambda key: module.init(key, dummy), apply=apply)


def make_simba_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    policy_num_blocks: int = 2,
    policy_hidden: int = 256,
    value_num_blocks: int = 2,
    value_hidden: int = 512,
    activation: networks.ActivationFn = linen.relu,
) -> PPONetworks:
    """SimBa torso를 쓰는 PPONetworks. brax make_ppo_networks와 호환 인터페이스."""
    dist = distribution.NormalTanhDistribution(event_size=action_size)

    policy_module = SimbaTorso(
        out_size=dist.param_size, num_blocks=policy_num_blocks, hidden=policy_hidden, activation=activation
    )
    value_module = SimbaTorso(
        out_size=1, num_blocks=value_num_blocks, hidden=value_hidden, activation=activation
    )

    policy_network = _make_ffn(policy_module, observation_size, policy_obs_key, preprocess_observations_fn)
    value_network = _make_ffn(
        value_module, observation_size, value_obs_key, preprocess_observations_fn, squeeze=True
    )
    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=dist,
    )

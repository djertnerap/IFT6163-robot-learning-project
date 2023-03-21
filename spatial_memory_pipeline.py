# For now, this is just a template method to help figure out the structure.
import numpy as np
import torch


def spatial_memory_pipeline():
    # 1: perform step in env & receive observation (This could probably be the input to the function)

    # A
    # A1: encode observation with CNN autoencoder

    # A2: Compare embedding to stored embeddings

    # A3: Activate them proportionally to similarity (I think?)
    # P_reactivation(s | y_t, M) = e^{beta * y_t^T * m_s^(y)}
    # in essence, this is exp(beta * current_obs_embedding.T * mem_slot_s_embedding)
    # Note: beta = way to get its value is TODO...
    # This is the target of the model
    p_react = torch.exp_(...)

    # B
    # B1: input velocities to RNNs to predict memory slots activation

    # B2: with probability p_correction, apply correction in the RNNs

    # B3: get full prediction as the product of all RNNs activations

    # C
    # C1: calculate loss from target & prediction

    # C2 perform update

    # D (I think it may be done before C & doesn't matter)
    # D1: with probability p_storage, store y_t, x_{1,s}, x_{2,s} & x_{3,s} in memory slots.
    pass


def get_new_beta(p_react: float, H_react: float, previous_beta: float) -> float:
    """
    Gets the newly regulated parameter beta used to calculate the target memory reactivation.
    This should be called after every trajectory
    """
    beta_logit = np.log(previous_beta)
    # Perhaps we could apply a certain rounding that defines when they are close enough for us not to change it?
    if p_react < H_react:
        beta_logit += 0.001
    elif p_react > H_react:
        beta_logit -= 0.001
    return np.exp(beta_logit)


class MemoryMap:
    # I think this will be a class

    # M = {(m_s^(y), m_{1, s}^(x), ... m_{R, s}^(x))}_{s in 1...S}
    # 1 slot = {y (upstream input embedding), x_1, x_2, ..., x_R (RNNs' state)}
    pass

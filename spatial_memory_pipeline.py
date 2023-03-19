# For now, this is just a template method to help figure out the structure.
def spatial_memory_pipeline():
    # 1: perform step in env & receive observation (This could probably be the input to the function)

    # A
    # A1: encode observation with CNN autoencoder

    # A2: Compare embedding to stored embeddings

    # A3: Activate them proportionally to similarity (I think?)
    # P_reactivation(s | y_t, M) = e^{beta * y_t^T * m_s^(y)}
    # in essence, this is exp(beta * current_obs_embedding.T * mem_slot_s_embedding)
    # Note: beta = way to get its value is ...

    # B
    # B1: input velocities to RNNs to predict memory slots activation
    pass


class MemoryMap:
    # I think this will be a class

    # M = {(m_s^(y), m_{1, s}^(x), ... m_{R, s}^(x))}_{s in 1...S}
    # 1 slot = {y (upstream input embedding), x_1, x_2, ..., x_R (RNNs' state)}
    pass

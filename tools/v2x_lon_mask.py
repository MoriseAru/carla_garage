"""Longitudinal-only token routing for the TF++ V2X decoder (LOCAL HANDOVER 10328, section 1).

The decoder has predict_checkpoint_len (10) path queries followed by 1 target-speed query; the last v2x_k (16) memory
positions are the cooperative-vehicle tokens. With the hook installed:
  memory_mask (Q x S): rows 0..P-1 (path queries) are -inf on the last K columns (token slots), 0 elsewhere;
  tgt_mask    (Q x Q): rows 0..P-1 are -inf on column P (the speed query), 0 elsewhere.
The speed query is unrestricted. Hence the path outputs are exactly independent of the tokens.
Installed as a forward pre-hook on net.join (plain nn.TransformerDecoder), so the state_dict is unchanged.
"""
import torch


def install(net):
    cfg = net.config
    P, K = int(cfg.predict_checkpoint_len), int(cfg.v2x_k)
    assert isinstance(net.join, torch.nn.TransformerDecoder), type(net.join)

    def hook(mod, args, kwargs):
        query, memory = args[0], args[1]
        Q, S = query.shape[1], memory.shape[1]
        assert Q == P + 1, (Q, P)
        mm = torch.zeros(Q, S, dtype=memory.dtype, device=memory.device)
        mm[:P, S - K:] = float("-inf")
        tm = torch.zeros(Q, Q, dtype=query.dtype, device=query.device)
        tm[:P, P] = float("-inf")
        kwargs = dict(kwargs, tgt_mask=tm, memory_mask=mm)
        return args, kwargs

    return net.join.register_forward_pre_hook(hook, with_kwargs=True)

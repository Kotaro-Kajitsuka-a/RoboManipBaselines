from .WrenchPredictor4MLPModel import WrenchPredictor4MLPModel
from .WrenchPredictor4TransformerModel import WrenchPredictor4TransformerModel


def WrenchPredictor4Model(
    image_feature_dim,
    state_dim,
    action_dim,
    wrench_dim,
    num_objects,
    pb_dim,
    horizon,
    n_obs_steps,
    hidden_dim=256,
    nhead=8,
    num_encoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    output_head="mlp_only",
    wrench_loss_weight=1.0,
) -> WrenchPredictor4MLPModel | WrenchPredictor4TransformerModel:
    """Create the WP4 architecture selected by the CLI or checkpoint metadata."""
    model_args = dict(
        image_feature_dim=image_feature_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        wrench_dim=wrench_dim,
        num_objects=num_objects,
        pb_dim=pb_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        wrench_loss_weight=wrench_loss_weight,
    )
    if output_head == "mlp":
        return WrenchPredictor4TransformerModel(
            **model_args,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
        )
    elif output_head == "mlp_only":
        return WrenchPredictor4MLPModel(**model_args)
    else:
        raise ValueError(f"Unsupported output_head: {output_head}")

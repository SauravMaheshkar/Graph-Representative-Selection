"""Default Hyperparameter Configuration for the RS-GNN Model"""
import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Data Hyperparameters
    config.data_path = "data/"
    config.dataset = "cora"

    # Model Hyperparameters
    config.activation_fn = "PReLU"
    config.lambda_value = 0.001
    config.gcn_c_hidden_dim = 32
    config.gcn_c_weight_decay = 5e-4
    config.rsgnn_hidden_dim = 512

    # Training Hyperparameters
    config.gcn_c_epochs = 1000
    config.rsgnn_epochs = 2000
    config.num_rep_multiplier = 2
    config.learning_rate = 0.001
    config.dropout_rate = 0.5

    # Validation Hyperparameters
    config.valid_epochs = 10
    config.num_valid_nodes = 500

    # Integer for PRNG random seed
    config.seed = 42

    return config


def get_rsgnn_flags(
    num_classes: int, config: ml_collections.ConfigDict
) -> ml_collections.ConfigDict:
    """
    Update ConfigDict as per RSGNN

    :param num_classes: Number of Classes
    :type num_classes: int
    :param config: ML Collections ConfigDict
    :type config: ml_collections.ConfigDict
    :return: Updated ConfigDict
    :rtype: ml_collections.ConfigDict
    """
    config.hid_dim = config.rsgnn_hidden_dim
    config.epochs = config.rsgnn_epochs
    config.num_classes = num_classes
    config.num_reps = config.num_rep_multiplier * config.num_classes
    return config

from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf
import fedDropShake_android
import fl_server
from flwr.server.client_manager import SimpleClientManager


def main() -> None:
    # Create strategy, and define number of clients
    strategy = fedDropShake_android.FedDropShakeAndroid(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=5,
        min_eval_clients=5,
        min_available_clients=5,
        eval_fn=None,
        on_fit_config_fn=fit_config,
        initial_parameters=None,
    )

    # Start Flower server for 65 rounds of federated learning
    client_manager = SimpleClientManager()
    server = fl_server.Server(client_manager=client_manager, strategy=strategy)
    fl.server.start_server(
        "192.168.1.7:1999",
        config={
            "num_rounds": 35},
        server=server,
        strategy=strategy)


def fit_config(rnd: int, p=1.0):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 128,
        "local_epochs": 1,
        "p_val": p,
    }
    return config


if __name__ == "__main__":
    main()

from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf
import fedDropFem_android
import fl_server
from flwr.server.client_manager import SimpleClientManager


def main() -> None:
    # Create strategy, and define number of clients
    strategy = fedDropFem_android.FedDropFemAndroid(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        eval_fn=None,
        on_fit_config_fn=fit_config,
        initial_parameters=None,
    )

    # Start Flower server for 250 rounds of federated learning
    client_manager = SimpleClientManager()
    server = fl_server.Server(client_manager=client_manager, strategy=strategy)
    fl.server.start_server(
        "10.0.0.76:1999",
        config={
            "num_rounds": 10},
        server=server,
        strategy=strategy)


def fit_config(rnd: int, iter=1, p=1.0):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 10,
        "local_epochs": iter,
        "p_val": p,
    }
    return config


if __name__ == "__main__":
    main()

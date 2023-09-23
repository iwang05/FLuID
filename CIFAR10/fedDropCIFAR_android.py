# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Modifications Copyright 2023 The FLuID Authors. All Rights Reserved.
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.
#
# Modifications made to enable federated dropout with FedAvg. 
# Modification include additional methods drop_rand, drop_order, drop_dynamic()
# find_min(), find_stable(), aggregate_drop(),
# and other changes documented in comments below
# ==============================================================================
"""Federated Dropout strategy for CIFAR10 dataset, based on Federated Averaging (FedAvg) [McMahan et al., 2016] strategy with custom
serialization for Android devices.

FedAvg Paper: https://arxiv.org/abs/1602.05629
"""

from logging import WARNING, INFO
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import sys

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy
import random
from functools import reduce

DEPRECATION_WARNING = """
DEPRECATION WARNING: deprecated `eval_fn` return format

    loss, accuracy

move to

    loss, {"accuracy": accuracy}

instead. Note that compatibility with the deprecated return format will be
removed in a future release.
"""

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""


class FedDropCIFARAndroid(Strategy):
    """Configurable Federated Dropout (CIFAR10) strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self,
                 fraction_fit: float = 0.1,
                 fraction_eval: float = 0.1,
                 min_fit_clients: int = 2,
                 min_eval_clients: int = 2,
                 min_available_clients: int = 2,
                 eval_fn: Optional[Callable[[Weights],
                                            Optional[Tuple[float,
                                                           Dict[str,
                                                                Scalar]]]]] = None,
                 on_fit_config_fn: Optional[Callable[[int],
                                            Dict[str,
                                                 Scalar]]] = None,
                 on_evaluate_config_fn: Optional[Callable[[int],
                                                 Dict[str,
                                                      Scalar]]] = None,
                 accept_failures: bool = True,
                 initial_parameters: Optional[Parameters] = None,
                 ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Args:
            fraction_fit (float, optional): Fraction of clients used during
                training. Defaults to 0.1.
            fraction_eval (float, optional): Fraction of clients used during
                validation. Defaults to 0.1.
            min_fit_clients (int, optional): Minimum number of clients used
                during training. Defaults to 2.
            min_eval_clients (int, optional): Minimum number of clients used
                during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total
                clients in the system. Defaults to 2.
            eval_fn (Callable[[Weights], Optional[Tuple[float, float]]], optional):
                Function used for validation. Defaults to None.
            on_fit_config_fn (Callable[[int], Dict[str, Scalar]], optional):
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn (Callable[[int], Dict[str, Scalar]], optional):
                Function used to configure validation. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds
                containing failures. Defaults to True.
            initial_parameters (Parameters, optional): Initial global model parameters.
        """
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.droppedWeights: Dict[str, List] = {}
        # list of straggler cids
        self.straggler: Dict[str, float] = {}
        self.JustUpdatedStrag = False

        # Multi dimension shape of the weight parameters
        # all shapes are 1-d when sent back from Android clients, hence we need
        # to reshape it when sent back to server
        self.weight_shapes = [(3, 3, 3, 32), (32,), (3, 3, 32, 32), (32,), (3, 3, 32, 64), (64,), (3, 3, 64, 64), (64,), (
            3, 3, 64, 128), (128,), (3, 3, 128, 128), (128,), (2048, 512), (512,), (512, 256), (256,), (256, 10), (10,)]
        self.droppedWeightsShape: Dict[str, List] = {}

        # sub-model size (defualt to 0.95, will be initialized a round 2)
        self.p_val = 0.95
        self.newIteration = 5
        self.parameters: Parameters

        # list to save invariant weight inidcies
        self.unchagedWeights = [[] for x in range(len(self.weight_shapes))]
        self.defDropWeights = [[] for x in range(len(self.weight_shapes))]
        self.prevDropWeights = [[] for x in range(len(self.weight_shapes))]
        self.append = False

        # update threshold (will be initialized at round 2 based on training results)
        # FOR CIFAR10 since all layer have drastically different weight update % pattern,
        # we assign an individual threshold for each layer
        self.changeThreshold = [1.5 for x in range(len(self.weight_shapes))]
        self.changeIncrement = 0.2
        self.roundCounter = 0
        self.stopChange = [False for x in range(len(self.weight_shapes))]

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(
            num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(
            self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(
            num_clients, self.min_eval_clients), self.min_available_clients

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if isinstance(initial_parameters, list):
            log(WARNING, DEPRECATION_WARNING_INITIAL_PARAMETERS)
            initial_parameters = self.weights_to_parameters(
                weights=initial_parameters)
        return initial_parameters

    def evaluate(
            self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = self.parameters_to_weights(parameters)
        eval_res = self.eval_fn(weights)
        if eval_res is None:
            return None
        loss, other = eval_res
        if isinstance(other, float):
            print(DEPRECATION_WARNING)
            metrics = {"accuracy": other}
        else:
            metrics = other
        return loss, metrics

    def configure_fit(
            self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.parameters = parameters
        config = {}
        config_drop = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
            config_drop = self.on_fit_config_fn(rnd, p=self.p_val)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        clientList = []
        for client in clients:
            # create submodel for any stragglers
            if (client.cid in self.straggler) and rnd > 2:
                # Select the Dropout technique here from : drop_rand,
                # drop_order, and drop_dynamic (Invariant dropout)
                fit_ins_drop = FitIns(
                    self.drop_dynamic(
                        parameters, self.p_val, [
                            0, 2, 4, 6, 8, 12, 14], 10, client.cid), config_drop)
                clientList.append((client, fit_ins_drop))
            else:
                clientList.append((client, fit_ins))
        return clientList

    def configure_evaluate(
            self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        config = {}
        config_drop = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
            config_drop = self.on_evaluate_config_fn(rnd, self.p_val)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # send the same global model to all clients (straggler or not)
        clientList = []
        for client in clients:
            if (client.cid in self.straggler) and rnd > 2:
                evaluate_ins_drop = EvaluateIns(parameters, config_drop)
                clientList.append((client, evaluate_ins_drop))
            else:
                clientList.append((client, evaluate_ins))
        return clientList

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (self.parameters_to_weights(fit_res.parameters), fit_res.num_examples, client.cid)
            for client, fit_res in results
        ]

        idxlist = [0, 2, 4, 6, 8, 12, 14]

        # Total training round for CIFAR10 set to 35
        # FOR CIFAR10 since all layer have drastically different weight update % pattern,
        # we assign an individual threshold for each layer
        if (rnd > 10):
            self.roundCounter += 1
            if (self.roundCounter >= 2):
                for idx in idxlist:
                    if not self.stopChange[idx]:
                        self.changeThreshold[idx] += self.changeIncrement
                        self.roundCounter = 0
                        print(
                            "threshold for",
                            idx,
                            "updated to: ",
                            self.changeThreshold[idx])

        self.find_stable(
            self.parameters, weights_results, [
                0, 2, 4, 6, 8, 12, 14], 10)
        self.find_min(
            self.parameters, weights_results, [
                0, 2, 4, 6, 8, 12, 14], rnd)

        weights = self.aggregate_drop(
            weights_results,
            self.droppedWeights,
            self.weight_shapes,
            self.droppedWeightsShape)

        # sort the clients based on training duration (at round2 )
        if (len(self.straggler) == 0) and rnd > 1:
            def time(elem):
                return elem[1].metrics.get('duration')

            results.sort(key=time)
            self.straggler[results[len(results) -
                                   1][0].cid] = results[len(results) -
                                                        1][1].metrics.get('duration')
            print(self.straggler)

            # Set sub-model size based on slowest client vs targert time
            stragglerDur = results[len(results) - 1][1].metrics.get('duration')
            nextSlowDur = results[len(results) - 2][1].metrics.get('duration')
            percentDiff = nextSlowDur / stragglerDur
            if (percentDiff >= 0.90):
                self.p_val = 0.95
            elif (percentDiff < 0.90 and percentDiff >= 0.80):
                self.p_val = 0.85
            elif (percentDiff < 0.80 and percentDiff >= 0.70):
                self.p_val = 0.75
            elif (percentDiff < 0.70 and percentDiff >= 0.60):
                self.p_val = 0.65
            else:
                self.p_val = 0.5
            print("Updated p val to:", self.p_val)

        # for remaining rounds check if straggler changed
        elif (len(self.straggler) != 0) and rnd > 1 and self.JustUpdatedStrag == False:
            def time(elem):
                return elem[1].metrics.get('duration')

            results.sort(key=time)
            slowest = results[len(results) - 1]
            if (slowest[0].cid not in self.straggler):
                # estimate current straggler's original training time without
                # dropout
                for i in range(len(results)):
                    if results[i][0].cid in self.straggler:
                        self.straggler[results[i][0].cid] = (
                            results[i][1].metrics.get('duration') / self.p_val) * 1.15
                        print("Updated estimate straggler orig time to:",
                              self.straggler[results[i][0].cid])
                stragglerList = list(self.straggler.items())

                # Compare slowest device against straggler's orig training time
                if (slowest[1].metrics.get('duration') > stragglerList[0][1]):
                    self.straggler[slowest[0].cid] = slowest[1].metrics.get(
                        'duration')

                    # Set sub-model size based on slowest client vs targert
                    # time
                    stragglerDur = slowest[1].metrics.get('duration')
                    nextSlowDur = stragglerList[0][1]
                    percentDiff = nextSlowDur / stragglerDur
                    if (percentDiff >= 0.90):
                        self.p_val = 0.95
                    elif (percentDiff < 0.90 and percentDiff >= 0.80):
                        self.p_val = 0.85
                    elif (percentDiff < 0.80 and percentDiff >= 0.70):
                        self.p_val = 0.75
                    elif (percentDiff < 0.70 and percentDiff >= 0.60):
                        self.p_val = 0.65
                    else:
                        self.p_val = 0.5
                    print("Updated p val to:", self.p_val)
                    self.JustUpdatedStrag = True

                    self.straggler.pop(stragglerList[0][0])
                    self.droppedWeights.pop(stragglerList[0][0])
                    stragglerList.pop(0)
            print(self.straggler)
        else:
            self.JustUpdatedStrag = False

        return self.weights_to_parameters(weights), {}

    def aggregate_evaluate(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        _, evaluate_res = results[0]
        if 'accuracy' in evaluate_res.metrics:
            acc_aggregated = weighted_loss_avg(
                [
                    (
                        evaluate_res.num_examples,
                        evaluate_res.metrics.get('accuracy'),
                        evaluate_res.accuracy,
                    )
                    for _, evaluate_res in results
                ]
            )

        log(
            INFO,
            "eval progress: (%s, %s, %s)",
            rnd,
            loss_aggregated,
            {'accuracy': acc_aggregated},
        )

        return loss_aggregated, {'accuracy': acc_aggregated}

    def weights_to_parameters(self, weights: Weights) -> Parameters:
        """Convert NumPy weights to parameters object."""
        tensors = [self.ndarray_to_bytes(ndarray) for ndarray in weights]
        return Parameters(tensors=tensors, tensor_type="numpy.nda")

    def parameters_to_weights(self, parameters: Parameters) -> Weights:
        """Convert parameters object to NumPy weights."""
        return [self.bytes_to_ndarray(tensor) for tensor in parameters.tensors]

    # pylint: disable=R0201
    def ndarray_to_bytes(self, ndarray: np.ndarray) -> bytes:
        """Serialize NumPy array to bytes."""
        return cast(bytes, ndarray.tobytes())

    # pylint: disable=R0201
    def bytes_to_ndarray(self, tensor: bytes) -> np.ndarray:
        """Deserialize NumPy array from bytes."""
        ndarray_deserialized = np.frombuffer(tensor, dtype=np.float32)
        return cast(np.ndarray, ndarray_deserialized)

    def drop_rand(
            self,
            parameters: Parameters,
            p: float,
            idxList: List[int],
            idxConvFC: int,
            cid: str):
        """Baseline 1: Federated Dropout technique - random dropout """
        # Note: for each layer in the model, it has related weight parameters in 3 indices:
        #       activation, bias, and input of next layer

        # Args: parameters: global model parameters
        #          p: sub-model size
        #          idxList: list for the starting indices of each layer
        #          idxConvFC: the index of the last convolutional layer before the FC layer
        #          cid: the id of the straggler client

        weights = self.parameters_to_weights(parameters)
        # Weight parameters are sent back as 1D arrays from the Android, clients, it is easier to transform
        # the weights back to original multi-D shape for dropout

        weights[0] = np.reshape(weights[0], self.weight_shapes[0])
        weights[2] = np.reshape(weights[2], self.weight_shapes[2])
        weights[4] = np.reshape(weights[4], self.weight_shapes[4])
        weights[6] = np.reshape(weights[6], self.weight_shapes[6])
        weights[8] = np.reshape(weights[8], self.weight_shapes[8])
        weights[10] = np.reshape(weights[10], self.weight_shapes[10])
        weights[12] = np.reshape(weights[12], self.weight_shapes[12])
        weights[14] = np.reshape(weights[14], self.weight_shapes[14])
        weights[16] = np.reshape(weights[16], self.weight_shapes[16])

        # Initialize list variables for the straggler
        if cid not in self.droppedWeights:
            self.droppedWeights[cid] = [[[], []] for x in range(len(weights))]
            self.droppedWeightsShape[cid] = [x for x in self.weight_shapes]

        # for each layer, select (1-p)% neurons to dropout
        for idx in idxList:
            # indicies of the corresponding dimension that the weight matrix is reduced (
            # ie for convolutional layers it is the number of filters, for FC
            # layers its the number of output neurons
            first = weights[idx].ndim - 1
            second = weights[idx + 1].ndim - 1
            third = weights[idx + 2].ndim - 2

            # calculate number of neurons to drop based on shape
            shape = weights[idx].shape
            numToDrop = shape[first] - int(p * shape[first])

            # randomly select neurons to drop
            fullList = [x for x in range(shape[first])]
            list = sorted(random.sample(fullList, numToDrop))
            self.prevDropWeights[idx] = list.copy()
            print(
                "Dropped weights idx ",
                idx,
                ": ",
                (self.prevDropWeights[idx]))

            self.droppedWeights[cid][idx][0] = list.copy()
            self.droppedWeights[cid][idx + 1][0] = list.copy()
            self.droppedWeights[cid][idx + 2][1] = list.copy()

            # remove each row/column from the back
            weights[idx] = np.delete(weights[idx], list, first)
            weights[idx + 1] = np.delete(weights[idx + 1], list, second)
            weights[idx + 2] = np.delete(weights[idx + 2], list, third)

        # Dropping for the last convolutional layer (has to be flattened for FC
        # layer)

        # indices of the corresponding dimension that the weight matrix is reduced (
        # ie for convolutional layers it is the number of filters, for FC
        # layers its the number of output neurons
        first = weights[idxConvFC].ndim - 1
        second = weights[idxConvFC + 1].ndim - 1
        third = weights[idxConvFC + 2].ndim - 2

        shape = weights[idxConvFC].shape
        listFC: Tuple = random.sample(
            range(1, shape[first]), shape[first] - int(p * shape[first]))
        listFC.sort()

        numToDrop = shape[first] - int(p * shape[first])
        fullList = [x for x in range(shape[first])]

        listFC = sorted(random.sample(fullList, numToDrop))
        self.prevDropWeights[idxConvFC] = listFC.copy()
        print(
            "Dropped weights idx ",
            idxConvFC,
            ": ",
            (self.prevDropWeights[idxConvFC]))

        # Extend drop list for the input of first fully connected layer
        FClist = []
        for drop in listFC:
            FClist.extend(range(drop * 4 * 4, drop * 4 * 4 + 4 * 4))

        # drop the neurons from each weight parameter ( # filters for conv
        # layer, bias, # of inputs for first FC layer)
        self.droppedWeights[cid][idxConvFC][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 1][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 2][1] = FClist.copy()

        weights[idxConvFC] = np.delete(weights[idxConvFC], listFC, first)
        weights[idxConvFC +
                1] = np.delete(weights[idxConvFC +
                                       1], listFC, second)
        weights[idxConvFC +
                2] = np.delete(weights[idxConvFC +
                                       2], FClist, third)

        # record the shape fo the resulting sub-model
        for i in range(len(weights)):
            self.droppedWeightsShape[cid][i] = weights[i].shape

        # flatten the weight parameters to 1D before sending to Android Clients
        weights[0] = weights[0].flatten()
        weights[2] = weights[2].flatten()
        weights[4] = weights[4].flatten()
        weights[6] = weights[6].flatten()
        weights[8] = weights[8].flatten()
        weights[10] = weights[10].flatten()
        weights[12] = weights[12].flatten()
        weights[14] = weights[14].flatten()
        weights[16] = weights[16].flatten()

        return self.weights_to_parameters(weights)

    def drop_order(
            self,
            parameters: Parameters,
            p: float,
            idxList: List[int],
            idxConvFC: int,
            cid: str):
        """Baseline 2: Fjord technique - Ordered dropout """
        # Note: for each layer in the model, it has related weight parameters in 3 indices:
        #       activation, bias, and input of next layer

        # Args: parameters: global model parameters
        #          p: sub-model size
        #          idxList: list for the starting indices of each layer
        #          idxConvFC: the index of the last convolutional layer before the FC layer
        #          cid: the id of the straggler client

        # Weight parameters are sent back as 1D arrays from the Android, clients, it is easier to transform
        # the weights back to original multi-D shape for dropout
        weights = self.parameters_to_weights(parameters)
        weights[0] = np.reshape(weights[0], self.weight_shapes[0])
        weights[2] = np.reshape(weights[2], self.weight_shapes[2])
        weights[4] = np.reshape(weights[4], self.weight_shapes[4])
        weights[6] = np.reshape(weights[6], self.weight_shapes[6])
        weights[8] = np.reshape(weights[8], self.weight_shapes[8])
        weights[10] = np.reshape(weights[10], self.weight_shapes[10])
        weights[12] = np.reshape(weights[12], self.weight_shapes[12])
        weights[14] = np.reshape(weights[14], self.weight_shapes[14])
        weights[16] = np.reshape(weights[16], self.weight_shapes[16])

        # Initialize list variables for the straggler
        if cid not in self.droppedWeights:
            self.droppedWeights[cid] = [[[], []] for x in range(len(weights))]
            self.droppedWeightsShape[cid] = [x for x in self.weight_shapes]

        # for each layer, select (1-p)% neurons to dropout
        for idx in idxList:
            # indices of the corresponding dimension that the weight matrix is reduced
            # ie for convolutional layers it is the number of filters, for FC
            # layers its the number of output neurons
            first = weights[idx].ndim - 1
            second = weights[idx + 1].ndim - 1
            third = weights[idx + 2].ndim - 2

            # calculate number of neurons to drop based on shape
            # drop (1-p)% neurons from the left of the model
            shape = weights[idx].shape
            numToDrop = shape[first] - int(p * shape[first])
            list = [x for x in range(shape[first] - numToDrop, shape[first])]

            # save a copy of the dropped neurons
            self.prevDropWeights[idx] = list.copy()
            print(
                "Dropped weights idx ",
                idx,
                ": ",
                (self.prevDropWeights[idx]))

            self.droppedWeights[cid][idx][0] = list.copy()
            self.droppedWeights[cid][idx + 1][0] = list.copy()
            self.droppedWeights[cid][idx + 2][1] = list.copy()

            # remove each row/column from the back
            weights[idx] = np.delete(weights[idx], list, first)
            weights[idx + 1] = np.delete(weights[idx + 1], list, second)
            weights[idx + 2] = np.delete(weights[idx + 2], list, third)

        # Dropping for the last convolutional layer (has to be flattened for FC
        # layer)

        # indicies of the corresponding dimension that the weight matrix is reduced (
        # ie for convolutional layers it is the number of filters, for FC
        # layers its the number of output neurons
        first = weights[idxConvFC].ndim - 1
        second = weights[idxConvFC + 1].ndim - 1
        third = weights[idxConvFC + 2].ndim - 2

        shape = weights[idxConvFC].shape
        listFC: Tuple = random.sample(
            range(1, shape[first]), shape[first] - int(p * shape[first]))
        listFC.sort()

        # drop (1-p)% neurons from the left of the model
        numToDrop = shape[first] - int(p * shape[first])
        listFC = [x for x in range(shape[first] - numToDrop, shape[first])]
        self.prevDropWeights[idxConvFC] = listFC.copy()
        print(
            "Dropped weights idx ",
            idxConvFC,
            ": ",
            (self.prevDropWeights[idxConvFC]))

        # Extend drop list for the input of first fully connected layer
        FClist = []
        for drop in listFC:
            FClist.extend(range(drop * 4 * 4, drop * 4 * 4 + 4 * 4))

        # drop the neurons from each weight parameter ( # filters for conv
        # layer, bias, # of inputs for first FC layer)
        self.droppedWeights[cid][idxConvFC][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 1][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 2][1] = FClist.copy()
        weights[idxConvFC] = np.delete(weights[idxConvFC], listFC, first)
        weights[idxConvFC +
                1] = np.delete(weights[idxConvFC +
                                       1], listFC, second)
        weights[idxConvFC +
                2] = np.delete(weights[idxConvFC +
                                       2], FClist, third)

        for i in range(len(weights)):
            self.droppedWeightsShape[cid][i] = weights[i].shape

        # flatten the weight parameters to 1D before sending to Android Clients
        weights[0] = weights[0].flatten()
        weights[2] = weights[2].flatten()
        weights[4] = weights[4].flatten()
        weights[6] = weights[6].flatten()
        weights[8] = weights[8].flatten()
        weights[10] = weights[10].flatten()
        weights[12] = weights[12].flatten()
        weights[14] = weights[14].flatten()
        weights[16] = weights[16].flatten()

        return self.weights_to_parameters(weights)

    def drop_dynamic(
            self,
            parameters: Parameters,
            p: float,
            idxList: List[int],
            idxConvFC: int,
            cid: str):
        """Invariant Dropout - create sub-models based on unchanging neurons """
        # Note: for each layer in the model, it has related weight parameters in 3 indices:
        #       activation, bias, and input of next layer

        # Args: parameters: global model parameters
        #          p: sub-model size
        #          idxList: list for the starting indices of each layer
        #          idxConvFC: the index of the last convolutional layer before the FC layer
        #          cid: the id of the straggler client

        weights = self.parameters_to_weights(parameters)
        # self.weight_shapes = [(5, 5, 1, 16), (16,), (5, 5, 16, 64), (64,), (3136, 120), (120,), (120, 62), (62,)]
        # Weight parameters are sent back as 1D arrays from the Android, clients, it is easier to transform
        # the weights back to original multi-D shape for dropout
        weights[0] = np.reshape(weights[0], self.weight_shapes[0])
        weights[2] = np.reshape(weights[2], self.weight_shapes[2])
        weights[4] = np.reshape(weights[4], self.weight_shapes[4])
        weights[6] = np.reshape(weights[6], self.weight_shapes[6])
        weights[8] = np.reshape(weights[8], self.weight_shapes[8])
        weights[10] = np.reshape(weights[10], self.weight_shapes[10])
        weights[12] = np.reshape(weights[12], self.weight_shapes[12])
        weights[14] = np.reshape(weights[14], self.weight_shapes[14])
        weights[16] = np.reshape(weights[16], self.weight_shapes[16])

        # Initialize list variables for the straggler
        if cid not in self.droppedWeights:
            self.droppedWeights[cid] = [[[], []] for x in range(len(weights))]
            self.droppedWeightsShape[cid] = [x for x in self.weight_shapes]

        # for each layer, select (1-p)% neurons to dropout
        for idx in idxList:
            # indices of the corresponding dimension that the weight matrix is reduced
            # ie for convolutional layers it is the number of filters, for FC
            # layers its the number of output neurons
            first = weights[idx].ndim - 1
            second = weights[idx + 1].ndim - 1
            third = weights[idx + 2].ndim - 2

            # calculate number of neurons to drop based on shape
            shape = weights[idx].shape
            numToDrop = shape[first] - int(p * shape[first])

            # first, prioritize dropping any neurons in the defDropWeights list (repeatedly under threshold)
            # next drop neurons that are under the threshold for this round
            # Finally, randomly drop neurons if needed
            if len(self.defDropWeights[idx]) >= numToDrop:
                if (idx == 0):
                    self.stopChange[idx] = True
                dropList = sorted(
                    random.sample(
                        self.defDropWeights[idx],
                        numToDrop))
            elif len(self.unchagedWeights[idx]) >= numToDrop:
                self.stopChange[idx] = True

                fullList = self.unchagedWeights[idx].copy()
                for x in self.defDropWeights[idx]:
                    fullList.remove(x)
                dropList = random.sample(
                    fullList, numToDrop - len(self.defDropWeights[idx]))
                dropList.extend(self.defDropWeights[idx])
                dropList.sort()

            else:
                fullList = [x for x in range(shape[first])]
                for x in self.unchagedWeights[idx]:
                    fullList.remove(x)
                dropList = random.sample(
                    fullList, numToDrop - len(self.unchagedWeights[idx]))
                dropList.extend(self.unchagedWeights[idx])
                dropList.sort()

            # save a copy of the dropped neurons
            self.prevDropWeights[idx] = dropList.copy()
            print(
                "Dropped weights idx ",
                idx,
                ": ",
                (self.prevDropWeights[idx]))

            self.droppedWeights[cid][idx][0] = dropList.copy()
            self.droppedWeights[cid][idx + 1][0] = dropList.copy()
            self.droppedWeights[cid][idx + 2][1] = dropList.copy()

            # remove each row/column from the back
            weights[idx] = np.delete(weights[idx], dropList, first)
            weights[idx + 1] = np.delete(weights[idx + 1], dropList, second)
            weights[idx + 2] = np.delete(weights[idx + 2], dropList, third)

        # Dropping for the last convolutional layer (has to be flattened for FC
        # layer)

        # indices of the corresponding dimension that the weight matrix is reduced (
        # ie for convolutional layers it is the number of filters, for FC
        # layers its the number of output neurons
        first = weights[idxConvFC].ndim - 1
        second = weights[idxConvFC + 1].ndim - 1
        third = weights[idxConvFC + 2].ndim - 2

        shape = weights[idxConvFC].shape

        numToDrop = shape[first] - int(p * shape[first])
        if len(self.unchagedWeights[idxConvFC]) >= numToDrop:
            listFC = sorted(
                random.sample(
                    self.unchagedWeights[idxConvFC],
                    numToDrop))
        else:
            fullList = [x for x in range(shape[first])]
            for x in self.unchagedWeights[idxConvFC]:
                fullList.remove(x)
            listFC = random.sample(fullList, numToDrop -
                                   len(self.unchagedWeights[idxConvFC]))
            listFC.extend(self.unchagedWeights[idxConvFC])
            listFC.sort()
        self.prevDropWeights[idxConvFC] = listFC.copy()
        print(
            "Dropped weights idx ",
            idxConvFC,
            ": ",
            (self.prevDropWeights[idxConvFC]))

        # Extend drop list for the input of first fully connected layer
        FClist = []
        for drop in listFC:
            FClist.extend(range(drop * 4 * 4, drop * 4 * 4 + 4 * 4))

        # drop the neurons from each weight parameter ( # filters for conv
        # layer, bias, # of inputs for first FC layer)
        self.droppedWeights[cid][idxConvFC][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 1][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 2][1] = FClist.copy()

        weights[idxConvFC] = np.delete(weights[idxConvFC], listFC, first)
        weights[idxConvFC +
                1] = np.delete(weights[idxConvFC +
                                       1], listFC, second)
        weights[idxConvFC +
                2] = np.delete(weights[idxConvFC +
                                       2], FClist, third)

        for i in range(len(weights)):
            self.droppedWeightsShape[cid][i] = weights[i].shape

        # flatten the weight parameters to 1D before sending to Android Clients
        weights[0] = weights[0].flatten()
        weights[2] = weights[2].flatten()
        weights[4] = weights[4].flatten()
        weights[6] = weights[6].flatten()
        weights[8] = weights[8].flatten()
        weights[10] = weights[10].flatten()
        weights[12] = weights[12].flatten()
        weights[14] = weights[14].flatten()
        weights[16] = weights[16].flatten()

        return self.weights_to_parameters(weights)

    def find_stable(self,
                    parameters: Parameters,
                    results: List[Tuple[Weights,
                                        int]],
                    idxList: List[int],
                    idxConvFC: int):
        """Find the invariant neurons that are under the update threshold"""
        # Note: for each layer in the model, it has related weight parameters in 3 indices:
        #       activation, bias, and input of next layer

        # Args: parameters: global model parameters
        #          results: updated model of each client
        #          idxList: list for the starting indices of each layer
        # idxConvFC: the index of the last convolutional layer before the FC
        # layer

        weights = self.parameters_to_weights(parameters)
        weights[0] = np.reshape(weights[0], self.weight_shapes[0])
        weights[2] = np.reshape(weights[2], self.weight_shapes[2])
        weights[4] = np.reshape(weights[4], self.weight_shapes[4])
        weights[6] = np.reshape(weights[6], self.weight_shapes[6])
        weights[8] = np.reshape(weights[8], self.weight_shapes[8])
        weights[10] = np.reshape(weights[10], self.weight_shapes[10])
        weights[12] = np.reshape(weights[12], self.weight_shapes[12])
        weights[14] = np.reshape(weights[14], self.weight_shapes[14])
        weights[16] = np.reshape(weights[16], self.weight_shapes[16])

        # (Optional) once a neuron is added to the list, keep it in there
        list = [[] for x in range(len(weights))]
        # list[0] = self.unchagedWeights[0]
        # list[2] = self.unchagedWeights[2]
        # list[4] = self.unchagedWeights[4]
        # list[6] = self.unchagedWeights[6]
        # list[8] = self.unchagedWeights[8]

        # Since each layer has its own threshold, we check if parameters fall
        # under the threshold individually
        for idx in idxList:
            difference = []
            for i in range(len(weights)):
                difference.append(np.full(self.weight_shapes[i], 0))

            for cWeights, num_examples, cid in results:
                if cid in self.straggler:
                    continue
                clientWeights = cWeights
                clientWeights[0] = np.reshape(
                    clientWeights[0], self.weight_shapes[0])
                clientWeights[2] = np.reshape(
                    clientWeights[2], self.weight_shapes[2])
                clientWeights[4] = np.reshape(
                    clientWeights[4], self.weight_shapes[4])
                clientWeights[6] = np.reshape(
                    clientWeights[6], self.weight_shapes[6])
                clientWeights[8] = np.reshape(
                    clientWeights[8], self.weight_shapes[8])
                clientWeights[10] = np.reshape(
                    clientWeights[10], self.weight_shapes[10])
                clientWeights[12] = np.reshape(
                    clientWeights[12], self.weight_shapes[12])
                clientWeights[14] = np.reshape(
                    clientWeights[14], self.weight_shapes[14])
                clientWeights[16] = np.reshape(
                    clientWeights[16], self.weight_shapes[16])

                for i in range(len(weights)):
                    difference[i] += (np.abs(clientWeights[i] - weights[i]) <= np.abs(
                        self.changeThreshold[idx] * weights[i])) * 1

            # We treat weight parameter as "Invariant" only if its an
            # "invariant" weight parameter for at least 75% of the
            # non-straggler clients
            for i in range(len(difference)):
                difference[i] = difference[i] >= (
                    0.75 * (len(results) - len(self.straggler)))

            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(weights[idx].ndim - 1)

            # perform reduction for all other dimensions (so we know which idx
            # has constant weights
            idx0Layer = np.all(difference[idx], axis=tuple(dim))
            idx1Layer = difference[idx + 1]

            dim = [x for x in range(weights[idx + 2].ndim)]
            dim.remove(weights[idx + 2].ndim - 2)
            idx2Layer = np.all(difference[idx + 2], axis=tuple(dim))

            # Calculate which neuron is "invariant" for all related weight
            # parameters related to that neuron
            noChangeIdx = idx0Layer & idx1Layer & idx2Layer

            # set unchanged list
            for i in range(len(noChangeIdx)):
                if noChangeIdx[i]:
                    if i not in list[idx]:
                        list[idx].append(i)
            # print("unchanged idx ", idx, ": ", list[idx])

            # Check whcih neurons were dropped last round and is still in the
            # unchangedlist
            self.defDropWeights[idx] = []
            if len(self.prevDropWeights[idx]) > 0:
                for i in self.prevDropWeights[idx]:
                    if i in list[idx]:
                        self.defDropWeights[idx].append(i)
            # print("def drop idx ", idx, ": ", self.defDropWeights[idx])

        # Repeat process for last convolutional layer
        dim = [x for x in range(weights[idxConvFC].ndim)]
        dim.remove(weights[idxConvFC].ndim - 1)
        idx0Layer = np.all(difference[idxConvFC], axis=tuple(dim))

        idx1Layer = difference[idxConvFC + 1]

        dim = [x for x in range(weights[idxConvFC + 2].ndim)]
        dim.remove(weights[idx + 2].ndim - 2)
        idx2Layer = np.all(difference[idxConvFC + 2], axis=tuple(dim))

        reducedList = np.array([])
        for i in range(len(idx0Layer)):
            # "unflatten" the first FC layer
            reducedList = np.array(np.append(reducedList, np.all(
                idx2Layer[i * 4 * 4: i * 4 * 4 + 4 * 4])), dtype=bool)

        noChangeIdx = idx0Layer & idx1Layer & reducedList

        for i in range(len(noChangeIdx)):
            if noChangeIdx[i]:
                list[idxConvFC].append(i)
        # print("unchanged idx ", idxConvFC, ": ", list[idxConvFC])

        self.defDropWeights[idxConvFC] = []
        if len(self.prevDropWeights[idxConvFC]) > 0:
            for i in self.prevDropWeights[idxConvFC]:
                if i in list[idxConvFC]:
                    self.defDropWeights[idxConvFC].append(i)
        # print("def drop idx ", idxConvFC, ": ", self.defDropWeights[idxConvFC])

        self.unchagedWeights = list

        return list

    def find_min(self,
                 parameters: Parameters,
                 results: List[Tuple[Weights,
                                     int]],
                 idxList: List[int],
                 rnd: int):
        """Find the Minimum percent change for each layer of the model this round"""
        # Note: for each layer in the model, it has related weight parameters in 3 indices:
        #       activation, bias, and input of next layer

        # Args: parameters: global model parameters
        #          results: updated model of each client
        #          idxList: list for the starting indices of each layer
        #          rnd: Current round of training

        weights = self.parameters_to_weights(parameters)
        weights[0] = np.reshape(weights[0], self.weight_shapes[0])
        weights[2] = np.reshape(weights[2], self.weight_shapes[2])
        weights[4] = np.reshape(weights[4], self.weight_shapes[4])
        weights[6] = np.reshape(weights[6], self.weight_shapes[6])
        weights[8] = np.reshape(weights[8], self.weight_shapes[8])
        weights[10] = np.reshape(weights[10], self.weight_shapes[10])
        weights[12] = np.reshape(weights[12], self.weight_shapes[12])
        weights[14] = np.reshape(weights[14], self.weight_shapes[14])
        weights[16] = np.reshape(weights[16], self.weight_shapes[16])

        difference = []
        for i in range(len(weights)):
            difference.append(np.full(self.weight_shapes[i], 0.0))

        # Calculate the maximum change of each weight parameter for all clients
        for cWeights, num_examples, cid in results:
            if cid in self.straggler:
                continue
            clientWeights = cWeights
            clientWeights[0] = np.reshape(
                clientWeights[0], self.weight_shapes[0])
            clientWeights[2] = np.reshape(
                clientWeights[2], self.weight_shapes[2])
            clientWeights[4] = np.reshape(
                clientWeights[4], self.weight_shapes[4])
            clientWeights[6] = np.reshape(
                clientWeights[6], self.weight_shapes[6])
            clientWeights[8] = np.reshape(
                clientWeights[8], self.weight_shapes[8])
            clientWeights[10] = np.reshape(
                clientWeights[10], self.weight_shapes[10])
            clientWeights[12] = np.reshape(
                clientWeights[12], self.weight_shapes[12])
            clientWeights[14] = np.reshape(
                clientWeights[14], self.weight_shapes[14])
            clientWeights[16] = np.reshape(
                clientWeights[16], self.weight_shapes[16])

            for i in range(len(weights)):
                difference[i] = np.maximum(
                    difference[i],
                    (np.abs(
                        clientWeights[i] -
                        weights[i])) /
                    np.abs(
                        weights[i]))

        # For each layer, calculate the % change of each neuron based on the
        # maximum % change of its related weight parameters
        for idx in idxList:
            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(weights[idx].ndim - 1)

            # perform reduction for all other dimensions (so we know which idx
            # has constant weights
            idx0Layer = np.amax(difference[idx], axis=tuple(dim))

            idx1Layer = difference[idx + 1]

            dim = [x for x in range(weights[idx + 2].ndim)]
            dim.remove(weights[idx + 2].ndim - 2)
            idx2Layer = np.amax(difference[idx + 2], axis=tuple(dim))
            sum = np.maximum(np.maximum(idx0Layer, idx1Layer), idx2Layer)

            noChangeIdx = np.argsort(sum)

            print("% difference: ", sum[noChangeIdx[0]])

            # Take average min values of neurons as initial update Threshold
            # value
            if (rnd == 2):
                self.changeThreshold[idx] = sum[noChangeIdx[0]]
                print(
                    "threshold for ",
                    idx,
                    "updated to: ",
                    self.changeThreshold[idx])
            if (rnd == 3):
                self.changeThreshold[idx] = (
                    self.changeThreshold[idx] + sum[noChangeIdx[0]]) / 2
                print(
                    "threshold for ",
                    idx,
                    "updated to: ",
                    self.changeThreshold[idx])

    def aggregate_drop(self,
                       results: List[Tuple[Weights,
                                           int,
                                           str]],
                       dropWeights: Dict[str,
                                         List],
                       origWeightsShape: List,
                       dropWeightsShape: Dict[str,
                                              List]) -> Weights:
        """Compute weighted average for a federated drop technique """

        # initialize list to keep track of the total number of examples used during training for each neuron
        # since we are dropping neurons from the model for some clients, so the num examples that each neuron
        # trained on will be different
        num_examples_total = sum(
            [num_examples for _, num_examples, _ in results])
        total_examples_wDrop = []
        for i in range(len(origWeightsShape)):
            total_examples_wDrop.append(
                np.full(
                    origWeightsShape[i],
                    num_examples_total))

        # transform the list of weights into original format
        # We will expand sub-models to the global model shape by filling in 0s
        # for dropped weights
        transformedResults = []
        for (clientWeights, num_examples, cid) in results:
            layer = 0
            transformedWeights = clientWeights

            # no transformation needed if not a straggler
            if cid not in dropWeights:
                for i in range(len(origWeightsShape)):
                    # since Android clients send weights in 1 Dimension, we
                    # reshape it for easier calculation
                    if transformedWeights[i].shape != origWeightsShape[i]:
                        transformedWeights[i] = np.reshape(
                            transformedWeights[i], origWeightsShape[i])
                transformedResults.append((transformedWeights, num_examples))
                continue

            # client was a straggler:
            for i in range(len(origWeightsShape)):
                # since Android clients send weights in 1 Dimension, we reshape
                # it for easier calculation
                if transformedWeights[i].shape != dropWeightsShape[cid][i]:
                    transformedWeights[i] = np.reshape(
                        transformedWeights[i], dropWeightsShape[cid][i])

            # transform sub-model to global model shape for all layers
            # (row refers to layer's output dimension, while col is the layer's input dimension))

            for [row, col] in dropWeights[cid]:
                transformedWeights[layer] = clientWeights[layer]

                colLen = len(col)
                rowLen = len(row)

                # for each row that's dropped add a row in the weight parameter
                # with all 0s
                if (rowLen != 0):
                    transformedWeights[layer] = np.insert(
                        transformedWeights[layer],
                        row -
                        np.arange(
                            len(row)),
                        0,
                        axis=transformedWeights[layer].ndim -
                        1)
                    # since the row was dropped, the neuron did not train with this client's data
                    # Hence remove client's data count from total examples
                    # trained for related weights.
                    total_examples_wDrop[layer][..., row] -= num_examples

                # for each row that's dropped add a row in the weight parameter
                # with all 0s
                if (colLen != 0):
                    transformedWeights[layer] = np.insert(
                        transformedWeights[layer],
                        col -
                        np.arange(
                            len(col)),
                        0,
                        axis=transformedWeights[layer].ndim -
                        2)
                    # since the colum was dropped, the neuron did not train with this client's data
                    # Hence remove client's data count from total examples
                    # trained for related weights.
                    total_examples_wDrop[layer][..., col, :] -= num_examples

                    # Check if any number of examples for any weights were
                    # subtracted twice if both its row and col was dropped.
                    k = [range(total_examples_wDrop[layer].shape[i]) for i in
                         range(total_examples_wDrop[layer].ndim - 2)]
                    k.append(col)
                    k.append(row)

                    total_examples_wDrop[layer][np.ix_(*k)] += num_examples
                layer += 1

            # Append the transformed client model to the result list, with
            # number of examples that each individual weight trained on.
            transformedResults.append((transformedWeights, num_examples))

        # Create a list of weights, each multiplied by the related number of
        # examples
        weighted_weights = [[layer * num_examples for layer in weights]
                            for weights, num_examples in transformedResults]

        # Compute average weights of each layer
        weights_prime: Weights = [
            np.divide(reduce(np.add, layer_updates), total_examples_wDrop[i])
            for i, layer_updates in enumerate(zip(*weighted_weights))
        ]

        for i in range(len(origWeightsShape)):
            weights_prime[i] = np.float32(weights_prime[i].flatten())

        return weights_prime

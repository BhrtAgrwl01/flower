import flwr as fl
import sys

# Start Flower server for three rounds of federated learning
# This requires 3 clients to be trained federatedly.

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=3,
        min_eval_clients=3,
        min_available_clients=3,
    )
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 10}, strategy=strategy)
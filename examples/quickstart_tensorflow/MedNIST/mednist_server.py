import flwr as fl
import sys

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    for i in range(int(sys.argv[1])):
        fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 10})
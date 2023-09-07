import argparse

import math
import torch
import torchcde
from tqdm import tqdm

from dataset import get_data
from model import CDEFunc, NeuralCDE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(num_epochs=30):
    train_X, train_y = get_data()
    if args.model == "cde":
        model = NeuralCDE(input_channels=3, hidden_channels=8, output_channels=1).to(
            device
        )

        train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            train_X
        )

        train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

    elif args.model == "ode":
        model = torch.nn.ODEAdjoint(
            CDEFunc(input_channels=3, hidden_channels=8),
            t=torch.linspace(0.0, 4 * math.pi, 100),
        ).to(device)

        train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

    else:
        model = torch.nn.RNN(input_size=3, hidden_size=8, batch_first=True).to(device)

        train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader, desc="Epoch {}".format(epoch)):
            batch_coeffs, batch_y = batch
            batch_coeffs = batch_coeffs.to(device)
            batch_y = batch_y.to(device)
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch: {}   Training loss: {}".format(epoch, loss.item()))

    test_X, test_y = get_data()
    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)
    pred_y = model(test_coeffs.to(device)).squeeze(-1)
    test_y = test_y.to(device)
    binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
    prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
    proportion_correct = prediction_matches.sum() / test_y.size(0)
    print("Test Accuracy: {}".format(proportion_correct))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        choices=["cde", "rnn", "ode"],
        default="cde",
        help="Model to use",
    )

    args = parser.parse_args()
    main()

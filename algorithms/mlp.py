import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class MLP:
    def __init__(self, params, input_dim, output_dim, **kwargs):
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.out_dtype = torch.float32
        self.loss_fn = F.mse_loss

        params = copy.deepcopy(self.params)
        params["sizes"] = [input_dim] + params["sizes"] + [output_dim]
        self.model = get_mlp(**params).to(self.device)

        self.mu = None
        self.sigma = None

    def preprocess(self, X):
        assert self.mu is not None
        mu = self.mu.reshape((1, X.shape[1]))
        sigma = self.sigma.reshape((1, X.shape[1]))

        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = mu[0, i]

        return np.clip((X - mu) / (sigma + 1e-8), -10, 10)

    def set_weights(self, weights):
        self.model.load_state_dict(weights["model"])
        self.mu = weights["mu"]
        self.sigma = weights["sigma"]

    def get_weights(self):
        return {"model": self.model.state_dict(),
                "mu": self.mu,
                "sigma": self.sigma}

    def predict(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)

        self.model.eval()
        X = torch.tensor(X).float().to(self.device)
        with torch.no_grad():
            pred = self.model(X).squeeze(1)
        return pred.cpu().numpy()

    def eval(self, X, y, preprocess=False):
        if preprocess:
            X = self.preprocess(X)

        if X is not None:
            self.model.eval()
            with torch.no_grad():
                X = torch.tensor(X).float().to(self.device)
                y = torch.tensor(y).to(self.out_dtype).to(self.device)
                out = self.model(X)
                eval_loss = self.loss_fn(out.squeeze(1), y)
                eval_loss = eval_loss.item()
            return -eval_loss

    def train(self, X_train, y_train, X_eval=None, y_eval=None, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma
        X_train = self.preprocess(X_train)
        if X_eval is not None:
            X_eval = self.preprocess(X_eval)

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params.get("lr", 0.001))

        params = self.params
        epoch = params.get("epochs", 20)
        batch_size = params.get("batch_size", 2048)
        eval = None
        for epoch in range(epoch):
            inds = np.arange(X_train.shape[0])
            np.random.shuffle(inds)

            info = dict(loss=0, count=0)

            self.model.train()
            for i in range(0, len(inds), batch_size):
                batch_inds = inds[i:i+batch_size]
                X = torch.tensor(X_train[batch_inds]).float().to(self.device)
                y = torch.tensor(y_train[batch_inds]).to(self.device)
                y = y.to(self.out_dtype)

                optimizer.zero_grad()
                # Forward pass
                y_pred = self.model(X)
                # Compute Loss
                loss = self.loss_fn(y_pred.squeeze(1), y)

                # Backward pass
                loss.backward()
                optimizer.step()

                info["loss"] += loss.item()
                info["count"] += 1

            info["loss"] = info["loss"] / info["count"]
            del info["count"]

            if X_eval is not None:
                eval_score = self.eval(X_eval, y_eval)
                info["eval_score"] = eval_score
                if eval is None or eval_score >= eval["score"]:
                    eval = dict(score=eval_score, epoch=epoch)

                if eval is not None:
                    if params.get('save_best_eval') and epoch == eval["epoch"]:
                        eval["weights"] = self.model.state_dict()

                    elif params.get("patience") and (epoch - eval["epoch"]) > params["patience"]:
                        break

            print(f"\nEpoch.{epoch}")
            for k, v in info.items():
                print(k, ':', v)

        if eval is not None and "weights" in eval:
            print("Saved weights of epoch:",
                  eval["epoch"], 'with evaluation score:', eval["score"])
            self.model.load_state_dict(eval["weights"])

        if eval is not None and "score" in eval:
            return eval["score"]


def get_mlp(sizes,
            dropout=None,
            first_batch_norm=False,
            use_swish=False,
            use_batch_norm=False,
            **kwargs):
    layers = []
    init_size = sizes[0]

    if first_batch_norm:
        layers.append(nn.BatchNorm1d(init_size))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

    for size in sizes[1:-1]:
        layers.append(nn.Linear(init_size, size))
        torch.nn.init.xavier_uniform_(layers[-1].weight)
        torch.nn.init.zeros_(layers[-1].bias)

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(init_size))

        layers.append(nn.SiLU() if use_swish else nn.ReLU())

        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        init_size = size

    layers.append(nn.Linear(init_size, sizes[-1]))
    torch.nn.init.xavier_uniform_(layers[-1].weight)
    torch.nn.init.zeros_(layers[-1].bias)

    model = nn.Sequential(*layers)
    return model


if __name__ == "__main__":
    config = dict(model_params=dict(sizes=[10, 15, 30, 15, 10]),)
    mlp = MLP(config)
    print(mlp.model)
    state = mlp.save()
    mlp.load(state)

    X = np.zeros((10, 10))
    out = mlp.predict(X)
    print(out.shape)



import torch
from torch import nn

from torchdiffeq import odeint

def L96_eq1_xdot(t, X, F = 8, advect=True):
    """
    Calculate the time rate of change for the X variables for the Lorenz '96, equation 1:
        d/dt X[k] = -X[k-2] X[k-1] + X[k-1] X[k+1] - X[k] + F

    Args:
        X : Values of X variables at the current time step
        F : Forcing term
    Returns:
        dXdt : Array of X time tendencies
    """

    K = len(X)
    Xdot = torch.zeros(K)

    if advect:
        Xdot = torch.roll(X, 1) * (torch.roll(X, -1) - torch.roll(X, 2)) - X + F
    else:
        Xdot = -X + F
    #     for k in range(K):
    #         Xdot[k] = ( X[(k+1)%K] - X[k-2] ) * X[k-1] - X[k] + F
    return Xdot

def estimate_log_likelihood(model, particles, y_obs, dt, noise_scale, model_noise, localize = False, localization_matrix = None, reg = 1e-5):
    nt = y_obs.shape[0]
    log_est = torch.tensor([0.0])
    n_particles = particles.shape[0]
    K = particles.shape[1]
    for i in range(nt):
        # propagate particles one step forward
        pred = odeint(L96_eq1_xdot, particles, torch.tensor([0.0, dt]), method='rk4')[-1] + model(particles) + torch.normal(mean=0, std=model_noise, size = (n_particles, K))

        #Compute the empirical covariance matrix
        cov = torch.cov(pred.T)
        mean = torch.mean(pred, axis=0)
        
        if localize and localization_matrix is not None:
            cov = localization_matrix * cov
        y = y_obs[i]
        
        # Draw observational noise and correct the particles
        noise = torch.normal(mean = 0.0, std=noise_scale, size=particles.shape)
        increment = y - noise - pred
        correction = cov@torch.linalg.solve(cov + (noise_scale**2 + reg)*torch.eye(K), increment.T)
        
        particles = pred + correction.T

        # Compute the log-likelihood
        diff = y - mean
        log_est += -0.5 * torch.linalg.solve(cov + (noise_scale**2)*torch.eye(K), diff).T@diff
    return log_est/nt/K


def train_model(model, X, optimizer, n_epochs, train_length, model_noise, noise_scale, dt, N_particles = 50, localize = False, localization_matrix = None, verbose = True):
    n_seq = X.shape[0] // train_length
    K = X.shape[1]
    loss_hist = []
    for epoch in range(n_epochs):
        for i in range(n_seq):
            
            # Extract a sub-sequence
            X_seq = X[i*train_length: (i+1)*train_length]
            # Define the initial condition
            particles = X_seq[0] + model_noise * torch.normal(mean=0, std=1.0, size = (N_particles, K))
            # Define the observations
            y_seq = X_seq[1:]

            
            optimizer.zero_grad()
            loss = -estimate_log_likelihood(model, particles, y_seq, dt, noise_scale, model_noise, localize, localization_matrix)
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())
            if verbose:
                print(f"Epoch {epoch}, Iteration {i}, Loss {loss.item()}")

    return model, loss_hist


def filter(model, x_0, Y, n_iter,model_noise, observation_noise, localize = False, localization_matrix = None, reg = 1e-5):
    n_iter = Y.shape[0]

    prediction = []
    analysis = []
    correction = []

    N_particles = x_0.shape[0]
    K = x_0.shape[1]
    for i in range(n_iter):
        pred = model(x_0) + torch.normal(mean=0, std=model_noise, size = (N_particles, K))

        #Compute the empirical covariance matrix
        cov = torch.cov(pred.T)

        if localize:
            cov = localization_matrix * cov

        y = Y[i]
        # Draw observational noise and correct the particles

        noise = torch.normal(mean = 0.0, std=observation_noise, size=pred.shape)
        increment = y - noise - pred
        corr = cov@torch.linalg.solve(cov + (observation_noise**2 + reg)*torch.eye(K), increment.T)

        x_0 = pred + corr.T

        prediction.append(pred)
        analysis.append(x_0)
        correction.append(corr)
    
    prediction = torch.stack(prediction)
    analysis = torch.stack(analysis)
    correction = torch.stack(correction)

    prediction = prediction.detach().numpy()
    analysis = analysis.detach().numpy()
    correction = correction.detach().numpy()

    return prediction, analysis, correction

    


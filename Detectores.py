"""  El pulso puede generarse mediante la función puede modelarse con la función

    $S(t) = A e^{-(t-t_0)^2/2 \sigma^2} \, \cos\left[\omega (t-t_0 + \sigma) \right]$

    donde A se elige para que el área sea 1.
""" 
import numpy as np

def pulso(t, t0, sigma, omega):
    gauss = np.exp(-(t - t0)**2 / (2 * sigma**2))
    seno = np.cos(omega * (t - t0 + sigma))
    S = gauss * seno
    # Normalizamos para que el área sea 1
    A = 1 / np.trapz(S, t)
    return A * S

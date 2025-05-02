"""  El pulso puede generarse mediante la función puede modelarse con la función

    $S(t) = A e^{-(t-t_0)^2/2 \sigma^2} \, \cos\left[\omega (t-t_0 + \sigma) \right]$

    donde A se elige para que el área sea 1.
""" 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import betaprime
from scipy.interpolate import interp1d
from scipy.optimize import leastsq

# Pulso normalizado
def pulso(t, t0, sigma, omega):
    gauss = np.exp(-(t - t0)**2 / (2 * sigma**2))
    seno = np.cos(omega * (t - t0 + sigma))
    S = gauss * seno
    A = 1 / np.trapz(S, t)
    return A * S

# Simulación de una medición
def simular_medicion(t, t0s, probs, sigma0=0.05, sigma_sigma=0.005, omega0=5*np.pi, sigma_omega=0.05):
    señal = np.zeros_like(t)
    for t0, p in zip(t0s, probs):
        if np.random.rand() < p:
            x = betaprime.rvs(a=4, b=2)
            t_centro = t0 + (x - 4 / 2) * 0.15
            sigma = np.random.normal(sigma0, sigma_sigma)
            omega = np.random.normal(omega0, sigma_omega)
            señal += pulso(t, t_centro, sigma, omega)
    ruido_rel = np.random.normal(1, 0.1, size=t.shape)
    ruido_abs = np.random.normal(0, 0.2, size=t.shape)
    fondo = (-t + 2) / 100
    return señal * ruido_rel + ruido_abs + fondo

# Simulación de N mediciones
def simular_N_mediciones(N, t, t0s, probs):
    return np.array([simular_medicion(t, t0s, probs) for _ in range(N)])

# Cálculo del FWHM (ancho a mitad de altura)
def calcular_fwhm(t, y):
    y = y / np.max(y)
    half_max = 0.5
    indices = np.where(y >= half_max)[0]
    if len(indices) >= 2:
        fwhm = t[indices[-1]] - t[indices[0]]
        return fwhm
    return 0

# Ajuste por mínimos cuadrados
def ajustar_area(t, señal_base, promedio, t0, ventana=0.5):
    interp_base = interp1d(t, señal_base, bounds_error=False, fill_value=0)
    mask = (t > t0 - ventana/2) & (t < t0 + ventana/2)
    t_fit = t[mask]
    y_data = promedio[mask]

    def modelo(params):
        A, shift = params
        return A * interp_base(t_fit - shift)

    def error(params):
        return y_data - modelo(params)

    p0 = [1, 0]
    resultado = leastsq(error, p0)[0]
    A, shift = resultado
    return A

# Simulación para los tres casos
t = np.linspace(0, 8, 4000)
casos = [
    ([1.8, 2.8, 5.0], [1.0, 0.0, 0.0]),
    ([1.8, 2.8, 5.0], [0.8, 0.1, 0.1]),
    ([1.8, 2.8, 3.5], [0.85, 0.1, 0.05])
]

promedios = []
for t0s, probs in casos:
    mediciones = simular_N_mediciones(10000, t, t0s, probs)
    promedio = np.mean(mediciones, axis=0)
    promedios.append((t0s, promedio))

# Señal base para ajuste (caso 1, primer pico)
base_t0s, base_prom = promedios[0]
base_modelo = base_prom

# Ajuste para los tres casos
areas_estimadas = []
for t0s, promedio in promedios:
    areas = []
    for t0 in t0s:
        area = ajustar_area(t, base_modelo, promedio, t0)
        areas.append(area)
    areas_estimadas.append((t0s, areas))

# Cálculo de máximos y FWHM para graficar
resultados_fwhm = []
for t0s, promedio in promedios:
    datos = []
    for t0 in t0s:
        idx = np.abs(t - t0).argmin()
        t_range = (t > t0 - 0.2) & (t < t0 + 0.2)
        t_local = t[t_range]
        y_local = promedio[t_range]
        max_t = t_local[np.argmax(y_local)]
        fwhm = calcular_fwhm(t_local, y_local)
        datos.append((max_t, fwhm))
    resultados_fwhm.append((t0s, datos))

import pandas as pd
df_resultados = pd.DataFrame({
    "Caso": ["Caso 1", "Caso 2", "Caso 3"],
    "Picos t0": [str(c[0]) for c in casos],
    "Áreas estimadas": [a[1] for a in areas_estimadas],
    "Máximos y FWHM": [r[1] for r in resultados_fwhm]
})

import ace_tools as tools; tools.display_dataframe_to_user(name="Resultados de la Simulación", dataframe=df_resultados)


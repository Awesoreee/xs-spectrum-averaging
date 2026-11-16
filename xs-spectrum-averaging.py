import pandas as pd
import numpy as np 
import scipy as sc

def composite_reactor_spectrum(
    E: np.ndarray,
    temperature_k: float = 293.6,
    beta: float = 0.0,
    gamma: float = 0.0,
    watt_a: float = 0.988,
    watt_b: float = 2.249
) -> np.ndarray:
    """
    Универсальный спектр нейтронов φ(E) [1/эВ] во всём диапазоне энергий:
    Тепловой (Максвелл) + Промежуточный (1/E) + Быстрый (Уатт).
    По умолчанию (beta=0, gamma=0) возвращает чистый тепловой спектр Максвелла.
    """
    E = np.asarray(E, dtype=float)
    phi = np.zeros_like(E)
    mask = E > 0
    e_pos = E[mask]

    # 1. Тепловая компонента (Максвелл)
    kT = 8.617333262e-5 * temperature_k  # эВ
    # Используем безопасное вычисление экспоненты без переполнения при больших E
    phi_maxwell = (e_pos / kT**2) * np.exp(-np.minimum(e_pos / kT, 700.0))

    # 2. Промежуточная компонента (1/E с плавным обрезанием на краях)
    if beta > 0:
        cutoff_low = 1.0 / (1.0 + (5.0 * kT / e_pos) ** 4)
        cutoff_high = np.exp(-e_pos / 1.0e6)  # спад выше 1 МэВ
        phi_fermi = (1.0 / e_pos) * cutoff_low * cutoff_high
    else:
        phi_fermi = 0.0

    # 3. Быстрая компонента (спектр деления Уатта)
    if gamma > 0:
        e_mev = e_pos * 1.0e-6
        # Нормировочный множитель спектра Уатта (в 1/МэВ -> переводим в 1/эВ через 1e-6)
        norm_watt = np.sqrt(np.pi * watt_b * watt_a / 4.0) * watt_a * np.exp(watt_b * watt_a / 4.0)
        phi_watt = (1.0e-6 / norm_watt) * np.exp(-e_mev / watt_a) * np.sinh(np.sqrt(watt_b * e_mev))
    else:
        phi_watt = 0.0

    phi[mask] = phi_maxwell + beta * phi_fermi + gamma * phi_watt
    return phi

# Импорт данных
df = pd.read_csv(
    'table.csv',
    sep=r"\s*;\s*",
    engine="python",
    skiprows=[1, 2]
    )
df = df.dropna(axis=1, how="all")

energy_col = df.columns[0]
reaction_cols = df.columns[1:]

# Основные расчёты
E = df[energy_col].to_numpy(dtype=float)
phi = composite_reactor_spectrum(E)
norm = np.trapezoid(phi, E)

sigmas = df[reaction_cols].to_numpy(dtype=float)
sigma_average = np.trapezoid(sigmas * phi[:, None], E, axis=0) / norm
average_maxwell_from_janis = pd.Series(sigma_average, index=reaction_cols)

print(average_maxwell_from_janis)
import numpy as np
import pandas as pd
from typing import Dict
from scipy.interpolate import griddata

class VolatilitySurfaceAnalyzer:
    def __init__(self):
        self.surface = None
        
    def calculate_volatility_surface(self, strikes: np.ndarray, expirations: np.ndarray, volatilities: np.ndarray) -> pd.DataFrame:
        """변동성 표면 계산"""
        grid_x, grid_y = np.meshgrid(
            np.linspace(strikes.min(), strikes.max(), 50),
            np.linspace(expirations.min(), expirations.max(), 50)
        )
        grid_z = griddata(
            points=(strikes, expirations),
            values=volatilities,
            xi=(grid_x, grid_y),
            method='cubic'
        )
        self.surface = pd.DataFrame(grid_z, index=grid_y[:, 0], columns=grid_x[0, :])
        return self.surface

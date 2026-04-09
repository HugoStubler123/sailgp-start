"""
F50 catamaran polar model.
Boat speed as a function of TWS (True Wind Speed) and TWA (True Wind Angle).
All speeds in km/h, angles in degrees.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# Default F50 polar table
DEFAULT_TWS = np.array([8, 12, 16, 20, 24, 28, 32], dtype=float)
DEFAULT_TWA = np.array([35, 40, 45, 50, 60, 75, 90, 110, 135, 150, 180], dtype=float)

# Boat speed (km/h) indexed by [TWS_row][TWA_col]
DEFAULT_SPEEDS = np.array([
    #  35   40   45   50   60   75   90  110  135  150  180
    [  5,   8,  11,  13,  16,  18,  20,  22,  24,  22,  18],  # TWS  8
    [  8,  14,  18,  22,  27,  32,  36,  40,  42,  38,  30],  # TWS 12
    [ 12,  20,  28,  33,  38,  44,  50,  54,  52,  48,  38],  # TWS 16
    [ 15,  26,  35,  40,  46,  52,  58,  62,  60,  54,  44],  # TWS 20
    [ 16,  30,  40,  46,  52,  58,  64,  68,  66,  60,  48],  # TWS 24
    [ 16,  32,  42,  50,  56,  62,  68,  72,  70,  64,  52],  # TWS 28
    [ 16,  33,  44,  52,  58,  64,  70,  74,  72,  66,  54],  # TWS 32
], dtype=float)


class Polar:
    """Yacht performance polar with 2D interpolation (TWS x TWA -> boat speed)."""

    def __init__(self, tws_values=None, twa_values=None, speed_table=None):
        self.tws_values = np.asarray(
            tws_values if tws_values is not None else DEFAULT_TWS, dtype=float
        )
        self.twa_values = np.asarray(
            twa_values if twa_values is not None else DEFAULT_TWA, dtype=float
        )
        self.speed_table = np.asarray(
            speed_table if speed_table is not None else DEFAULT_SPEEDS, dtype=float
        )
        self._interp = RegularGridInterpolator(
            (self.tws_values, self.twa_values),
            self.speed_table,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def boat_speed(self, tws: float, twa: float) -> float:
        """Boat speed (km/h) at given TWS (km/h) and |TWA| (degrees)."""
        twa = float(np.clip(abs(twa), self.twa_values[0], self.twa_values[-1]))
        tws = float(np.clip(tws, self.tws_values[0], self.tws_values[-1]))
        return max(float(self._interp((tws, twa))), 0.1)

    def boat_speed_ms(self, tws: float, twa: float) -> float:
        """Boat speed in m/s."""
        return self.boat_speed(tws, twa) / 3.6

    def optimal_upwind_twa(self, tws: float) -> float:
        """TWA (degrees) that maximises upwind VMG."""
        best_twa, best_vmg = 45.0, 0.0
        for twa in np.arange(30, 80, 0.5):
            speed = self.boat_speed(tws, twa)
            vmg = speed * np.cos(np.radians(twa))
            if vmg > best_vmg:
                best_vmg = vmg
                best_twa = float(twa)
        return best_twa

    def vmg_upwind(self, tws: float) -> float:
        """Best upwind VMG (km/h)."""
        opt = self.optimal_upwind_twa(tws)
        return self.boat_speed(tws, opt) * np.cos(np.radians(opt))

    def vmg_upwind_ms(self, tws: float) -> float:
        """Best upwind VMG (m/s)."""
        return self.vmg_upwind(tws) / 3.6

    def get_polar_curve(self, tws: float):
        """Return (twa_array, speed_array) for a given TWS."""
        twa_range = np.arange(30, 181, 1.0)
        speeds = np.array([self.boat_speed(tws, t) for t in twa_range])
        return twa_range, speeds

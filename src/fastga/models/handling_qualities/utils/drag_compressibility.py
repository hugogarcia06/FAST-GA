#  This file is part of FAST-OAD : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022 ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
import math
from scipy.misc import derivative


def get_drag_compressibility(mach, mach_crit=0.6, max_delta_drag=0.03, mach_max_delta_drag=1.2):
    """
    Method to estimate the drag due to compressibility effects found in Snorri Gudmundsson - General Aviation Aircraft 
    Design (Equation 15-206)
    
    :param mach: mach number where we want to compute the compressibility effect.
    :param mach_crit: critical mach number for the aircraft.
    :param max_delta_drag: the maximum drag due to mach drag divergence.
    :param mach_max_delta_drag: mach number at which the maximum drag due to mach divergence occurs.
    """

    max_drag = max_delta_drag
    mach_max = mach_max_delta_drag
    M = mach

    A = (
            (math.atanh((max_drag - 0.0002) / max_drag) - math.atanh(0.0002 / max_drag - 1.0))
            / (mach_max - mach_crit)
    )
    B = math.atanh(0.0002 / max_drag - 1.0) - A * mach_crit
    delta_drag = max_drag / 2.0 * (1.0 + math.tanh(A * M + B))

    return delta_drag


def get_drag_mach_derivative(mach):
    """
    Computes de drag due to mach derivative from the function defined above.

    :param mach: mach number where the derivative wants to be computed.
    """

    dCD_dM = derivative(get_drag_compressibility, mach, dx=1e-6)

    return dCD_dM

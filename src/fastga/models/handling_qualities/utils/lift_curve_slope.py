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


def get_lift_curve_slope(cl_alpha, aspect_ratio, mach, sweep_50):
    """
    For conventional, straight tapered wings, with moderate sweep angles, the lift curve slope may be
    estimated from Equation 8.22 from Roskam - Aircraft Design Part VI: Preliminary Calculation of Aerodynamic Thrust and Power
    Characteristics. It may also apply to horizontal and vertical tails when the specific parameters are changed.

    :param cl_alpha: airfoil lift curve slope of the surface we are calculating.
    :param aspect_ratio: aspect ratio of the surface we are calculating.
    :param mach: Mach number of the desired flight condition.
    :param sweep_50: sweep angle at half the chord in radians.
    :return CL_alpha: lift curve slope of the surface.
    """
    beta = math.sqrt(1 - mach ** 2)
    # In some other parts of the FAST-OAD code, and also in some bibliography, the k value is approximated to 0.95

    k = cl_alpha / (2 * math.pi)
    CL_alpha = (2 * math.pi * aspect_ratio) / (
            2
            +
            math.sqrt(
                aspect_ratio ** 2 * beta ** 2 / k ** 2
                * (1 + (math.tan(sweep_50)) ** 2 / beta ** 2)
                + 4
            )
    )



    return CL_alpha

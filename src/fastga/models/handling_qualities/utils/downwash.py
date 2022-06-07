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

from fastga.models.handling_qualities.utils.lift_curve_slope import get_lift_curve_slope


def get_downwash(cl_alpha_wing, aspect_ratio, taper_ratio, span, l_H, h_H, sweep_25, mach, sweep_50):
    """
    Function used to compute the downwash seen by the horizontal tail due to the wing. Equation 8.45 from
    Roskam - Aircraft Design Part VI.

    :param cl_alpha_wing: wing airfoil lift curve slope
    :param aspect_ratio: wing aspect ratio
    :param taper_ratio: wing taper ratio
    :param span: wing span
    :param l_H: distance in the x-axis from the horizontal tail mac quarter-chord point to the wing mac quarter-chord
    point
    :param h_H: vertical distance from the horizontal tail mac quarter-chord point to the x-axis
    :param sweep_25: wing sweep angle of the quarter-chord point line in radians.
    :param mach: flight mach number
    :param sweep_50: wing sweep angle of the half-chord point line
    """

    deps_dalpha_0 = get_downwash_low_speed(aspect_ratio, taper_ratio, span, l_H, h_H, sweep_25)
    CL_alpha_W = get_lift_curve_slope(cl_alpha_wing, aspect_ratio, mach, sweep_50)
    CL_alpha_W0 = get_lift_curve_slope(cl_alpha_wing, aspect_ratio, 0, sweep_50)
    deps_dalpha = deps_dalpha_0 * CL_alpha_W / CL_alpha_W0

    return deps_dalpha


def get_downwash_low_speed(aspect_ratio, taper_ratio, span, l_H, h_H, sweep_25):
    """
    Function used to compute the downwash at low speed (M=0). Equation 8.45 from Roskam - Aircraft Design Part VI.

    :param aspect_ratio: wing aspect ratio
    :param taper_ratio: wing taper ratio
    :param span: wing span
    :param l_H: distance in the x-axis from the horizontal tail mac quarter-chord point to the wing mac quarter-chord
        point
    :param h_H: vertical distance from the horizontal tail mac quarter-chord point to the x-axis
    :param sweep_25: wing sweep angle of the quarter-chord point line
    """

    A = aspect_ratio
    taper = taper_ratio
    b = span

    K_A = 1 / A - 1 / (1 + A ** 1.7)
    K_lambda = (10 - 3 * taper) / 7
    K_H = (1 - h_H / b) / ((2 * l_H / b) ** (1 / 3))

    deps_dalpha_0 = 4.44 * (K_A * K_lambda * K_H * math.sqrt(math.cos(sweep_25))) ** 1.19

    return deps_dalpha_0

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

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2


def get_dihedral_effect_surface(
        CL, mach,
        span, dihedral, twist, aspect_ratio, taper_ratio, sweep_25, sweep_50,
        ave_fuse_cross_section, l_f, z_w):
    """
    Function to compute the dihedral effect of a certain surface (wing or horizontal tail). The expressions used were
    obtained from Roskam - Airplane Design Part VI.

    :param CL: lift coefficient at the reference flight condition for the surface being computed.
    :param mach: mach number at the reference flight conditions.
    :param span: surface span.
    :param dihedral: surface dihedral angle.
    :param twist: surface twist angle.
    :param aspect_ratio: surface aspect_ratio.
    :param taper_ratio: surface taper_ratio.
    :param sweep_25: quarter-chord point line sweep angle of the surface.
    :param sweep_50: mid-chord point line sweep angle of the surface.
    :param ave_fuse_cross_section: average fuselage cross-section area.
    :param l_f: distance from the fuselage tip to the leading edge of the root chord plus the wing projection on the
    x-axis.
    :param z_w: distance from wing root quarter chord point to the fuselage centerline, positive below fuselage
    centerline.
    """
    # Rolling moment due to sideslip derivative Cl_beta. Also called Dihedral Effect
    # From Figure 10.20
    Clbeta_CL_sweep50 = FigureDigitization2.get_Clbeta_CL_sweep50(sweep_50, aspect_ratio, taper_ratio)
    # From Figure 10.21
    k_compress_sweep = FigureDigitization2.get_k_mach_sweep(mach, aspect_ratio, sweep_50)
    # From Figure 10.22
    k_fuselage = FigureDigitization2.get_k_fuselage(l_f, span, aspect_ratio, sweep_50)
    # From Figure 10.23
    Clbeta_CL_AR = FigureDigitization2.get_Clbeta_CL_aspect_ratio(aspect_ratio, taper_ratio)
    # From Figure 10.24
    Clbeta_dihedral = FigureDigitization2.get_Clbeta_dihedral(aspect_ratio, sweep_50, taper_ratio)
    # From Figure 10.25
    k_compress_dihedral = FigureDigitization2.get_k_mach_dihedral(mach, aspect_ratio, sweep_50)
    # Equation 10.35
    d_f_ave = math.sqrt(ave_fuse_cross_section - 0.7854)
    delta_Clbeta_dihedral = - 0.0005 * aspect_ratio * (d_f_ave / span) ** 2
    #
    delta_Clbeta_zw = 0.042 * math.sqrt(aspect_ratio) * (z_w / span) * (d_f_ave / span)
    # From Figure 10.26
    delta_Clbeta_twist = FigureDigitization2.get_delta_Clbeta_twist(aspect_ratio, taper_ratio)

    Cl_beta_surface = 57.3 * (CL * Clbeta_CL_sweep50 * k_compress_sweep * k_fuselage + Clbeta_CL_AR) + \
                      dihedral * (Clbeta_dihedral * k_compress_dihedral + delta_Clbeta_dihedral) + \
                      delta_Clbeta_zw + \
                      delta_Clbeta_twist * twist * math.tan(sweep_25)

    return Cl_beta_surface

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

import numpy as np
import math

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2
from fastga.models.handling_qualities.utils.lift_curve_slope import get_lift_curve_slope


# TODO: register class
class ClBetaHT(FigureDigitization2):

    def setup(self):
        # Reference flight conditions
        self.add_input("data:reference_flight_condition:mach", val=np.nan)
        self.add_input("data:reference_flight_condition:alpha", val=np.nan, units="rad")

        # Wing Geometry
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:vertical_position", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:leading_edge:x:absolut", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:half_chord_point:x:absolut", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        # Horizontal Tail Geometry
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:sweep_50", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:dihedral", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:twist", val=np.nan, units="deg")

        # Horizontal Tail Aerodynamics
        self.add_input("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", val=np.nan, units="rad**-1")

        # Fuselage Geometry
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_output("data:handling_qualities:lateral:derivatives:horizontal_tail:Cl:beta", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference flight conditions
        mach = inputs["data:reference_flight_condition:mach"]
        alpha = inputs["data:reference_flight_condition:alpha"]

        # Wing geometry
        b = inputs["data:geometry:wing:span"]
        S_W = inputs["data:geometry:wing:area"]
        z_w = inputs["data:geometry:wing:vertical_position"]
        x_leadingedge_rootchord = inputs["data:geometry:wing:root:leading_edge:x:absolut"]
        x_leadingedge_tipchord = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        tipchord_length = inputs["data:geometry:wing:tip:chord"]

        # Horizontal Tail Geometry
        S_H = inputs["data:geometry:horizontal_tail:area"]
        b_H = inputs["data:geometry:horizontal_tail:span"]
        A_H = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        taper_ratio_H = inputs["data:geometry:horizontal_tail:taper_ratio"]
        sweep_25_H = inputs["data:geometry:horizontal_tail:sweep_25"]
        sweep_50_H = inputs["data:geometry:horizontal_tail:sweep_50"]
        twist_H = inputs["data:geometry:horizontal_tail:dihedral"]
        dihedral_H = inputs["data:geometry:horizontal_tail:dihedral"]

        # Fuselage Geometry
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        ave_fuse_diameter = math.sqrt(width_max * height_max)
        ave_fuse_cross_section = math.pi * (ave_fuse_diameter / 2) ** 2
        # l_f is described in Figure 10.22 of Roskam - Airplane Design Part VI. It is the x-position of the half-chord
        # point of the wing tip chord w.r.t the tip of the cockpit.
        l_f = inputs["data:geometry:wing:tip:half_chord_point:x:absolut"]
        if l_f == np.nan:
            l_f = x_leadingedge_rootchord + x_leadingedge_tipchord + tipchord_length / 2.0

        # Horizontal tail contribution to dihedral effect
        cl_alpha_h = inputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"]
        CL_alpha_H = get_lift_curve_slope(cl_alpha_h, A_H, mach, sweep_50_H)
        CL_H = CL_alpha_H * alpha

        # Rolling moment due to sideslip derivative Cl_beta. Also called Dihedral Effect
        # From Figure 10.20
        Clbeta_CL_sweep50 = FigureDigitization2.get_Clbeta_CL_sweep50(sweep_50_H, A_H, taper_ratio_H)
        # From Figure 10.21
        k_compress_sweep = FigureDigitization2.get_k_mach_sweep(mach, A_H, sweep_50_H)
        # From Figure 10.22
        k_fuselage = FigureDigitization2.get_k_fuselage(l_f, b_H, A_H, sweep_50_H)
        # From Figure 10.23
        Clbeta_CL_AR = FigureDigitization2.get_Clbeta_CL_aspect_ratio(A_H, taper_ratio_H)
        # From Figure 10.24
        Clbeta_dihedral = FigureDigitization2.get_Clbeta_dihedral(A_H, sweep_50_H, taper_ratio_H)
        # From Figure 10.25
        k_compress_dihedral = FigureDigitization2.get_k_mach_dihedral(mach, A_H, sweep_50_H)
        # Equation 10.35
        d_f_ave = math.sqrt(ave_fuse_cross_section - 0.7854)
        delta_Clbeta_dihedral = - 0.0005 * A_H * (d_f_ave / b_H) ** 2
        #
        delta_Clbeta_zw = 0.042 * math.sqrt(A_H) * (z_w / b_H) * (d_f_ave / b_H)
        # From Figure 10.26
        delta_Clbeta_twist = FigureDigitization2.get_delta_Clbeta_twist(A_H, taper_ratio_H)

        Cl_beta_HB = 57.3 * (CL_H * Clbeta_CL_sweep50 * k_compress_sweep * k_fuselage + Clbeta_CL_AR) + \
                     dihedral_H * (Clbeta_dihedral * k_compress_dihedral + delta_Clbeta_dihedral) + \
                     delta_Clbeta_zw + \
                     delta_Clbeta_twist * twist_H * math.tan(sweep_25_H)

        Cl_beta_H = Cl_beta_HB * (S_H * b_H) / (S_W * b)

        outputs["data:handling_qualities:lateral:derivatives:horizontal_tail:Cl:beta"] = Cl_beta_H

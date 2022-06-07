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


# TODO: register class
class ClBetaWingBody(FigureDigitization2):

    def setup(self):
        # Reference flight conditions
        self.add_input("data:reference_flight_condition:CL", val=np.nan)
        self.add_input("data:reference_flight_condition:mach", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:sweep_50", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:twist", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:vertical_position", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:leading_edge:x:absolut", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:half_chord_point:x:absolut", val=np.nan, units="m")

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_output("data:handling_qualities:lateral:derivatives:wing-body:Cl:beta", units="rad**-1" )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference flight conditions
        CL_s = float(inputs["data:reference_flight_condition:CL"])
        mach = float(inputs["data:reference_flight_condition:mach"])

        # Wing geometry
        b = float(inputs["data:geometry:wing:span"])
        A_W = float(inputs["data:geometry:wing:aspect_ratio"])
        taper_ratio_W = float(inputs["data:geometry:wing:taper_ratio"])
        sweep_25_W = float(inputs["data:geometry:wing:sweep_25"])
        sweep_50_W = float(inputs["data:geometry:wing:sweep_50"])
        tipchord_length = float(inputs["data:geometry:wing:tip:chord"])
        dihedral_W = float(inputs["data:geometry:wing:dihedral"])  # Angle in degrees
        twist_W = float(inputs["data:geometry:wing:twist"])
        z_w = float(inputs["data:geometry:wing:vertical_position"])
        x_leadingedge_rootchord = float(inputs["data:geometry:wing:root:leading_edge:x:absolut"])
        x_leadingedge_tipchord = float(inputs["data:geometry:wing:tip:leading_edge:x:local"])

        # Fuselage Geometry
        width_max = float(inputs["data:geometry:fuselage:maximum_width"])
        height_max = float(inputs["data:geometry:fuselage:maximum_height"])
        ave_fuse_diameter = math.sqrt(width_max * height_max)
        ave_fuse_cross_section = math.pi * (ave_fuse_diameter / 2) ** 2
        # l_f is described in Figure 10.22 of Roskam - Airplane Design Part VI. It is the x-position of the half-chord
        # point of the wing tip chord w.r.t the tip of the cockpit.
        l_f = float(inputs["data:geometry:wing:tip:half_chord_point:x:absolut"])
        if l_f == np.nan:
            l_f = x_leadingedge_rootchord + x_leadingedge_tipchord + tipchord_length / 2.0

        # Wing contribution to dihedral effect.
        # Rolling moment due to sideslip derivative Cl_beta. Also called Dihedral Effect
        # From Figure 10.20
        Clbeta_CL_sweep50 = self.get_Clbeta_CL_sweep50(sweep_50_W, A_W, taper_ratio_W)
        # From Figure 10.21
        k_compress_sweep = self.get_k_mach_sweep(mach, A_W, sweep_50_W)
        # From Figure 10.22
        k_fuselage = self.get_k_fuselage(l_f, b, A_W, sweep_50_W)
        # From Figure 10.23
        Clbeta_CL_AR = self.get_Clbeta_CL_aspect_ratio(A_W, taper_ratio_W)
        # From Figure 10.24
        Clbeta_dihedral = self.get_Clbeta_dihedral(A_W, sweep_50_W, taper_ratio_W)
        # From Figure 10.25
        k_compress_dihedral = self.get_k_mach_dihedral(mach, A_W, sweep_50_W)
        # Equation 10.35
        d_f_ave = math.sqrt(ave_fuse_cross_section / 0.7854)
        delta_Clbeta_dihedral = - 0.0005 * math.sqrt(A_W) * (d_f_ave / b) ** 2
        #
        delta_Clbeta_zw = 0.042 * math.sqrt(A_W) * (z_w / b) * (d_f_ave / b)
        # From Figure 10.26
        delta_Clbeta_twist = self.get_delta_Clbeta_twist(A_W, taper_ratio_W)

        Cl_beta_WB = float(
            57.3 * (CL_s * (Clbeta_CL_sweep50 * k_compress_sweep * k_fuselage + Clbeta_CL_AR) +
                     dihedral_W * (Clbeta_dihedral * k_compress_dihedral + delta_Clbeta_dihedral) +
                     delta_Clbeta_zw +
                     delta_Clbeta_twist * twist_W * math.tan(sweep_25_W)
        ))

        outputs["data:handling_qualities:lateral:derivatives:wing-body:Cl:beta"] = Cl_beta_WB

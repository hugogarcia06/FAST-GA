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
class CYBetaBody(FigureDigitization2):
    """
    Class to compute the contribution of the fuselage to the side force coefficient due to sideslip.
    Expressions obtained from Roskam - Airplane Design Part VI and USAF DATCOM.
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:volume", val=np.nan, units="m**3")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:vertical_position", val=np.nan, units="m")

        self.add_output("data:handling_qualities:lateral:derivatives:fuselage:CY:beta", units="rad**-1")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Fuselage Geometry
        width_max = float(inputs["data:geometry:fuselage:maximum_width"])
        height_max = float(inputs["data:geometry:fuselage:maximum_height"])
        fus_length = float(inputs["data:geometry:fuselage:length"])
        volume_fus = float(inputs["data:geometry:fuselage:volume"])
        ave_fuse_diameter = math.sqrt(width_max * height_max)

        # Wing geometry
        S_W = float(inputs["data:geometry:wing:area"])
        z_w = float(inputs["data:geometry:wing:vertical_position"])

        # z_f is the maximum fuselage depth
        z_f = math.sqrt(width_max * height_max)
        # Looking at typical commercial configurations we can estimate the ratio between zw and the fuselage radius (rf)
        # to be around 0.8
        # rf is half of the maximum fuselage depth (the "radius" of the fuselage). rf = zf/2
        r_f = z_f / 2.0
        if z_w != np.nan:
            zw_rf_ratio = z_w / r_f
        else:
            zw_rf_ratio = 0.75

        # Figure 10.8 de Roskam - Aircraft Design PART VI.
        if -1 <= zw_rf_ratio < 0:  # HIGH WING
            k_i = 1.0 - 0.85 * zw_rf_ratio

        elif 1 >= zw_rf_ratio > 0:  # LOW WING
            k_i = 1.0 + 0.49 * zw_rf_ratio

        elif zw_rf_ratio == 0:
            k_i = 1.0

        else:
            raise ValueError("Value of wing vertical position divided by fuselage radius must be between -1.0 and 1.0.")
        # S_o is the cross-sectional area of the fuselage at station x_o, where the flow ceases to be potential.
        # In this case it is approximated to the average cross-section of the fuselage.
        S_o_body = math.pi * (ave_fuse_diameter / 2) ** 2
        # S_o_nacelle = N_engines * math.pi * (diameter_nacelle/2)**2
        # S_o =  S_o_body + S_o_nacelle
        S_o = S_o_body
        # Fuselage and nacelle contribution are estimated by:
        # The 2 in the following equation can be changed for the more precise expression that takes into account the
        # fuselage fineness ratio: k2_k1. It can be obtained from Figure 4.2.1.1-20a in the USAF DATCOM.
        k2_k1 = self.get_k2_k1(fus_length, ave_fuse_diameter)
        # Expression 4.2.1.1-a from USAF DATCOM
        CL_alpha_fus = 2 * k2_k1 * S_o / (volume_fus ** (2 / 3))
        body_reference_area = volume_fus ** (2 / 3)
        CY_beta_B = float(- k_i * CL_alpha_fus * body_reference_area / S_W)
        # CY_beta_B = - 2 * k_i * S_o / S_W

        outputs["data:handling_qualities:lateral:derivatives:fuselage:CY:beta"] = CY_beta_B

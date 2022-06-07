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
from scipy import interpolate

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2
from fastga.models.handling_qualities.utils.lift_curve_slope import get_lift_curve_slope


# TODO: register class
class CYBetaVT(FigureDigitization2):
    # TODOC
    """

    """

    def setup(self):
        # Reference flight conditions
        self.add_input("data:reference_flight_condition:mach", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:vertical_position", val=np.nan, units="m")

        # Vertical tail Geometry
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_0", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:sweep_50", val=np.nan, units="rad")

        # Vertical tail Aerodynamics
        self.add_input("data:aerodynamics:vertical_tail:k_ar_effective", val=np.nan)
        self.add_input("data:aerodynamics:vertical_tail:airfoil:CL_alpha", val=np.nan, units="rad**-1")

        # Fuselage Geometry
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:depth_quarter_vt", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_output("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference flight conditions
        mach = inputs["data:reference_flight_condition:mach"]

        # Wing geometry
        S_W = inputs["data:geometry:wing:area"]
        A_W = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25_W = inputs["data:geometry:wing:sweep_25"]
        z_w = inputs["data:geometry:wing:vertical_position"]

        # Vertical Tail geometry
        S_V = inputs["data:geometry:vertical_tail:area"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        root_chord_vt = inputs["data:geometry:vertical_tail:root:chord"]
        A_V = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_ratio_V = inputs["data:geometry:vertical_tail:taper_ratio"]
        sweep_0_V = inputs["data:geometry:vertical_tail:sweep_0"]
        sweep_50_V = inputs["data:geometry:vertical_tail:sweep_50"]
        if sweep_50_V == np.nan and sweep_0_V != np.nan:
            sweep_50_V = math.atan(math.tan(sweep_0_V) - 2 / A_V * ((1 - taper_ratio_V) / (1 + taper_ratio_V)))
        elif sweep_50_V == np.nan and sweep_0_V == np.nan:
            print("No sweep angle has been specified")

        # Vertical tail Aerodynamics
        k_ar_effective = inputs["data:aerodynamics:vertical_tail:k_ar_effective"]
        A_V_eff = A_V * k_ar_effective
        cl_alpha_v = inputs["data:aerodynamics:vertical_tail:airfoil:CL_alpha"]

        # Fuselage Geometry
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        l_ar = inputs["data:geometry:fuselage:rear_length"]

        # Aerodynamics
        # The variable wake effect takes into account the wake deflection and the tail dynamic pressure:
        # This is Equation 10.31 from Roskam - Aircraft Design Part VI
        # wake_effect = (1 + dsigma_dbeta)*eta_v
        # z_w is the distance from wing root quarter chord point to the fuselage centerline, positive below fuselage
        # centerline.
        # z_f is the maximum fuselage depth
        z_f = np.sqrt(width_max * height_max)
        # Looking at typical commercial configurations we can estimate the ratio between zw and the fuselage radius (rf)
        # to be around 0.8
        # rf is half of the maximum fuselage depth. rf = zf/2
        zw_rf_ratio = 0.8
        # z_w = zw_rf_ratio * (z_f/2)
        wake_effect = 0.724 + 3.06 * ((S_V / S_W) / (1 + math.cos(sweep_25_W))) + 0.4 * z_w / z_f + 0.009 * A_W

        CL_alpha_V = get_lift_curve_slope(cl_alpha_v, A_V_eff, mach, sweep_50_V)
        # k_v empirical factor for estimating side-force due to sideslip of a single vertical tail.
        # It depends on the ratio b_v/2r1, where 2r1 is the fuselage depth at quarter chord-point of vertical panels
        # Figure 10.12 Roskam Airplane Design Part VI
        avg_fus_depth = inputs["data:geometry:fuselage:depth_quarter_vt"]
        if inputs["data:geometry:fuselage:depth_quarter_vt"] != np.nan:
            avg_fus_depth = inputs["data:geometry:fuselage:depth_quarter_vt"]
        elif inputs["data:geometry:fuselage:depth_quarter_vt"] == np.nan and l_ar != np.nan:
            avg_fus_depth = np.sqrt(width_max * height_max) * root_chord_vt / (2.0 * l_ar)
        else:
            print("There is geometrical data missing to compute the constant k_v")

        if 0.0 < (b_v / avg_fus_depth) < 2.0:
            k_v = 0.75
        elif 2.0 <= (b_v / avg_fus_depth) < 3.5:
            k_v = interpolate.interp1d([2.0, 3.5], [0.75, 1.0])(float(b_v / avg_fus_depth))
        else:
            k_v = 1.0

        CY_beta_V = float(- k_v * CL_alpha_V * wake_effect * S_V / S_W)

        outputs["data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta"] = CY_beta_V

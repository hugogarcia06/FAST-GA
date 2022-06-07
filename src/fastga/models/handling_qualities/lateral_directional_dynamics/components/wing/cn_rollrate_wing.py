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

from fastga.models.handling_qualities.utils.lift_curve_slope import get_lift_curve_slope

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2


# TODO: register class
class CnRollRateWing(FigureDigitization2):

    def setup(self):
        # Flight reference condition
        self.add_input("data:reference_flight_condition:alpha", val=np.nan)
        self.add_input("data:reference_flight_condition:mach", val=np.nan)
        self.add_input("data:reference_flight_condition:CL", val=np.nan)
        self.add_input("data:reference_flight_condition:flaps_deflection", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:twist", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        # flaps
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)

        self.add_input("data:handling_qualities:stick_fixed_static_margin", val=np.nan)

        # Wing Aerodynamics
        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

        self.add_input("data:aerodynamics:flaps:landing:CL_2D", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL_2D", val=np.nan)

        self.add_input("data:handling_qualities:lateral:derivatives:wing:Cl:rollrate", val=np.nan, units="rad**-1")

        self.add_output("data:handling_qualities:lateral:derivatives:wing:Cn:rollrate", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Flight reference condition
        alpha = inputs["data:reference_flight_condition:alpha"]
        mach = inputs["data:reference_flight_condition:mach"]
        CL_s = inputs["data:reference_flight_condition:CL"]
        CL_W = CL_s
        # Flaps deflection used
        delta_flaps = inputs["data:reference_flight_condition:flaps_deflection"]

        # Wing geometry
        b = inputs["data:geometry:wing:span"]
        mac_W = inputs["data:geometry:wing:MAC:length"]
        A_W = inputs["data:geometry:wing:aspect_ratio"]
        taper_ratio_W = inputs["data:geometry:wing:taper_ratio"]
        twist_W = inputs["data:geometry:wing:twist"]
        sweep_25_W = inputs["data:geometry:wing:sweep_25"]
        flaps_span_ratio = inputs["data:geometry:flap:span_ratio"]

        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        # Wing aerodynamics
        cl_alpha_w = inputs["data:aerodynamics:wing:airfoil:CL_alpha"]
        # Coefficient from Equation 10.65 of Roskam - Airplane Design Part VI
        B = math.sqrt(1 - mach ** 2 * (math.cos(sweep_25_W)) ** 2)
        A = A_W
        # x is the distance from the c.g. to the a.c. positive for the a.c. aft of the c.g.
        x = static_margin * mac_W
        c = mac_W
        Cnp_CL_0_0 = - 1 / 6 * (
                (
                        A + 6 * (A + math.cos(sweep_25_W)) * (
                        x / c * math.tan(sweep_25_W) / A + (math.tan(sweep_25_W)) ** 2 / 12)
                ) / (
                        A + 4 * math.cos(sweep_25_W)
                )
        )
        # Coefficient from Equation 10.63 of Roskam - Airplane Design Part VI
        Cnp_CL_0 = Cnp_CL_0_0 * (
                (A + 4 * math.cos(sweep_25_W)) / (A * B + 4 * math.cos(sweep_25_W))
        ) * (
                           (A * B + 0.5 * (A * B + math.cos(sweep_25_W)) * (math.tan(sweep_25_W)) ** 2) / (
                           A + 0.5 * (A + math.cos(sweep_25_W)) * (math.tan(sweep_25_W)) ** 2)
                   )
        # Figure 10.37
        Cnp_twist = self.get_delta_Cnp_twist(A_W, taper_ratio_W)
        # TODO: introduce flap condition
        delta_flaps = 0.0
        slope_flaps = 0.0
        delta_Cnp_flaps = 0.0
        """
        if delta_flaps != np.nan:
            # Figure 10.38
            delta_Cnp_flaps = self.get_delta_Cnp_flaps(A_W, taper_ratio_W, flaps_span_ratio)
            # Equation 10.66 from Roskam
            if (landing condition):
                delta_cl = inputs["data:aerodynamics:flaps:landing:CL_2D"]
            elif (takeoff condition):
                delta_cl = inputs["data:aerodynamics:flaps:takeoff:CL_2D"]

            slope_flaps = delta_cl / (cl_alpha_w * delta_flaps)
        else:
            delta_flaps = 0.0
            slope_flaps = 0.0
        """

        Cn_p_W = Cnp_CL_0 * CL_W + Cnp_twist * twist_W + delta_Cnp_flaps * slope_flaps * delta_flaps

        # Cl_p_W = inputs["data:handling_qualities:lateral:derivatives:wing:Cl:rollrate"]
        # TODO: compute the dimensionless correction factor (USAF DATCOM Equation 7.1.2.3-d)
        # K =
        # Cn_p_W = - Cl_p_W * math.tan(alpha) - K *(- Cl_p_W * math.tan(alpha) - Cnp_CL_0 * CL_W) + Cnp_twist * twist_W + \
        #          delta_Cnp_flaps * slope_flaps * delta_flaps


        outputs["data:handling_qualities:lateral:derivatives:wing:Cn:rollrate"] = Cn_p_W

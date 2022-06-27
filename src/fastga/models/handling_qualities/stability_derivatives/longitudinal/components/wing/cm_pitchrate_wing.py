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
import numpy as np

import openmdao.api as om

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2


class CmPitchRateWing(om.ExplicitComponent):

    def setup(self):
        # Reference Flight Condition
        # self.add_input(weight)
        # self.add_input(dynamic_pressure)
        self.add_input("data:reference_flight_condition:mach", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

        # Wing Aerodynamics
        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

        self.add_input("data:handling_qualities:stick_fixed_static_margin", val=np.nan)

        self.add_output(
            "data:handling_qualities:longitudinal:derivatives:wing:Cm:pitchrate", units="rad**-1"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference Flight Conditions
        mach = inputs["data:reference_flight_condition:mach"]

        # Wing Geometry
        A_W = inputs["data:geometry:wing:aspect_ratio"]
        mac_W = inputs["data:geometry:wing:MAC:length"]
        sweep_25_W = inputs["data:geometry:wing:sweep_25"]
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]
        X_W = static_margin * mac_W

        # Wing Aerodynamics
        cl_alpha_wing = inputs["data:aerodynamics:wing:airfoil:CL_alpha"]

        beta = math.sqrt(1 - mach ** 2 * (math.cos(sweep_25_W)) ** 2)
        # There is bibliography (Tierno) where the Cmq is estimated as 1.10*Cmq_H, as the contribution of the
        # wing-fuselage is small compared to that of the horizontal tail.

        k = FigureDigitization2.get_k_w(A_W)
        # N_1 = A_W * ((2 * (X_W / mac_W) ** 2) + 0.5 * (X_W / mac_W))
        # D_1 = A_W + 2 * math.cos(sweep_25_W)
        # N_2 = A_W ** 3 * (math.tan(sweep_25_W)) ** 2
        # D_2 = 24 * (A_W + 6 * math.cos(sweep_25_W))
        # Cm_q_W0 = - k * cl_alpha_wing * math.cos(sweep_25_W) * (N_1/D_1 + N_2/D_2 + 1/8)
        Cm_q_W0 = - k * cl_alpha_wing * math.cos(sweep_25_W) * (
                (
                        A_W * ((2 * (X_W / mac_W) ** 2) + 0.5 * (X_W / mac_W))
                ) / (
                        A_W + 2 * math.cos(sweep_25_W)
                ) + (
                        A_W ** 3 * (math.tan(sweep_25_W)) ** 2
                ) / (
                        24 * (A_W + 6 * math.cos(sweep_25_W))
                ) + 1 / 8
        )

        Cm_q_W = Cm_q_W0 * (
                (
                        (A_W ** 3 * (math.tan(sweep_25_W)) ** 2) / (A_W * beta + 6 * math.cos(sweep_25_W)) + 3 / beta
                ) / ((
                             A_W ** 3 * (math.tan(sweep_25_W)) ** 2
                     ) / (
                             A_W + 6 * math.cos(sweep_25_W)) + 3
                     )
        )

        outputs["data:handling_qualities:longitudinal:derivatives:wing:Cm:pitchrate"] = Cm_q_W

    def get_Cm_q_0_wing(self, cl_alpha_wing, sweep_25, A, X_W, mac_W):
        """
        :param cl_alpha_wing: spanwise average value of the wing section lift curve slope
        :param sweep_25: quarter chord wing sweep angle expressed in radians.
        :param A: aspect ratio of the wing.
        :param X_W: the (positive rearward) distance from the airplane cg to the wing aerodynamic center
        :param mac_W: mean aerodynamic chord length
        """

        # Computation of correction constant for wing contribution to Cmq.
        # Can also be found in Roskam - Aircraft Design PArt VI, Figure 10.40
        k = FigureDigitization2.get_k_w(A)

        Cm_q_W0 = - k * cl_alpha_wing * math.cos(sweep_25) * (
                (
                        A * (2 * (X_W / mac_W) ** 2) + 0.5 * (X_W / mac_W)
                ) / (
                        A + 2 * math.cos(sweep_25)
                ) + (
                        A ** 3 * (math.tan(sweep_25)) ** 2
                ) / (
                        24 * (A + 6 * math.cos(sweep_25))
                ) + 1 / 8
        )

        return Cm_q_W0

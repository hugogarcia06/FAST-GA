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

from fastga.models.handling_qualities.utils.lift_curve_slope import get_lift_curve_slope


class CLPitchRateWing(om.ExplicitComponent):
    # TODOC
    """
    Computes the wing-body contribution to the CL pitch rate derivative.
    """

    def setup(self):
        # Reference Flight Condition
        # self.add_input(weight)
        # self.add_input(dynamic_pressure)
        self.add_input("data:reference_flight_condition:mach", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_50", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:sweep_0", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

        # Wing Aerodynamics
        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

        self.add_input("data:handling_qualities:stick_fixed_static_margin", val=np.nan)

        self.add_output("data:handling_qualities:longitudinal:derivatives:wing:CL:pitchrate", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference Flight Conditions
        mach = inputs["data:reference_flight_condition:mach"]

        # Wing Geometry
        A_W = inputs["data:geometry:wing:aspect_ratio"]
        taper_ratio_W = inputs["data:geometry:wing:taper_ratio"]
        mac_W = inputs["data:geometry:wing:MAC:length"]
        sweep_0_W = inputs["data:geometry:wing:sweep_0"]
        sweep_25_W = inputs["data:geometry:wing:sweep_25"]
        sweep_50_W = inputs["data:geometry:wing:sweep_50"]
        if sweep_50_W == np.nan and sweep_0_W != np.nan:
            sweep_50_W = sweep_0_W - 2 / A_W * ((1 - taper_ratio_W) / (1 + taper_ratio_W))
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]
        X_W = static_margin * mac_W

        cl_alpha_wing = inputs["data:aerodynamics:wing:airfoil:CL_alpha"]
        beta = math.sqrt(1 - mach ** 2 * (math.cos(sweep_25_W)) ** 2)
        # CL_q_W0 = self.get_CL_q_0(cl_alpha_wing, X_W, mac_W, A_W, sweep_50_W)
        CL_alpha_0 = get_lift_curve_slope(cl_alpha_wing, A_W, 0, sweep_50_W)
        CL_q_W0 = (0.5 + 2 * X_W / mac_W) * CL_alpha_0
        CL_q_W = CL_q_W0 * (
                A_W + 2 * math.cos(sweep_25_W
                                   )
        ) / (
                         A_W * beta + 2 * math.cos(sweep_25_W)
                 )

        outputs["data:handling_qualities:longitudinal:derivatives:wing:CL:pitchrate"] = CL_q_W

    # The following class allows to compute the CL_q derivative at M = 0 for a certain surface
    def get_CL_q_0(self, cl_alpha_wing, X_W, mac, aspect_ratio, sweep_50):
        """
        Wing lift due to pitch rate derivative at M = 0
        :param cl_alpha_wing:
        :param X_w: the (positive rearward) distance from the airplane cg to the wing aerodynamic center
        :param mac: mean aerodynamic chord length
        :param aspect_ratio: wing aspect ratio
        :param sweep_50: sweep angle at half the chord of the surface we are calculating
        :return:
        """
        CL_alpha_0 = get_lift_curve_slope(cl_alpha_wing, aspect_ratio, 0, sweep_50)
        CL_q_0 = (0.5 + 2 * X_W / mac) * CL_alpha_0

        return CL_q_0

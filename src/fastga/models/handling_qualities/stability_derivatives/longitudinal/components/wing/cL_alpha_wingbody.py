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

import openmdao.api as om
import numpy as np
import math

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2
from fastga.models.handling_qualities.utils.lift_curve_slope import get_lift_curve_slope


class CLAlphaWingBody(om.ExplicitComponent):
    # TODOC
    """
    Computes the wing-body contribution to the CL_alpha derivative.
    """

    def setup(self):
        # Reference Flight Condition
        self.add_input("data:reference_flight_condition:mach", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_0", val=np.nan, units="rad")

        # Fuselage Geometry
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        # Wing Aerodynamics
        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

        self.add_output("data:handling_qualities:longitudinal:derivatives:wing:CL:alpha", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference flight conditions
        mach = inputs["data:reference_flight_condition:mach"]

        # Wing Geometry
        A_W = inputs["data:geometry:wing:aspect_ratio"]
        taper_ratio_W = inputs["data:geometry:wing:taper_ratio"]
        b = inputs["data:geometry:wing:span"]
        sweep_0_W = inputs["data:geometry:wing:sweep_0"]
        sweep_50_W = math.atan(math.tan(sweep_0_W) - 2/A_W*((1-taper_ratio_W)/(1+taper_ratio_W)))

        # Fuselage Geometry
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        # Equivalent fuselage diameter
        d = math.sqrt(width_max * height_max)

        cl_alpha_wing = inputs["data:aerodynamics:wing:airfoil:CL_alpha"]
        CL_alpha_W = get_lift_curve_slope(cl_alpha_wing, A_W, mach, sweep_50_W)

        # NOTE: one of these method is more appropiate than the other. (See DATCOM section 4.3.1.2)
        # The USAF DATCOM presents two different methods to compute the wing-body interference in the lift-curve slope.
        # In each method a different constants is used.
        # Method 1:
        K_WB = FigureDigitization2.get_k_wb(d, b)
        K_BW = FigureDigitization2.get_k_bw(d, b)
        CL_alpha_WB = (K_WB + K_BW) * CL_alpha_W

        # Method 2, also presented in Roskam's.
        # If the ratio of the wing span, b, to fuselage diameter, d, is reasonably large, say >2, a good approximation
        # is the following (Equation 8.44. Roskam - Aircraft Design Part VI):
        k_wb = 1 - 0.25 * (d / b) ** 2 + 0.025 * (d / b)
        CL_alpha_WB2 = k_wb * CL_alpha_W

        outputs["data:handling_qualities:longitudinal:derivatives:wing:CL:alpha"] = CL_alpha_WB

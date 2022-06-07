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
from ....utils.downwash import get_downwash
from ....utils.lift_curve_slope import get_lift_curve_slope


class CLAlphaRateHT(om.ExplicitComponent):
    # TODOC
    """

    """

    def setup(self):
        # Reference Flight Condition
        self.add_input("data:reference_flight_condition:mach", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:sweep_0", val=np.nan, units="rad")

        # Wing Aerodynamics
        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

        # Horizontal Tail Geometry
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:sweep_0", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="m")

        # Horizontal Tail Aerodynamics
        self.add_input("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)

        self.add_output("data:handling_qualities:longitudinal:derivatives:horizontal_tail:CL:alpharate", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference flight conditions
        mach = inputs["data:reference_flight_condition:mach"]

        # Wing Geometry
        S_W = inputs["data:geometry:wing:area"]
        A_W = inputs["data:geometry:wing:aspect_ratio"]
        mac_W = inputs["data:geometry:wing:MAC:length"]
        taper_ratio_W = inputs["data:geometry:wing:taper_ratio"]
        b = inputs["data:geometry:wing:span"]
        sweep_25_W = inputs["data:geometry:wing:sweep_25"]
        sweep_0_W = inputs["data:geometry:wing:sweep_0"]
        sweep_50_W = sweep_0_W - 2/A_W*((1-taper_ratio_W)/(1+taper_ratio_W))

        # Wing Aerodynamics
        cl_alpha_wing = inputs["data:aerodynamics:wing:airfoil:CL_alpha"]

        # Horizontal Tail Geometry
        # TODO: compute exactly the distance of l_h (distance from ht mac to aircraft cg)
        l_H = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        S_H = inputs["data:geometry:horizontal_tail:area"]
        V_H = l_H * S_H / (S_W * mac_W)
        A_H = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        taper_ratio_H = inputs["data:geometry:horizontal_tail:taper_ratio"]
        sweep_0_H = inputs["data:geometry:horizontal_tail:sweep_0"]
        sweep_50_H = math.atan(math.tan(sweep_0_H) - 2 / A_H * ((1 - taper_ratio_H) / (1 + taper_ratio_H)))
        h_H = inputs["data:geometry:horizontal_tail:z:from_wingMAC25"]

        # Roskam presents a method of computing this ratio in his Airplane Design PART IV: Preliminary Calculation...
        # This method comes from DATCOM 1978.
        # The dynamic pressure ratio at the horizontal tail can usually be assumed in the range 0.9 - 1.0.
        # A value of .9 being on the conservative side in power-off flight.
        eta_H = inputs["data:aerodynamics:horizontal_tail:efficiency"]

        cl_alpha_h = inputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"]
        CL_alpha_H = get_lift_curve_slope(cl_alpha_h, A_H, mach, sweep_50_H)
        deps_dalpha = get_downwash(cl_alpha_wing, A_W, taper_ratio_W, b, l_H, h_H, sweep_25_W, mach, sweep_50_W)
        CL_alpharate_H = 2 * CL_alpha_H * eta_H * V_H * deps_dalpha

        outputs["data:handling_qualities:longitudinal:derivatives:horizontal_tail:CL:alpharate"] = CL_alpharate_H

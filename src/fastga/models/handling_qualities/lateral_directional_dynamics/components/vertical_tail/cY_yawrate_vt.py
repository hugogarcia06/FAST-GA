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
class CYYawRateVT(FigureDigitization2):

    def setup(self):
        # Flight reference condition
        self.add_input("data:reference_flight_condition:alpha", val=np.nan, units="rad")

        # Wing Geometry
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        # Vertical Tail Geometry
        self.add_input("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", val=np.nan, units="rad**-1")
        self.add_input("data:geometry:vertical_tail:MAC:z", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")

        self.add_output("data:handling_qualities:lateral:derivatives:vertical_tail:CY:yawrate", units="rad**-1")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Flight reference condition
        alpha = inputs["data:reference_flight_condition:alpha"] # in radians

        # Wing geometry
        b = inputs["data:geometry:wing:span"]

        # Vertical tail geometry
        # TODO: compute l_v as the distance between the vertical tail aerodynamic center and the CG of the aircraft
        l_v = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        z_v = inputs["data:geometry:vertical_tail:MAC:z"]

        CY_beta_V = inputs["data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta"]

        CY_r_V = -2 * CY_beta_V * (l_v*math.cos(alpha) + z_v*math.sin(alpha)) / b

        outputs["data:handling_qualities:lateral:derivatives:vertical_tail:CY:yawrate"] = CY_r_V
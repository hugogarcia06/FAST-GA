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
from fastga.models.handling_qualities.utils.dihedral_effect import get_dihedral_effect_surface


# TODO: register class
class CnBetaVT(FigureDigitization2):

    def setup(self):
        # Reference flight conditions
        self.add_input("data:reference_flight_condition:alpha", val=np.nan, units="rad")

        # Wing Geometry
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        # Vertical Tail Geometry
        self.add_input("data:geometry:vertical_tail:MAC:z", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")

        self.add_input("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", val=np.nan, units="rad**-1")

        self.add_output("data:handling_qualities:lateral:derivatives:vertical_tail:Cn:beta", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference flight conditions
        alpha = float(inputs["data:reference_flight_condition:alpha"])

        # Wing geometry
        b = float(inputs["data:geometry:wing:span"])

        # Vertical Tail geometry
        z_v = float(inputs["data:geometry:vertical_tail:MAC:z"])
        # TODO: compute l_v as the distance between the vertical tail aerodynamic center and the CG of the aircraft
        l_v = float(inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"])
        CY_beta_V = float(inputs["data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta"])

        Cn_beta_V = float(- CY_beta_V * ((l_v * math.cos(alpha) + z_v * math.sin(alpha)) / b))

        outputs["data:handling_qualities:lateral:derivatives:vertical_tail:Cn:beta"] = Cn_beta_V


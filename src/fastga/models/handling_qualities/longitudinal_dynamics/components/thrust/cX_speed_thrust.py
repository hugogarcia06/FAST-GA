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


class CXSpeedThrust(om.ExplicitComponent):
    """
    Estimation of the coefficient CX speed derivative due to thrust. Expressions found in Roskma - Airplane Design Part
    VI: Preliminary Calculation of Aerodynamic Thrust and Power Characteristics.
    """

    def setup(self):
        self.add_input("data:reference_flight_condition:CT", val=np.nan)
        self.add_input("data:reference_flight_condition:dynamic_pressure", val=np.nan, units="Pa")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:handling_qualities:longitudinal:derivatives:thrust:CX:speed", units="rad**-1")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        CT = inputs["data:reference_flight_condition:CT"]
        q = inputs["data:reference_flight_condition:dynamic_pressure"]
        S = inputs["data:geometry:wing:area"]

        # Variable Pitch (constant speed propellers)
        CX_T_u = - 3 * CT

        # Fixed pitch propellers
        # TODO: compute derivative for fixed pitch propellers and IF condition. Are propellers differed inside FAST?
        # dP_du =
        # CX_T_u = 1 / (q * S) * dP_du - 3 * CT

        outputs["data:handling_qualities:longitudinal:derivatives:thrust:CX:speed"] = CX_T_u

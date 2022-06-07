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


class CmAlphaThrust(om.ExplicitComponent):
    """
    Estimation of the coefficient Cm alpha derivative due to thrust. Expressions found in Roskma - Airplane Design Part
    VI: Preliminary Calculation of Aerodynamic Thrust and Power Characteristics.
    """

    def setup(self):
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan, units="m")


        self.add_output("data:handling_qualities:longitudinal:derivatives:thrust:Cm:speed", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        c = inputs["data:geometry:wing:MAC:length"]
        # Number of engines
        N = inputs["data:geometry:propulsion:engine:count"]

        # delta_dCm_dCL =

        # Equation 10.20 from Roskam
        # TODO: write de expression
        # Cm_T_alpha = delta_dCm_dCL * CL_alpha
        Cm_T_alpha = 0.0

        outputs["data:handling_qualities:longitudinal:derivatives:thrust:Cm:alpha"] = Cm_T_alpha

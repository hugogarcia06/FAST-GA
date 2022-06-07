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


class CmSpeedThrust(om.ExplicitComponent):
    """
    Estimation of the coefficient Cm speed derivative due to thrust. Expressions found in Roskma - Airplane Design Part
    VI: Preliminary Calculation of Aerodynamic Thrust and Power Characteristics.
    """

    def setup(self):
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:handling_qualities:longitudinal:derivatives:thrust:CX:speed", val=np.nan, units="rad**-1")
        self.add_input("data:reference_flight_condition:CG:z", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:z", val=np.nan, units="m")

        self.add_output("data:handling_qualities:longitudinal:derivatives:thrust:Cm:speed", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        c = inputs["data:geometry:wing:MAC:length"]
        CX_T_u = inputs["data:handling_qualities:longitudinal:derivatives:thrust:CX:speed"]
        z_cg = inputs["data:reference_flight_condition:CG:z"]
        z_engine = inputs["data:weight:propulsion:engine:CG:z"]
        # Thrust moment arm relative to the center of gravity (vertical distance).
        # Counted positive of beneath the center of gravity.
        d_T = z_cg - z_engine

        Cm_T_u = d_T / c * CX_T_u

        outputs["data:handling_qualities:longitudinal:derivatives:thrust:Cm:speed"] = Cm_T_u

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
import math
import numpy as np

from fastga.models.handling_qualities.utils.drag_compressibility import *


# TODO: register the class
class SpeedDerivatives(om.ExplicitComponent):
    """
    Computes de stability speed derivatives of the CD, CL and Cm coefficients. The expressions used for the
    computation of the different coefficients were obtained from Roskam. Estimating Stability and Control Derivatives.

    """

    def setup(self):
        self.add_input("data:reference_flight_condition:mach", val=np.nan)
        self.add_input("data:reference_flight_condition:CL", val=np.nan)

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_output("data:handling_qualities:longitudinal:derivatives:CL:speed", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:CD:speed", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:Cm:speed", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference Flight Conditions
        mach = inputs["data:reference_flight_condition:mach"]
        CL = inputs["data:reference_flight_condition:CL"]

        # Wing Geometry
        sweep_25_W = inputs["data:geometry:wing:sweep_25"]

        # Drag due to mach number obtained with Snorri Gudmundsson.
        dCD_dM = get_drag_mach_derivative(mach)
        # TODO: derivative of the aerodynamic center with mach number.
        dxacw_dM = 1.0

        CD_u = mach * dCD_dM
        # Roskam presents two different expressions in two of his books
        # 1. CL_u = mach**2*/(1-mach**2) * CL
        # 2. CL_u = mach**2*(math.cos(sweep_25_W)**2/(1-mach**2*(math.cos(sweep_25_W)**2) * CL
        CL_u = mach ** 2 * (math.cos(sweep_25_W)) ** 2 / (1 - mach ** 2 * (math.cos(sweep_25_W)) ** 2) * CL
        Cm_u = -CL * dxacw_dM * mach

        outputs["data:handling_qualities:longitudinal:derivatives:CD:speed"] = CD_u
        outputs["data:handling_qualities:longitudinal:derivatives:CL:speed"] = CL_u
        outputs["data:handling_qualities:longitudinal:derivatives:Cm:speed"] = Cm_u

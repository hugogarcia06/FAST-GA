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
from ..utils.downwash import get_downwash
from ..utils.lift_curve_slope import get_lift_curve_slope


# TODO: register the class
class AoaRateDerivatives(om.ExplicitComponent):
    """
    Computes de stability angle of attack rate derivatives of the CD, CL and Cm coefficients. The expressions used for
    the computation of the different coefficients were obtained from Roskam. Estimating Stability and Control
    Derivatives.
    """

    def setup(self):

        self.add_input("data:handling_qualities:longitudinal:derivatives:horizontal_tail:CL:alpharate", val=np.nan,
                       units="rad**-1")
        self.add_input("data:handling_qualities:longitudinal:derivatives:horizontal_tail:Cm:alpharate", val=np.nan,
                       units="rad**-1")

        self.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpharate", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:CD:alpharate", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:Cm:alpharate", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # CD derivative
        CD_alpharate = 0.0

        # CL derivative Except for triangular wings, no explicit formulas are available for calculating the
        # CL_alpharate_W (Roskam) You can observe that the following formulas are based on the assumption that the
        # contribution of the horizontal tail is the only important contribution to these derivatives. This
        # assumption is frequently satisfied.
        CL_alpharate_W = 0.0
        CL_alpharate_H = inputs["data:handling_qualities:longitudinal:derivatives:horizontal_tail:CL:alpharate"]
        CL_alpharate = CL_alpharate_W + CL_alpharate_H

        # Cm derivative
        # In this case, the calculation is based on the assumption that the contribution of the horizontal tail is
        # the most important to these derivatives. This assumption is frequently satisfied.
        Cm_alpharate_W = 0.0
        Cm_alpharate_H = inputs["data:handling_qualities:longitudinal:derivatives:horizontal_tail:Cm:alpharate"]
        Cm_alpharate = Cm_alpharate_W + Cm_alpharate_H

        outputs["data:handling_qualities:longitudinal:derivatives:CD:alpharate"] = CD_alpharate
        outputs["data:handling_qualities:longitudinal:derivatives:CL:alpharate"] = CL_alpharate
        outputs["data:handling_qualities:longitudinal:derivatives:Cm:alpharate"] = Cm_alpharate

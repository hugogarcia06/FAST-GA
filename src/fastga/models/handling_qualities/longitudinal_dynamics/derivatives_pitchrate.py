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

# IMPORTANT: in some expressions of this code fragment it is necessary to use a chord length. Roskam uses the standard
# mean chord in his equations, however, in this code we are going to use the MAC (mean aerodynamic chord).

import numpy as np
import openmdao.api as om


# TODO: register the class
class PitchRateDerivatives(om.ExplicitComponent):
    """
    Computes de stability pitch rate derivatives of the CD, CL and Cm coefficients. The expressions used for the
    computation of the different coefficients were obtained from Roskam. Estimating Stability and Control Derivatives.
    Same expressions can be found in Roskam - Aircraft Design Part VI: Preliminary Calculation of Aerodynamic Thrust and
    Power Characteristics.
    """

    def setup(self):
        self.add_input(
            "data:handling_qualities:longitudinal:derivatives:wing:CL:pitchrate", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:handling_qualities:longitudinal:derivatives:wing:Cm:pitchrate", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:handling_qualities:longitudinal:derivatives:horizontal_tail:CL:pitchrate", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:handling_qualities:longitudinal:derivatives:horizontal_tail:Cm:pitchrate", val=np.nan, units="rad**-1"
        )

        self.add_output("data:handling_qualities:longitudinal:derivatives:CL:pitchrate", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:CD:pitchrate", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:Cm:pitchrate", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # ---- CD pitch rate derivative ----
        CD_q = 0

        # ---- CL pitch rate derivative ----
        CL_q_W = inputs["data:handling_qualities:longitudinal:derivatives:wing:CL:pitchrate"]
        CL_q_H = inputs["data:handling_qualities:longitudinal:derivatives:horizontal_tail:CL:pitchrate"]

        CL_q = CL_q_W + CL_q_H

        # ---- Cm pitch rate derivative ----
        Cm_q_W = inputs["data:handling_qualities:longitudinal:derivatives:wing:Cm:pitchrate"]
        Cm_q_H = inputs["data:handling_qualities:longitudinal:derivatives:horizontal_tail:Cm:pitchrate"]

        Cm_q = Cm_q_W + Cm_q_H

        outputs["data:handling_qualities:longitudinal:derivatives:CD:pitchrate"] = CD_q
        outputs["data:handling_qualities:longitudinal:derivatives:CL:pitchrate"] = CL_q
        outputs["data:handling_qualities:longitudinal:derivatives:Cm:pitchrate"] = Cm_q

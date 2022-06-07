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

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2


# TODO: register class
class CnBetaBody(FigureDigitization2):

    def setup(self):
        self.add_input("data:aerodynamics:fuselage:cruise:CnBeta", val=np.nan, units="rad**-1")

        self.add_output("data:handling_qualities:lateral:derivatives:wing:Cn:beta", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # The fuselage contribution can be computed in two different ways: Roskam or Raymer
        # Roskam:
        # Figure 10.28
        # k_n =
        # Figure 10.29
        # k_Rl =
        # S_Bs: Body Side Area and l_b: length of the fuselage
        # S_Bs =
        # l_B =
        # Cn_beta_B = -57.3 * k_n * k_Rl * S_Bs/S_W * l_B/b # Equation 10.42 from Roskam

        # However, we find a different expression in Raymer (Equation 16.50) for computing the contribution of the
        # fuselage. This expression is the one previously used in the code tu compute Cn_beta_fuselage
        # Cn_beta_B = - 1.3 * volume / (S_W * b) * (Df/Wf). We can just receive this value as an input from other
        # modules
        Cn_beta_B = inputs["data:aerodynamics:fuselage:cruise:CnBeta"]

        outputs["data:handling_qualities:lateral:derivatives:wing:Cn:beta"] = Cn_beta_B

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
class YawRateDerivatives(FigureDigitization2):
    """
    Computes de stability roll rate derivatives of the CY, Cl and Cn coefficients. The expressions used for the
    computation of the different coefficients were obtained from Roskam - Estimating Stability and Control Derivatives.
    """

    def setup(self):
        self.add_input("data:handling_qualities:lateral:derivatives:vertical_tail:CY:yawrate", val=np.nan,
                       units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:wing:Cl:yawrate", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:vertical_tail:Cl:yawrate", val=np.nan,
                       units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:wing:Cn:yawrate", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:vertical_tail:Cn:yawrate", val=np.nan,
                       units="rad**-1")

        self.add_output("data:handling_qualities:lateral:derivatives:CY:yawrate", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:Cl:yawrate", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:Cn:yawrate", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # ------ Side Force Coefficient CY_r. Primarily influenced by the vertical tail ------
        CY_r_V = inputs["data:handling_qualities:lateral:derivatives:vertical_tail:CY:yawrate"]
        CY_r = CY_r_V

        # ------ Rolling moment coefficient Cl_r ------
        Cl_r_W = inputs["data:handling_qualities:lateral:derivatives:wing:Cl:yawrate"]
        Cl_r_V = inputs["data:handling_qualities:lateral:derivatives:vertical_tail:Cl:yawrate"]
        Cl_r = Cl_r_W + Cl_r_V

        # ------ Yawing moment coefficient Cn_r ------
        Cn_r_W = inputs["data:handling_qualities:lateral:derivatives:wing:Cn:yawrate"]
        Cn_r_V = inputs["data:handling_qualities:lateral:derivatives:vertical_tail:Cn:yawrate"]
        Cn_r = Cn_r_W + Cn_r_V

        outputs["data:handling_qualities:lateral:derivatives:CY:yawrate"] = CY_r
        outputs["data:handling_qualities:lateral:derivatives:Cl:yawrate"] = Cl_r
        outputs["data:handling_qualities:lateral:derivatives:Cn:yawrate"] = Cn_r

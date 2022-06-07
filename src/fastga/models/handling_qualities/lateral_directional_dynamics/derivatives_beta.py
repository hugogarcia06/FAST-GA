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
class BetaDerivatives(FigureDigitization2):
    """
    Computes de stability sideslip angle (beta) derivatives of the CY, Cl (roll coefficient) and Cn (yaw coefficient).
    The expressions used for the computation of the different coefficients were obtained from
    Roskam - Estimating Stability and Control Derivatives.
    """

    def setup(self):

        self.add_input("data:handling_qualities:lateral:derivatives:wing:CY:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:fuselage:CY:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", val=np.nan, units="rad**-1")

        self.add_input("data:handling_qualities:lateral:derivatives:wing-body:Cl:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:horizontal_tail:Cl:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:vertical_tail:Cl:beta", val=np.nan, units="rad**-1")

        self.add_input("data:handling_qualities:lateral:derivatives:wing:Cn:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:fuselage:Cn:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:vertical_tail:Cn:beta", val=np.nan, units="rad**-1")


        self.add_output("data:handling_qualities:lateral:derivatives:CY:beta", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:Cl:beta", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:Cn:beta", units="rad**-1")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # ------ Lateral Force Coefficient Derivative CY_beta ------
        # The coefficient is computed as the contribution of the wing, body and vertical tail
        CY_beta_W = inputs["data:handling_qualities:lateral:derivatives:wing:CY:beta"]
        CY_beta_B = inputs["data:handling_qualities:lateral:derivatives:fuselage:CY:beta"]
        CY_beta_V = inputs["data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta"]

        CY_beta = CY_beta_W + CY_beta_B + CY_beta_V

        # ------ Rolling Moment Derivative Cl_beta ------
        # Wing-body contribution to dihedral effect
        Cl_beta_WB = inputs["data:handling_qualities:lateral:derivatives:wing-body:Cl:beta"]
        # Horizontal tail contribution to dihedral effect
        Cl_beta_H = inputs["data:handling_qualities:lateral:derivatives:horizontal_tail:Cl:beta"]
        # Vertical tail contribution
        Cl_beta_V = inputs["data:handling_qualities:lateral:derivatives:vertical_tail:Cl:beta"]

        # Cl_beta = Cl_beta_WB + Cl_beta_H + Cl_beta_V
        # The USAF DATCOM neglects the contribution of the horizontal tail to the Cl_beta derivative
        Cl_beta = Cl_beta_WB + Cl_beta_V

        # ------ Yawing Moment Coefficient Derivative Cn_beta ------
        # The wing contribution is very small except at high angles of attack. It is usually conservative to neglect it.
        Cn_beta_W = inputs["data:handling_qualities:lateral:derivatives:wing:Cn:beta"]
        Cn_beta_B = inputs["data:handling_qualities:lateral:derivatives:fuselage:Cn:beta"]
        Cn_beta_V = inputs["data:handling_qualities:lateral:derivatives:vertical_tail:Cn:beta"]
        Cn_beta = Cn_beta_W + Cn_beta_B + Cn_beta_V

        # TODO: contribution of the propeller?

        outputs["data:handling_qualities:lateral:derivatives:CY:beta"] = CY_beta
        outputs["data:handling_qualities:lateral:derivatives:Cl:beta"] = Cl_beta
        outputs["data:handling_qualities:lateral:derivatives:Cn:beta"] = Cn_beta


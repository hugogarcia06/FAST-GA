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
from scipy import interpolate

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2
from fastga.models.handling_qualities.utils.lift_curve_slope import get_lift_curve_slope
from fastga.models.handling_qualities.utils.dihedral_effect import get_dihedral_effect_surface


# TODO: register class
class CYBetaWing(FigureDigitization2):
    # TODO: documentation
    """
    Class to compute the contribution of the wing to the side force coefficient due to sideslip.
    Expressions obtained from Roskam - Airplane Design Part VI and USAF DATCOM.
    """

    def setup(self):
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="deg")

        self.add_output("data:handling_qualities:lateral:derivatives:wing:CY:beta", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Wing Geometry
        dihedral_W = float(inputs["data:geometry:wing:dihedral"])  # Angle in degrees

        # The contribution of the wing/body is normally difficult to compute and sometimes can be supposed to be
        # much lower than the vertical tail contribution.

        CY_beta_W = -0.0001 * abs(dihedral_W) * 57.3  # dihedral angle in degrees.

        outputs["data:handling_qualities:lateral:derivatives:wing:CY:beta"] = CY_beta_W

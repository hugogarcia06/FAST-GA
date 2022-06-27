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

from fastga.models.aerodynamics.components.compute_cnbeta_fuselage import ComputeCnBetaFuselage
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.fuselage.cY_beta_body import CYBetaBody
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.fuselage.cn_beta_body import CnBetaBody


# TODO: Register Submodel
class ComputeFuselageLateralDirectionalDerivatives(om.Group):
    # TODOC
    """

    """

    def setup(self):
        # This is the method used in the aerodynamics module to compute the cn beta due to the body.
        self.add_subsystem(
            "cn_beta_body_previous", ComputeCnBetaFuselage(), promotes=["*"]
        )

        self.add_subsystem(
            "cn_beta_body", CnBetaBody(), promotes=["*"]
        )

        self.add_subsystem(
            "cY_beta_body", CYBetaBody(), promotes=["*"]
        )


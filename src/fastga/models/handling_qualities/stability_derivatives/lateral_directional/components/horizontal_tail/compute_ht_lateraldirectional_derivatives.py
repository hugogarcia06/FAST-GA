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

from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.horizontal_tail.cl_beta_ht import ClBetaHT
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.horizontal_tail.cl_rollrate_ht import \
    ClRollRateHT


# TODO: Register Submodel
class ComputeHTLateralDirectionalDerivatives(om.Group):
    # TODOC
    """

    """
    def setup(self):
        self.add_subsystem(
            "cl_beta_ht", ClBetaHT(), promotes=["*"]
        )

        self.add_subsystem(
            "cl_rollrate_ht", ClRollRateHT(), promotes=["*"]
        )

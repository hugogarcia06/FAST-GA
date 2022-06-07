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

from fastga.models.handling_qualities.longitudinal_dynamics.components.horizontal_tail.cL_alpha_ht import CLAlphaHT
from fastga.models.handling_qualities.longitudinal_dynamics.components.horizontal_tail.cL_alpharate_ht import \
    CLAlphaRateHT
from fastga.models.handling_qualities.longitudinal_dynamics.components.horizontal_tail.cL_pitchrate_ht import \
    CLPitchRateHT
from fastga.models.handling_qualities.longitudinal_dynamics.components.horizontal_tail.cm_alpharate_ht import \
    CmAlphaRateHT
from fastga.models.handling_qualities.longitudinal_dynamics.components.horizontal_tail.cm_pitchrate_ht import \
    CmPitchRateHT


# TODO: Register Submodel
class ComputeHTLongitudinalDerivatives(om.Group):
    # TODOC
    """

    """

    def setup(self):
        self.add_subsystem(
            "cL_alpha_ht", CLAlphaHT, promotes=["*"]
        )
        self.add_subsystem(
            "cL_alpharate_ht", CLAlphaRateHT(), promotes=["*"]
        )
        self.add_subsystem(
            "cL_pitchrate_ht", CLPitchRateHT(), promotes=["*"]
        )
        self.add_subsystem(
            "cm_alpharate_ht", CmAlphaRateHT(), promotes=["*"]
        )
        self.add_subsystem(
            "cm_pitchrate_ht", CmPitchRateHT(), promotes=["*"]
        )

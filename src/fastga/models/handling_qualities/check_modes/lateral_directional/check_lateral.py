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

from openmdao.api import Group

from fastga.models.handling_qualities.check_modes.lateral_directional.check_dutch_roll import CheckDutchRoll
from fastga.models.handling_qualities.check_modes.lateral_directional.check_roll_mode import CheckRollMode
from fastga.models.handling_qualities.check_modes.lateral_directional.check_spiral_mode import CheckSpiralMode


class CheckLateral(Group):

    def initialize(self):
        """Definition of the options of the group"""

    def setup(self):
        self.add_subsystem(
            "check_dutch_roll",
            CheckDutchRoll(),
            promotes=["*"],
        )

        self.add_subsystem(
            "check_roll_mode",
            CheckRollMode(),
            promotes=["*"],
        )

        self.add_subsystem(
            "check_spiral_mode",
            CheckSpiralMode(),
            promotes=["*"],
        )

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

from fastga.models.handling_qualities.check_modes.longitudinal.check_phugoid import CheckPhugoid
from fastga.models.handling_qualities.check_modes.longitudinal.check_short_period import CheckShortPeriod


class CheckLongitudinal(Group):

    def initialize(self):
        """Definition of the options of the group"""
        self.options.declare("result_folder_path", default="", types=str)

    def setup(self):
        self.add_subsystem(
            "check_phugoid",
            CheckPhugoid(
                result_folder_path=self.options["result_folder_path"]
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "check_short_period",
            CheckShortPeriod(
                result_folder_path=self.options["result_folder_path"]
            ),
            promotes=["*"],
        )


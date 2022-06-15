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

from fastga.models.handling_qualities.longitudinal_dynamics.components.thrust.cX_speed_thrust import CXSpeedThrust
from fastga.models.handling_qualities.longitudinal_dynamics.components.thrust.cm_alpha_thrust import CmAlphaThrust
from fastga.models.handling_qualities.longitudinal_dynamics.components.thrust.cm_speed_thrust import CmSpeedThrust


class ComputeThrustLongitudinalDerivatives(om.Group):
    """
    Computes the longitudinal stability derivatives associated to thrust via the semi-empirical methods
    found in the DATCOM.
    """

    def setup(self):
        self.add_subsystem(
            "cm_alpha_thrust", CmAlphaThrust(), promotes=["*"]
        )

        self.add_subsystem(
            "cm_speed_thrust", CmSpeedThrust(), promotes=["*"]
        )

        self.add_subsystem(
            "cX_speed_thrust", CXSpeedThrust(), promotes=["*"]
        )
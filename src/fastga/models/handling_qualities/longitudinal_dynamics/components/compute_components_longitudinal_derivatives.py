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

from fastga.models.handling_qualities.longitudinal_dynamics.components.horizontal_tail.compute_ht_longitudinal_derivatives import \
    ComputeHTLongitudinalDerivatives
from fastga.models.handling_qualities.longitudinal_dynamics.components.thrust.compute_thrust_longitudinal_derivatives import \
    ComputeThrustLongitudinalDerivatives
from fastga.models.handling_qualities.longitudinal_dynamics.components.wing.compute_wing_longitudinal_derivatives import \
    ComputeWingLongitudinalDerivatives


# TODO: Register Submodel
class ComputeComponentsLongitudinalDerivatives(om.Group):
    # TODOC
    """

    """

    def setup(self):
        self.add_subsystem(
            "ht_longitudinal_derivatives", ComputeHTLongitudinalDerivatives(), promotes=["*"]
        )

        self.add_subsystem(
            "wing_longitudinal_derivatives", ComputeWingLongitudinalDerivatives(), promotes=["*"]
        )

        self.add_subsystem(
            "thrust_longitudinal_derivatives", ComputeThrustLongitudinalDerivatives(), promotes=["*"]
        )

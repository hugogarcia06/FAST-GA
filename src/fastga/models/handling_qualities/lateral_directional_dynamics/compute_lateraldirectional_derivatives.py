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

from fastga.models.handling_qualities.lateral_directional_dynamics.components.fuselage.compute_lateraldirectional_fuselage_derivatives import \
    ComputeFuselageLateralDirectionalDerivatives
from fastga.models.handling_qualities.lateral_directional_dynamics.components.horizontal_tail.compute_ht_lateraldirectional_derivatives import \
    ComputeHTLateralDirectionalDerivatives
from fastga.models.handling_qualities.lateral_directional_dynamics.components.vertical_tail.compute_vt_lateraldirectional_derivatives import \
    ComputeVTLateralDirectionalDerivatives
from fastga.models.handling_qualities.lateral_directional_dynamics.components.wing.compute_wing_lateraldirectional_derivatives import \
    ComputeWingLateralDirectionalDerivatives

from fastga.models.handling_qualities.lateral_directional_dynamics.derivatives_beta import BetaDerivatives
from fastga.models.handling_qualities.lateral_directional_dynamics.derivatives_rollrate import RollRateDerivatives
from fastga.models.handling_qualities.lateral_directional_dynamics.derivatives_yawrate import YawRateDerivatives


# TODO: Register Submodel
class ComputeComponentsLateralDirectionalDerivatives(om.Group):

    def setup(self):
        self.add_subsystem(
            "compute_fuselage_lateraldirectional_derivatives", ComputeFuselageLateralDirectionalDerivatives(),
            promotes=["*"]
        )

        self.add_subsystem(
            "compute_ht_lateraldirectional_derivatives", ComputeHTLateralDirectionalDerivatives(),
            promotes=["*"]
        )

        self.add_subsystem(
            "compute_vt_lateraldirectional_derivatives", ComputeVTLateralDirectionalDerivatives(),
            promotes=["*"]
        )

        self.add_subsystem(
            "compute_wing_lateraldirectional_derivatives", ComputeWingLateralDirectionalDerivatives(),
            promotes=["*"]
        )


# TODO: Register Submodel
class ComputeLateralDirectionalDerivatives(om.Group):

    def setup(self):
        self.add_subsystem(
            "components_lateraldirectional_derivatives", ComputeComponentsLateralDirectionalDerivatives(),
            promotes=["*"]
        )

        self.add_subsystem(
            "beta_derivatives", BetaDerivatives(), promotes=["*"]
        )

        self.add_subsystem(
            "rollrate_derivatives", RollRateDerivatives(), promotes=["*"]
        )

        self.add_subsystem(
            "yawrate_derivatives", YawRateDerivatives(), promotes=["*"]
        )

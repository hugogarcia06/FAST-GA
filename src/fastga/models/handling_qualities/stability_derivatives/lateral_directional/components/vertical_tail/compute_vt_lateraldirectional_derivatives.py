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

from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.vertical_tail.cY_beta_vt import CYBetaVT
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.vertical_tail.cY_rollrate_vt import \
    CYRollRateVT
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.vertical_tail.cY_yawrate_vt import \
    CYYawRateVT
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.vertical_tail.cl_beta_vt import ClBetaVT
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.vertical_tail.cl_rollrate_vt import \
    ClRollRateVT
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.vertical_tail.cl_yawrate_vt import \
    ClYawRateVT
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.vertical_tail.cn_beta_vt import CnBetaVT
from fastga.models.handling_qualities.lateral_directional_dynamics import \
    CnRollRateVT
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.vertical_tail.cn_yawrate_vt import \
    CnYawRateVT


# TODO: Register Submodel
class ComputeVTLateralDirectionalDerivatives(om.Group):
    # TODOC
    """

    """

    def setup(self):
        self.add_subsystem(
            "cl_beta_vt", ClBetaVT(), promotes=["*"]
        )

        self.add_subsystem(
            "cl_rollrate_vt", ClRollRateVT(), promotes=["*"]
        )

        self.add_subsystem(
            "cl_yawrate_vt", ClYawRateVT(), promotes=["*"]
        )

        self.add_subsystem(
            "cn_beta_vt", CnBetaVT(), promotes=["*"]
        )

        self.add_subsystem(
            "cn_rollrate_vt", CnRollRateVT(), promotes=["*"]
        )

        self.add_subsystem(
            "cn_yawrate_vt", CnYawRateVT(), promotes=["*"]
        )

        self.add_subsystem(
            "cY_beta_vt", CYBetaVT(), promotes=["*"]
        )

        self.add_subsystem(
            "cY_rollrate_vt", CYRollRateVT(), promotes=["*"]
        )

        self.add_subsystem(
            "cY_yawrate_vt", CYYawRateVT(), promotes=["*"]
        )

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


# TODO: Register Submodel
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.wing.cY_beta_wing import CYBetaWing
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.wing.cY_rollrate_wing import \
    CYRollRateWing
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.wing.cl_beta_wingbody import \
    ClBetaWingBody
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.wing.cl_rollrate_wing import \
    ClRollRateWing
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.wing.cl_yawrate_wing import ClYawRateWing
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.wing.cn_beta_wing import CnBetaWing
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.wing.cn_rollrate_wing import \
    CnRollRateWing
from fastga.models.handling_qualities.stability_derivatives.lateral_directional.components.wing.cn_yawrate_wing import CnYawRateWing


class ComputeWingLateralDirectionalDerivatives(om.Group):
    # TODOC
    """

    """

    def setup(self):
        self.add_subsystem(
            "cl_beta_wingbody", ClBetaWingBody(), promotes=["*"]
        )

        self.add_subsystem(
            "cl_rollrate_wing", ClRollRateWing(), promotes=["*"]
        )

        self.add_subsystem(
            "cl_yawrate_wing", ClYawRateWing(), promotes=["*"]
        )

        self.add_subsystem(
            "cn_beta_wing", CnBetaWing(), promotes=["*"]
        )

        self.add_subsystem(
            "cn_rollrate_wing", CnRollRateWing(), promotes=["*"]
        )

        self.add_subsystem(
            "cn_yawrate_wing", CnYawRateWing(), promotes=["*"]
        )

        self.add_subsystem(
            "cY_beta_wing", CYBetaWing(), promotes=["*"]
        )

        self.add_subsystem(
            "cY_rollrate_wing", CYRollRateWing(), promotes=["*"]
        )


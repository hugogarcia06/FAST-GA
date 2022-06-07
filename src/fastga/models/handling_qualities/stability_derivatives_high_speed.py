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


from fastga.models.handling_qualities.external.openvsp import ComputeSTABopenvsp


# TODO: Register module
from fastga.models.handling_qualities.lateral_directional_dynamics.compute_lateraldirectional_derivatives import \
    ComputeLateralDirectionalDerivatives

from fastga.models.handling_qualities.longitudinal_dynamics.compute_longitudinal_derivatives import \
    ComputeLongitudinalDerivatives


class StabilityDerivativesHighSpeed(Group):
    """Model for high speed stability computation. """

    def initialize(self):
        self.options.declare("use_openvsp", default=True, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("wing_airfoil", default="naca23012.af", types=str, allow_none=True)
        self.options.declare("htp_airfoil", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("vtp_airfoil", default="naca0012.af", types=str, allow_none=True)

    def setup(self):
        if self.options["use_openvsp"]:
            # Use of OpenSVP for stability calculation
            self.add_subsystem(
                "stab_openvsp",
                ComputeSTABopenvsp(
                    result_folder_path=self.options["result_folder_path"],
                    openvsp_exe_path=self.options["openvsp_exe_path"],
                    wing_airfoil_file=self.options["wing_airfoil"],
                    htp_airfoil_file=self.options["htp_airfoil"],
                    vtp_airfoil_file=self.options["vtp_airfoil"],
                ),
                promotes=["*"],
            )

        else:
            # Use of semi-empirical expressions from DATCOM for stability calculation
            self.add_subsystem(
                "compute_lateral-directional_derivatives",
                ComputeLateralDirectionalDerivatives(),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_longitudinal_derivatives",
                ComputeLongitudinalDerivatives(),
                promotes=["*"],
            )

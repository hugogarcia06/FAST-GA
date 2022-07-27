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

from fastga.models.handling_qualities.check_modes.lateral_directional.check_lateral import CheckLateral
from fastga.models.handling_qualities.check_modes.longitudinal.check_longitudinal import CheckLongitudinal
from ..aircraft_modes.aircraft_modes_computation import AircraftModesComputation


class AircraftModesAnalysis(Group):
    """
    Analysis the aircraft's modes characteristics according to handling qualities regulation.
    """

    def initialize(self):
        """Definition of the options of the group"""
        self.options.declare("airplane_file", default="", types=str)
        self.options.declare("use_openvsp", default=True, types=bool)
        self.options.declare("add_fuselage", default=False, types=bool)
        self.options.declare("reference_flight_condition", default={}, types=dict)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("result_folder_path", default="", types=str, allow_none=True)
        self.options.declare("wing_airfoil", default="naca23012.af", types=str, allow_none=True)
        self.options.declare("htp_airfoil", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("vtp_airfoil", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("plot_modes", default=False, types=bool)

    def setup(self):
        # Compute aircraft modes
        self.add_subsystem(
            "aircraft_modes_computation",
            AircraftModesComputation(
                airplane_file=self.options["airplane_file"],
                use_openvsp=self.options["use_openvsp"],
                add_fuselage=self.options["add_fuselage"],
                reference_flight_condition=self.options["reference_flight_condition"],
                result_folder_path=self.options["result_folder_path"],
                openvsp_exe_path=self.options["openvsp_exe_path"],
                wing_airfoil=self.options["wing_airfoil"],
                htp_airfoil=self.options["htp_airfoil"],
                vtp_airfoil=self.options["vtp_airfoil"],
                plot_modes=self.options["plot_modes"],
            ),
            promotes=["*"],
        )


        # Check longitudinal dynamics
        self.add_subsystem(
            "check_longitudinal",
            CheckLongitudinal(
                result_folder_path=self.options["result_folder_path"],
            ),
            promotes=["*"],
        )

        # Check lateral-directional dynamics
        self.add_subsystem(
            "check_lateral",
            CheckLateral(
                result_folder_path=self.options["result_folder_path"],
            ),
            promotes=["*"],
        )


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

from .lateral_directional_spacestate import LateralDirectionalSpaceStateMatrix
from .longitudinal_spacestate import LongitudinalSpaceStateMatrix
from .plot_aircraft_modes import PlotAircraftModes
from ..stability_derivatives.stability_derivatives import StabilityDerivatives


class AircraftModesComputation(Group):
    """
    Computes the characteristics of the longitudinal and lateral-directional modes: dumping ratio, natural frequency...
    """

    def initialize(self):
        """Definition of the options of the group"""
        self.options.declare("airplane_file", default="", types=str)
        self.options.declare("use_openvsp", default=True, types=bool)
        self.options.declare("reference_flight_condition", default={}, types=dict)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("result_folder_path", default="", types=str, allow_none=True)
        self.options.declare("wing_airfoil", default="naca23012.af", types=str, allow_none=True)
        self.options.declare("htp_airfoil", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("vtp_airfoil", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("add_fuselage", default=False, types=bool, allow_none=False)
        self.options.declare("plot_modes", default=False, types=bool)

    def setup(self):
        # Compute stability derivatives
        self.add_subsystem(
            "stability_derivatives",
            StabilityDerivatives(
                airplane_file=self.options["airplane_file"],
                use_openvsp=self.options["use_openvsp"],
                reference_flight_condition=self.options["reference_flight_condition"],
                result_folder_path=self.options["result_folder_path"],
                openvsp_exe_path=self.options["openvsp_exe_path"],
                wing_airfoil=self.options["wing_airfoil"],
                htp_airfoil=self.options["htp_airfoil"],
                vtp_airfoil=self.options["vtp_airfoil"],
                add_fuselage=self.options["add_fuselage"]
            ),
            promotes=["*"],
        )

        # Compute longitudinal modes
        self.add_subsystem(
            "longitudinal_modes",
            LongitudinalSpaceStateMatrix(),
            promotes=["*"],
        )

        # Compute lateral-directional modes
        self.add_subsystem(
            "lateral_modes",
            LateralDirectionalSpaceStateMatrix(),
            promotes=["*"],
        )

        # Plot all the aircraft modes in the s-plane
        if self.options["plot_modes"] is True:
            self.add_subsystem(
                "plot_aircraft_modes",
                PlotAircraftModes(
                    result_folder_path=self.options["result_folder_path"]
                ),
                promotes=["*"],
            )



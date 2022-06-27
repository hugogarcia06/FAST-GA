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

from fastga.models.handling_qualities.reference_flight_condition import ReferenceFlightCondition
from fastga.models.handling_qualities.stability_derivatives_high_speed import StabilityDerivativesHighSpeed
from fastga.models.handling_qualities.utils.missing_geometry import MissingGeometry


# TODO: Register module
class StabilityDerivatives(Group):
    """
    Computes the aerodynamic stability coefficients of the aircraft for high or low speed via OpenVSP or with semi-empirical methods from
    USAF DATCOM.
    """
    # NOTE: for the moment only high speed is implemented

    def initialize(self):
        """Definition of the options of the group"""
        self.options.declare("use_openvsp", default=True, types=bool)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("result_folder_path", default="", types=str, allow_none=True)
        self.options.declare("wing_airfoil", default="naca23012.af", types=str, allow_none=True)
        self.options.declare("htp_airfoil", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("vtp_airfoil", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("add_fuselage", default=False, types=bool, allow_none=False)


    def setup(self):
        """
        Add the method to compute the stability derivatives
        """
        # Set Flight Reference Condition
        self.add_subsystem(
            "reference_flight_condition",
            ReferenceFlightCondition(),
            promotes=["*"],
        )
        # Compute missing geometry
        self.add_subsystem(
            "missing_geometry",
            MissingGeometry(),
            promotes=["*"],
        )
        # Compute cruise (high speed) characteristics
        self.add_subsystem(
            "stab_high_speed",
            StabilityDerivativesHighSpeed(
                use_openvsp=self.options["use_openvsp"],
                result_folder_path=self.options["result_folder_path"],
                openvsp_exe_path=self.options["openvsp_exe_path"],
                wing_airfoil=self.options["wing_airfoil"],
                htp_airfoil=self.options["htp_airfoil"],
                vtp_airfoil=self.options["vtp_airfoil"],
                add_fuselage=self.options["add_fuselage"]
            ),
            promotes=["*"],
        )

        # Here we could add a component to compute low speed stability coefficients.

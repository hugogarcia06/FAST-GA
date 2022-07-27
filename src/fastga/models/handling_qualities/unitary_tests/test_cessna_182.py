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

import pytest
import tempfile
import os
import os.path as pth
from pathlib import Path


from .test_functions import check_aircraft_modes, check_aircraft_modes_modified, aircraft_modes
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from ..check_modes.aircraft_modes_analysis import AircraftModesAnalysis
from tempfile import TemporaryDirectory

XML_FILE = "cessna_182.xml"


def _create_tmp_directory() -> TemporaryDirectory:
    """Provide temporary directory for calculation!"""
    for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
        if tmp_base_path is not None:
            os.makedirs(tmp_base_path, exist_ok=True)
        tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
        break

    return tmp_directory


def test_aircraft_modes():
    """
    Testing the module for obtaining the aircraft's modes.
    """

    add_fuselage = False
    use_openvsp = True
    XML_FILE = "cessna_182.xml"

    aircraft_modes(
        add_fuselage,
        use_openvsp,
        XML_FILE
    )


def test_check_modes_complete():
    """
    For testing the complete modes analysis, from calculating the stability derivatives, the aircraft's modes and
    checking them according to hte regulations.
    """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    add_fuselage = False
    use_openvsp = True

    ivc = get_indep_var_comp(
        list_inputs(AircraftModesAnalysis(
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        )), __file__, XML_FILE
    )

    # ivc.add_output("data:geometry:wing:MAC:length", val=4.9, units="ft")

    check_aircraft_modes_modified(
        add_fuselage,
        use_openvsp,
        XML_FILE,
        ivc
    )

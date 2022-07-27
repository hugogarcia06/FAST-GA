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

from fastga.models.handling_qualities.unitary_tests.test_functions import \
    dynamic_tail_sizing_loop, short_period_tail_sizing

XML_FILE = "cirrus_sr22_no_ht_area.xml"


def test_dynamic_tail_sizing_loop():

    add_fuselage = False
    use_openvsp = True

    dynamic_tail_sizing_loop(add_fuselage, use_openvsp, XML_FILE)


def test_short_period_tail_sizing():
    add_fuselage = False
    use_openvsp = True

    short_period_tail_sizing(add_fuselage, use_openvsp, XML_FILE="beechcraft_76_no_ht_area.xml")



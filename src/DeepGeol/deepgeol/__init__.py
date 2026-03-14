  # coding: utf-8 -*-
#
# This file is part of DeepGeol.
#
# DeepGeol is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# DeepGeol is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DeepGeol.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2024-2025 Anthony Larcher
"""


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2024-2025 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = "reS"


from .unet_parts import DoubleConv
from .unet_parts import Down2C
from .unet_parts import Up2C
from .unet_parts import OutConv

from .unet import UNet


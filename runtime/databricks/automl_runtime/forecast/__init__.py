#
# Copyright (C) 2022 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

WEEK_OFFSET = 'W'

OFFSET_ALIAS_MAP = {
    "w": WEEK_OFFSET,
    "W": WEEK_OFFSET,
    "week": WEEK_OFFSET,
    "weeks": WEEK_OFFSET,
    "d": "D",
    "D": "D",
    "days": "D",
    "day": "D",
    "hours": "H",
    "hour": "H",
    "hr": "H",
    "h": "H",
    "H": "H",
    "m": "min",
    "minute": "min",
    "min": "min",
    "minutes": "min",
    "T": "min",
    "S": "S",
    "seconds": "S",
    "sec": "S",
    "second": "S"
}

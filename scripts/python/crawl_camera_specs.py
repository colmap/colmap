# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

import re
import argparse
import requests
from lxml.html import soupparser


MAX_REQUEST_TRIALS = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib_path", required=True)
    args = parser.parse_args()
    return args


def request_trial(func, *args, **kwargs):
    for i in range(MAX_REQUEST_TRIALS):
        try:
            response = func(*args, **kwargs)
        except:
            continue
        else:
            return response

    raise SystemError


def main():
    args = parse_args()

    ##########################################################################
    # Header file
    ##########################################################################

    with open(args.lib_path + ".h", "w") as f:
        f.write("#include <vector>\n")
        f.write("#include <string>\n")
        f.write("#include <unordered_map>\n\n")
        f.write("// { make1 : ({ model1 : sensor-width in mm }, ...), ... }\n")
        f.write("typedef std::vector<std::pair<std::string, float>> make_specs_t;\n")
        f.write("typedef std::unordered_map<std::string, make_specs_t> camera_specs_t;;\n\n")
        f.write("camera_specs_t InitializeCameraSpecs();\n\n")

    ##########################################################################
    # Source file
    ##########################################################################

    makes_response = requests.get("http://www.digicamdb.com")
    makes_tree = soupparser.fromstring(makes_response.text)
    makes_node = makes_tree.find(".//select[@id=\"select_brand\"]")
    makes = [b.attrib["value"] for b in makes_node.iter("option")]

    with open(args.lib_path + ".cc", "w") as f:
        f.write("camera_specs_t InitializeCameraSpecs() {\n")
        f.write("  camera_specs_t specs;\n\n")
        for make in makes:
            f.write("  {\n")
            f.write("    auto& make_specs = specs[\"%s\"];\n" % make.lower().replace(" ", ""))

            models_response = request_trial(
                requests.post,
                "http://www.digicamdb.com/inc/ajax.php",
                data={"b": make, "role": "header_search"})

            models_tree = soupparser.fromstring(models_response.text)
            models_code = ""
            num_models = 0
            for model_node in models_tree.iter("option"):
                model = model_node.attrib.get("value")
                model_name = model_node.text
                if model is None:
                    continue

                url = "http://www.digicamdb.com/specs/{0}_{1}" \
                                            .format(make, model)
                specs_response = request_trial(requests.get, url)

                specs_tree = soupparser.fromstring(specs_response.text)
                for spec in specs_tree.findall(".//td[@class=\"info_key\"]"):
                    if spec.text.strip() == "Sensor:":
                        sensor_text = spec.find("..").find("./td[@class=\"bold\"]")
                        sensor_text = sensor_text.text.strip()
                        m = re.match(".*?([\d.]+) x ([\d.]+).*?", sensor_text)
                        sensor_width = m.group(1)
                        data = (model_name.lower().replace(" ", ""),
                                float(sensor_width.replace(" ", "")))
                        models_code += "    make_specs.emplace_back(\"%s\", %.4ff);\n" % data

                        print(make, model_name)
                        print("   ", sensor_text)

                        num_models += 1

            f.write("    make_specs.reserve(%d);\n" % num_models)
            f.write(models_code)
            f.write("  }\n\n")

        f.write("  return specs;\n")
        f.write("}\n")


if __name__ == "__main__":
    main()

from __future__ import print_function
from lxml import etree
import h5py


def xml2parameters(params, prefix, idx):
    print("Prefix is", prefix, "and idx is", idx)
    params = {}
    tree = etree.parse(prefix + ".task{i}.in.xml".format(i=idx))
    root = tree.getroot()
    for element in root.findall("PARAMETERS")[0].findall("PARAMETER"):
        attribute = element.values()[0]
        value = element.text
        print(attribute, " = ", value)
        params[attribute] = value

    return params


def parameters2hdf5(params, filename):
    f = h5py.File(filename, "a")

    try:
        f.create_group("parameters")
    except:
        print("Could not create group parameters")
        
    for k, v in params.items():
        if k in f["parameters"].keys():
            print("Key", k, "exists in file")
            raise
        f["parameters/{k}".format(k=k)] = str(v)

    f.close()

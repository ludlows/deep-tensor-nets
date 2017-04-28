
import logging
import os
import sys
import numpy as np
import tarfile
import tempfile
import shutil

from ase.db import connect
from ase.io.extxyz import read_xyz
from ase.units import Hartree, eV, Bohr, Ang



def generate_gdb9_db(dbpath):
    father_path = os.path.split(dbpath)[0]
    print('father_path is {}'.format(father_path))
    logging.info('Generating DB files...')
    tar_path = os.path.join(father_path, 'gdb9.tar.gz')
    raw_path = os.path.join(father_path, 'gdb9_xyz')
    
    tar = tarfile.open(tar_path)
    tar.extractall(raw_path)
    tar.close()

    prop_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo',
                  'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    conversions = [1., 1., 1., 1., Bohr ** 3 / Ang ** 3,
                   Hartree / eV, Hartree / eV, Hartree / eV,
                   Bohr ** 2 / Ang ** 2, Hartree / eV,
                   Hartree / eV, Hartree / eV, Hartree / eV,
                   Hartree / eV, 1.]
    tmpdir = tempfile.mkdtemp('gdb9')
    logging.info('Parse xyz files...')
    with connect(dbpath) as con:
        for i, xyzfile in enumerate(os.listdir(raw_path)):
            xyzfile = os.path.join(raw_path, xyzfile)

            if i % 10000 == 0:
                logging.info('Parsed: ' + str(i) + ' / 133885')
            properties = {}
            tmp = os.path.join(tmpdir, 'tmp.xyz')

            with open(xyzfile, 'r') as f:
                lines = f.readlines()
                l = lines[1].split()[2:]
                for pn, p, c in zip(prop_names, l, conversions):
                    properties[pn] = float(p) * c
                with open(tmp, "wt") as fout:
                    for line in lines:
                        fout.write(line.replace('*^', 'e'))

            with open(tmp, 'r') as f:
                ats = list(read_xyz(f, 0))[0]

            con.write(ats, key_value_pairs=properties)
    logging.info('INFO-- remove temp directory {}'.format(raw_path))
    
    shutil.rmtree(raw_path)

    logging.info('Done.')

    return True

def generate_atomref(at_path):
    logging.info('Downloading GDB-9 atom references...')
    print('atomrefs.txt path is {}'.format(at_path))
    father_path = os.path.split(at_path)[0]
    print('father path of atomrefs.txt is {}'.format(father_path))
    tmp_path = os.path.join(father_path, 'atomrefs.txt')
    atref = np.zeros((100, 6))
    labels = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']
    with open(tmp_path) as f:
        lines = f.readlines()
        for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
            atref[z, 0] = float(l.split()[1])
            atref[z, 1] = float(l.split()[2]) * Hartree / eV
            atref[z, 2] = float(l.split()[3]) * Hartree / eV
            atref[z, 3] = float(l.split()[4]) * Hartree / eV
            atref[z, 4] = float(l.split()[5]) * Hartree / eV
            atref[z, 5] = float(l.split()[6])
    np.savez(at_path, atom_ref=atref, labels=labels)
    return True
    
if __name__ == "__main__":
	logging.info('make sure you have download the file gdb9.tar.gz and atomref.txt in data_dir directory.')
    dbpath = os.path.join('data_dir', 'gdb9.db')
    at_path = os.path.join('data_dir', 'atom_refs.npz')

    res1 = generate_gdb9_db(dbpath)
    res2 = generate_atomref(at_path)
    if not all([res1,res2]):
    	logging.info('ERROR, cannot generate file')
    	sys.exit(1)





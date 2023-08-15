"""
    Helper functions for maintaining the library files.
"""

import os
import zipfile

num_states = [1,2,3,4,5,6]
base_fn = '%s.hs'

def zip_files():
    for n in num_states:
        fn = base_fn % n
        zf = zipfile.ZipFile(fn + '.zip', 'w', zipfile.ZIP_DEFLATED)
        zf.write(fn)
        zf.close()

def unzip_files():
    for n in num_states:
        fn = base_fn % n
        zf = zipfile.ZipFile(fn + '.zip', 'r')
        nf = open(fn, 'w')
        data = zf.read(fn)
        nf.write(data)
        nf.close()

def remove_unzipped():
    for n in num_states:
        fn = base_fn % n
        os.remove(fn)


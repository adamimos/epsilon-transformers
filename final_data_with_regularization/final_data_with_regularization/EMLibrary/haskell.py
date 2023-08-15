import os,sys
import tempfile

import cmpy
import cmpy.machines

from cmpy.exceptions import CMPyException
from cmpy.util import deprecate

__all__ = ['haskell_machine',
           'IndexedMachiness',
           'machines',
           'topological_eMs']

def haskell_to_RecurrentEpsilonMachine(edges, element):
    machine = cmpy.machines.RecurrentEpsilonMachine()
    for from_node, symbol, to_node in edges:
        machine.add_edge(from_node, to_node, probability=1, output=str(symbol))

    name = '%s-State Machine '
    if cmpy.cmpyParams['text.usetex']:
        name += '\#%s'
    else:
        name += '#%s'
    machine.set_name( name % (len(machine), element) )
    machine.graph['godel_number'] = element
    machine.renormalize_edges()

    return machine

def haskell_machine(indexed_file, index):
    line = indexed_file.get_raw_text(index)
    splits = line.split(' ')
    edges = eval(splits[2])
    m = haskell_to_RecurrentEpsilonMachine(edges, index)
    return m

def read_machines(filename):
    machines = []
    fh = open(filename, 'rU')

    count = 0
    for line in fh:
        count += 1
        splits = line.split(' ')
        start_nodes = eval(splits[1])
        edges = eval(splits[2])
        # Properly deal with the last ']'
        i = splits[3].index(']')
        accept_nodes = eval(splits[3][:i+1])

        # A MealyHMM has all states as both start and accept states.
        m = haskell_to_RecurrentEpsilonMachine(edges, count)
        machines.append(m)

    return machines

class IndexedMachines(object):
    def __init__(self, myfile):
        self.file = myfile
        self.temp_fn = None
        self.fh = None

        self.load_archive()
        self.load_index()

    def __del__(self):
        if self.fh is not None:
            self.fh.close()
        if self.temp_fn is not None:
            os.remove(self.temp_fn)

    def load_archive(self):
        import zipfile

        try:
            self.fh = open(self.file, 'r')
        except IOError:
            #sys.stdout.write("Unzipping archive...")
            #sys.stdout.flush()
            # we need to create the unzipped file
            zfh = zipfile.ZipFile(self.file + '.zip', 'r')
            fd, fn = tempfile.mkstemp()
            fh = os.fdopen(fd, 'w')
            zfn = os.path.basename(self.file)
            fh.write(zfh.read(zfn))
            fh.close()
            self.fh = open(fn, 'r')
            self.temp_fn = fn
            #sys.stdout.write("done\n")
            #sys.stdout.flush()

    def load_index(self):
        import cPickle

        indexed_fn = self.file + '.index'
        try:
            # load the index file, if it exists
            fh = open(indexed_fn, 'rb')
            self.positions = cPickle.load(fh)
            fh.close()
            #print "Loaded pre-existing index file."
        except:
            #sys.stdout.write("Creating index file...")
            #sys.stdout.flush()
            self.positions = [0]
            while self.fh.readline():
                self.positions.append(self.fh.tell())

            # Writing index file
            try:
                fh = open(indexed_fn, 'w')
                cPickle.dump(self.positions, fh)
                fh.close()
                #sys.stdout.write("done\n")

            except IOError:
                # no permission to create the index file
                #sys.stdout.write("skipping.\n")
                pass

            #sys.stdout.flush()

    def get_raw_text(self, x):
        self.fh.seek(self.positions[x])
        return self.fh.read(self.positions[x+1] - self.positions[x])

    def __getitem__(self, x):
        return haskell_machine(self, x)

    def __len__(self):
        return len(self.positions) - 1

    def machine_iter(self):
        """Yields epsilon machines."""
        for index in xrange(len(self)):
            yield haskell_machine(self, index)

@deprecate('Use cmpy.machines.library(0, n) instead.')
def topological_eMs(n):
    return machines(n)

def machines(n):
    """Returns an IndexedMachines instance for `n` number of states."""
    valid_n = [1,2,3,4,5,6]
    if n not in valid_n:
        raise CMPyException('Enumeration unknown for %s states.' % n)

    curdir = os.path.abspath(os.path.split(__file__)[0])
    fn = "%s.hs" % n

    return IndexedMachines(os.path.join(curdir, fn))


##############################################################################
#
#    Randal C. Burns
#    Department of Computer Science
#    Johns Hopkins University
#
################################################################################

from scipy.sparse import dok_matrix, csc_matrix
from scipy.io import loadmat, savemat
from fiber import Fiber
import zindex
import math
import itertools
from cStringIO import StringIO


#
#  This routine uses two different sparse matrix representations.
#  The calls need to be performed in the following order
#
#     Create FiberGraph
#     Add Edges (on dok format)
#     Complete -- this converts from dictionary of keys format to Column CSC
#     Write or SVD (on csc format)
#


#
#  Sparse matrix representation of a Fiber graph
#
class FiberGraph:

  #
  # Constructor: number of nodes in the graph
  #   convert it to a maximum element
  #
  def __init__(self, matrixdim ):
    
    # Round up to the nearest power of 2
    xdim = int(math.pow(2,math.ceil(math.log(matrixdim[0],2))))
    ydim = int(math.pow(2,math.ceil(math.log(matrixdim[1],2))))
    zdim = int(math.pow(2,math.ceil(math.log(matrixdim[2],2))))

    # Need the dimensions to be the same shape for zindex
    xdim = ydim = zdim = max ( xdim, ydim, zdim )

    # largest value is -1 in each dimension, then plus one because range(10) is 0..9
    self._maxval = zindex.XYZMorton ([xdim-1,ydim-1,zdim-1]) + 1

    # list of seed locations
    self.seeds = {}
 
    # dictionary of keys matrix for one by one insertion
    self.spedgemat = dok_matrix ( (self._maxval, self._maxval), dtype=float )
    # empty CSC matrix
    self.spcscmat = csc_matrix ( (self._maxval, self._maxval), dtype=float )


  #
  # Destructor
  #
  def __del__(self):
    self.spedgemat.clear()


  #
  # Add the vertex (seed voxel) from a fiber to the list of vertices
  #  These are the locations in the graph that we care about.
  #   all other spatial locations are not included in graph generation
  #
  def addVertex ( self, fiber ):
    """Add a fibger to the graph"""
    vertex = fiber.getSeed ()
    # RB testing
    if ( vertex == True ):
      print "Already added this vertex"
    self.seeds [ vertex ] = True
  
  #
  # Add a fiber to the graph.  
  #  This is not idempotent, i.e. if you add the same fiber twice you get a different result
  #  in terms of graph weigths.
  #
  def add ( self, fiber ):
    """Add a fiber to the graph"""

    # Get the set of voxels in the fiber
    allvoxels = fiber.getVoxels ()

    voxels = []
    # Use only the important voxels
    for i in allvoxels:
      if self.seeds.get( i ):
        voxels.append ( i )  

    for v1,v2 in itertools.combinations((voxels),2): 
      if ( v1 < v2 ):  
        self.spedgemat [ v1, v2 ] += 1.0
      else:
        self.spedgemat [ v2, v1 ] += 1.0

  #
  # Complete the graph.  Get it ready for analysis.
  #
  def complete ( self ):
    """Done adding fibers.  Prior to analysis"""

    self.spcscmat = csc_matrix ( self.spedgemat )
    self.spedgemat.clear()

  #
  #  Write the sparse matrix out in a format that can be reingested.
  #  fout should be an open file handle
  #
  def saveToMatlab ( self, key, filename ):
    """Save the sparse array to disk in the specified file name"""

    if 0 == self.spcscmat.getnnz():
      print "Call complete after adding all edges"
      assert 0

    print "Saving key ", key, " to file ", filename
    savemat( filename , {key: self.spcscmat})

  #
  #  Write the sparse matrix out in a format that can be reingested.
  #  fout should be an open file handle
  #
  def loadFromMatlab ( self, key, filename ):
    """Load the sparse array from disk by file name"""

    print "Reading key ", key, " from file ", filename

    # first convert to csc 
    t = loadmat ( filename  )
    self.spcscmat = t[key]

    print self.spcscmat

  #
  #  Write the matrix out in a format that SciDB can import.
  #  fout should be an open file handle
  #  chunkdim specifies the SciDB chunk 
  #
  def writeForSciDB ( self, chunkdim, fout ):
    """Write the graph in a SciDB compatible text format"""

    print "Convert to CSC matrix"
    # first convert to csc 
    spcscmat = csc_matrix ( self.spedgemat )

    # iterate over all of the chunks in 
    # RBTODO range needs to be +1 if mod = 0
    for row in range(self._maxval/chunkdim[0]):
      print "Processing row ", row, row*chunkdim[0]        
      for col in range(self._maxval/chunkdim[1]):
        outstr = StringIO();
        outstr.write ( '[[' )
        spchunk = spcscmat [ col*chunkdim[0]:(col+1)*chunkdim[0]-1, row*chunkdim[1]:(row+1)*chunkdim[1]-1 ]
        spchunknz = spchunk.nonzero()
        for k in range(len(spchunknz[0])):
          outstr.write ( '{' )
          outstr.write ( str(spchunknz[0][k]) )
          outstr.write ( ','  )
          outstr.write ( str(spchunknz[1][k]) )
          outstr.write (  '}('  )
          outstr.write ( str(spchunk[spchunknz[0][k], spchunknz[1][k]]) ) 
          outstr.write (  '),'  )
        outstr.write ( ']]\n' )
       
       # output a chunk if there are elements
        if ( len(spchunknz[0]) != 0 ):
           fout.write ( outstr.getvalue() )


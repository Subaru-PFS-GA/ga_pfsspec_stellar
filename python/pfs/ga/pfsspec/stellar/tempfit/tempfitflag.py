from enum import IntFlag

class TempFitFlag(IntFlag):
    """
    Class to hold the flags for the TempFit algorithm.
    """

    OK = 0                      # No flags
    BADINIT = 1 << 0            # Initial values are outside bounds or other problems
    NOCONVERGE = 1 << 1         # No convergence
    BADCONVERGE = 1 << 2        # Convergence to a bad solution
    MAXITER = 1 << 3            # Maximum iterations reached
    PARAMEDGE = 1 << 4          # Template parameters at the bounds
    BADERROR = 1 << 5           # Errors could not be calculated
    BADCOV = 1 << 6             # Covariance matrix could not be calculated
    UNLIKELYPRIOR = 1 << 7      # Unlikely prior
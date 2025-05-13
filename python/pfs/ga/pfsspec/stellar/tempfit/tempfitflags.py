class TempFitFlags:
    """
    Class to hold the flags for the TempFit algorithm.
    """

    BADINIT = 1 << 0            # Initial values are outside bounds or other problems
    NOCONVERGE = 1 << 1         # No convergence
    BADCONVERGE = 1 << 2        # Convergence to a bad solution
    MAXITER = 1 << 3            # Maximum iterations reached
    RVEDGE = 1 << 4             # RV at the bounds
    PARAMEDGE = 1 << 5          # Template parameters at the bounds
    BADERROR = 1 << 6           # Errors could not be calculated
    BADCOV = 1 << 7             # Covariance matrix could not be calculated
from typing import Annotated, Callable, Literal
import numpy.typing as npt

import numpy as np
import subprocess

# typing hinting for numpy
# using Annotated to specify size
# (why is this so convoluted idk)
FiftyFloats = Annotated[npt.NDArray[np.float32], (50,)]

def generate_sonification(
        pc1, pc2, pc3, 
        pc_mapper: Callable[[float, float, float], FiftyFloats], dest="assets/sonification.wav"):
    """
    def pc_mapper(pc1, pc2, pc3): 
        return param_map_scaler.transform(
            np.array([pc1, pc2, pc3]).reshape(1,3).dot(corr_loadings_matrix.T)
        )
    """
    args = [dest]
    principal_components_mapped_to_tags = pc_mapper(pc1, pc2, pc3)
    args.extend(principal_components_mapped_to_tags)
    call_chuck(args)

def call_chuck(*args):
    params = ":".join([str(s) for s in args[0]])
    results = subprocess.run(f"chuck -s ./sonification.ck:{params}", shell=True, capture_output=True, text=True)
    print("===========")
    print()
    print()
    print()
    print()
    print("calling chuck:")
    print(results.stderr)
    print()
    print()
    print()
    print()
    print("===========")


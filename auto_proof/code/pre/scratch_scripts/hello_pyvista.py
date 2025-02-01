import pyvista as pv
import numpy as np
from pyvista import examples

mesh = examples.load_uniform()

pl = pv.Plotter(shape=(1, 2))
_ = pl.add_mesh(
    mesh, scalars='Spatial Point Data', show_edges=True
)
pl.subplot(0, 1)
_ = pl.add_mesh(
    mesh, scalars='Spatial Cell Data', show_edges=True
)
pl.export_html('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/figures/hello_pv.html')  
# AutoProof
Automated Proofreading for Connectomics

conda install conda-forge::graph-tool
can't install this package with pip

did the opposite and downloaded conda ones first and then downloaded pip only:
caveclient
python -m pip install caveclient

cloud-volume does not work when trying to pip install it after the above packages. So instead I created a separate package when I wanted to add the supervoxels at the l2 node

When switching between conda and pip, its torch vs pytorch

Created an env with conda forge and graph-tool, conda create --name gt -c conda-forge graph-tool and then ran pip install . --ugrade on the requirements. cloud-volume I ran separately with pip install and it didn't work

Tried adding cloud-volume to requirements file, didn't work


Final version:
gt env: First create an env by conda create --name gt -c conda-forge graph-tool
Then run the pip install . --upgrade which should take the default requirements file
Cloud volume env: Create base conda env and then run pip install -r cv_requirements.txt which should have the first requirements file already pulled

Need to run conda install -c conda-forge trame in order to save html file from pyvista

Everything works now I just needed to run conda install pip first so that pip works in the conda env instaed of outside
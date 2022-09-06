import os

#script to execute
script = " \
    cd src/dcn_sim\n \
    python setup.py install\n \
    cd ../gnn_policy\n \
    python setup.py install\n \
    cd ../dc-env\n \
    pip install -e .\n \
"

os.system(script)
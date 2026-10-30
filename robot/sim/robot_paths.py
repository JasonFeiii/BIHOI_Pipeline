"""Runtime locations of the robot stages (retargeting and physics); all of them are ignored by Git.

BIHOI_ROBOT_DATA     inputs and prepared data   (default <repo>/data/robot)
                     raw/ standardized_hoi_30fps exports, retargeted/, tasks/, demo/, mano/ (chumpy-free MANO), manifest.json
BIHOI_ROBOT_OUTPUTS  runs, videos and reviews   (default <repo>/outputs/robot)
HOI_MANO_DIR         operator MANO_{LEFT,RIGHT}.pkl (default <repo>/.models/mano, as for the capture pipeline)
HOI_ENV_ROBOT        Python environment          (default <repo>/.environments/robot)
HOI_EXTERNAL_ROOT    pinned Do as I Do checkout with the Sharpa robot assets (scripts/fetch_external.py do-as-i-do)
"""
import os
from pathlib import Path

SIM=Path(__file__).resolve().parent;ROBOT=SIM.parent;REPO=ROBOT.parent
DATA=Path(os.environ.get('BIHOI_ROBOT_DATA',REPO/'data'/'robot'))
OUTPUTS=Path(os.environ.get('BIHOI_ROBOT_OUTPUTS',REPO/'outputs'/'robot'))
MANO_SOURCE=Path(os.environ.get('HOI_MANO_DIR',REPO/'.models'/'mano'))
DO_AS_I_DO=Path(os.environ.get('HOI_EXTERNAL_ROOT',REPO/'.external'))/'do-as-i-do'/'retargeting'/'retargeting'
PYTHON=str(Path(os.environ.get('HOI_ENV_ROBOT',REPO/'.environments'/'robot'))/'bin'/'python')

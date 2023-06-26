#OAR -O /srv/tempdd/egermani/Logs/job_%jobid%.output
#OAR -E /srv/tempdd/egermani/Logs/job_%jobid%.error

# Parameters
expe_name="louvain_matrix"
main_script=/srv/tempdd/egermani/pipeline_distance/src/preprocess.py

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE

# The job
. /etc/profile.d/modules.sh
module load miniconda

source /soft/igrida/miniconda/miniconda-latest/bin/activate /srv/tempdd/egermani/workEnv

# -u : Force les flux de sortie et d'erreur standards à ne pas utiliser de tampon. 
# Cette option n'a pas d'effet sur le flux d'entrée standard
/srv/tempdd/egermani/workEnv/bin/python -u $main_script 
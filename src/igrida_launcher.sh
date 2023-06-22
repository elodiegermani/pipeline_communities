#OAR -O /srv/tempdd/egermani/Logs/job_%jobid%.output
#OAR -E /srv/tempdd/egermani/Logs/job_%jobid%.error

output_file=$PATHLOG/$OAR_JOB_ID.txt

# Parameters
expe_name="louvain_matrix"
main_script=/srv/tempdd/egermani/pipeline_distance/src/louvain_matrix.py

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/srv/tempdd/egermani/Logs/${CURRENTDATE}_OARID_${OAR_JOB_ID}/"
echo "path log :"
echo $PATHLOG
mkdir $PATHLOG

output_file=$PATHLOG/$OAR_JOB_ID.txt

# The job
# source .bashrc
#source /srv/tempdd/egermani/miniconda3/etc/profile.d/conda.sh
#source /srv/tempdd/egermani/miniconda3/bin/activate
#conda activate workEnv

. /etc/profile.d/modules.sh
module load miniconda

source /soft/igrida/miniconda/miniconda-latest/bin/activate /srv/tempdd/egermani/workEnv

#conda activate workEnv

# -u : Force les flux de sortie et d'erreur standards à ne pas utiliser de tampon. 
# Cette option n'a pas d'effet sur le flux d'entrée standard
/srv/tempdd/egermani/workEnv/bin/python -u $main_script >> $output_file 
#$ -R y
#$ -q all.q
#$ -P EOSL
#$ -cwd
#$ -notify
#$ -V
#$ -l h_rt=10:00:00
$HOME/virtualenv/mil/bin/python $HOME/CS/adult.py

# Stdlib packages
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from copy import deepcopy


################################

GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)


################################


logger = logging.getLogger(__name__)

CWD = os.getcwd()
CONDOR_DIRPATH = os.path.join(CWD, '.condor_train', '')
if not os.path.exists(CONDOR_DIRPATH):
    os.makedirs(CONDOR_DIRPATH)
CONDOR_JOBS_DIRPATH, CONDOR_INPUT_DIRPATH = None, None
BASE_NAME = 'condor_train'
CONDOR_FILEPATHS = {
    'executable': lambda jobs_dir: os.path.join(jobs_dir, f"{BASE_NAME}.sh"),
    'submission': lambda jobs_dir: os.path.join(jobs_dir, f"{BASE_NAME}.sub"),
    'out': lambda jobs_dir: os.path.join(jobs_dir, f"{BASE_NAME}.$(ClusterId).$(ProcId).out"),
    'err': lambda jobs_dir: os.path.join(jobs_dir, f"{BASE_NAME}.$(ClusterId).$(ProcId).err"),
    'log': lambda jobs_dir: os.path.join(jobs_dir, f"{BASE_NAME}.$(ClusterId).log")
}

################################


def make_condor_sub_dirpath(condor_sub_dirpath: str):
    CONDOR_JOBS_DIRPATH = os.path.join(CONDOR_DIRPATH, condor_sub_dirpath, 'jobs', '')
    os.makedirs(CONDOR_JOBS_DIRPATH)

    CONDOR_INPUT_DIRPATH = os.path.join(CONDOR_DIRPATH, condor_sub_dirpath, 'inputs', '')
    os.makedirs(CONDOR_INPUT_DIRPATH)

    for ext in CONDOR_FILEPATHS.keys():
        CONDOR_FILEPATHS[ext] = CONDOR_FILEPATHS[ext](CONDOR_JOBS_DIRPATH)

def postprocessing(output_dirpath: str, eos_dirpath: str):
    with open('output_files.txt', 'r') as f:
        for line in f:
            stdline = line.strip()
            subprocess.run(['xrdcp', os.path.join(eos_dirpath, stdline), os.path.join(output_dirpath, stdline)], check=True)
    subprocess.run(['rm', 'output_files.txt'], check=True)

def submit(
    dataset_dirpath: str, output_dirpath: str, 
    eos_dirpath: str,  n_folds: int, 
    memory: str="10GB", queue: str='workday'
):
    """
    A method to submit all the jobs in the jobs_dir to the cluster
    """
    # Commits and pushes newest version of model to github so condor can pull directly from github
    subprocess.run(['git', 'commit', '-a', '-m', f'Commit before training at {CURRENT_TIME}'], check=True)
    subprocess.run(['git', 'push'], check=True)

    # Exports conda env information
    conda_filename = 'environment.yml'
    conda_filepath = os.path.join(GIT_REPO, conda_filename)
    subprocess.run(['conda', 'env', 'export', '--from-history', '>', conda_filepath], check=True)

    # Zips the datset for easy transfer to condor nodes
    lightweight_tarfilename = 'lightweight_dataset.tar.gz'
    lighweight_tarfilepath = os.path.join(GIT_REPO, lightweight_tarfilename)
    lighweight_EOStarfilepath = os.path.join(eos_dirpath, lightweight_tarfilename)
    subprocess.run(['tar', '-zcf', lighweight_tarfilepath, dataset_dirpath], check=True)
    subprocess.run(['xrdcp', '-f', lighweight_tarfilepath, lighweight_EOStarfilepath], check=True)

    # Makes directories on submitter machine for reviewing outputs/errors
    make_condor_sub_dirpath('/'.join(output_dirpath.split('/')[-5:]))

    # srv dirpaths
    dataset_srvdirpath = os.path.join('/srv', dataset_dirpath)
    output_srvdirpath = os.path.join('/srv', output_dirpath)
    conda_srvfilename = os.path.join('/srv', conda_filename)

    # Makes the executable file
    with open(CONDOR_FILEPATHS['executable'], "w") as executable_file:
        # Setting up conda environment and HiggsDNA directories
        executable_file.write("echo \"Start of job $1\"\n")
        executable_file.write("echo \"-------------------------------------\"\n")

        executable_file.write("echo \"Pulling miniforge conda to node\"\n")
        executable_file.write("wget -O Miniforge3.sh \"https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh\"\n")

        executable_file.write("echo \"Building miniforge conda\"\n")
        executable_file.write("bash Miniforge3.sh -b -p \"${HOME}/conda\"\n")
        executable_file.write("source \"${HOME}/conda/etc/profile.d/conda.sh\"\n")
        executable_file.write("conda activate\n")
        executable_file.write("echo \"-------------------------------------\"\n")

        executable_file.write("echo \"Pulling training code from github\"\n")
        executable_file.write(f"git clone https://github.com/tcoulvert/HHtobbyy\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write("ls -la\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write("cd\n")

        executable_file.write("echo \"Pulling training dataset from EOS\"\n")
        executable_file.write(f"xrdcp {lighweight_EOStarfilepath} .\n")
        executable_file.write(f"tar -xzf {lighweight_EOStarfilepath}\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write("ls -la\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write("cd\n")

        executable_file.write(f"echo \"Building conda env\"\n")
        executable_file.write(f"conda env create -f {conda_srvfilename}\n")
        executable_file.write("conda activate higgs-dna\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        
        executable_file.write("echo \"Running training processing\"\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write("cd /srv/HHtobbyy/training\n")
        for job_idx in range(n_folds):
            executable_file.write(f"if [ $1 -eq {i} ]; then\n")
            executable_file.write(f"    python run_training.py {dataset_srvdirpath} --fold {i}\n")
            executable_file.write("fi\n")
        executable_file.write("cd /srv\n")

        executable_file.write("echo \"Transfering output to EOS\"\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write("ls /srv\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write(f"ls {output_srvdirpath}\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write(f"ls {os.path.join(output_srvdirpath, '*BDT*')} | cut -c {len(output_srvdirpath)+1}- > output_files.txt\n")
        executable_file.write("cat output_files.txt\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write(f"while IFS= read -r line; do\n")
        executable_file.write(f"    if [ -f {output_srvdirpath}\"${{line}}\" ]; then\n")
        executable_file.write(f"        xrdcp {output_srvdirpath}\"${{line}}\" {eos_dirpath}\"${{line}}\"\n")
        executable_file.write(f"    fi \n")
        executable_file.write("done < output_files.txt\n")
    subprocess.run(['chmod', '775', CONDOR_FILEPATHS['executable']], check=True)

    # Makes the submission file
    with open(CONDOR_FILEPATHS['submission'], "w") as submit_file:
        submit_file.write(f"executable = {CONDOR_FILEPATHS['executable']}\n")
        submit_file.write("arguments = $(ProcId)\n")
        submit_file.write(f"output = {CONDOR_FILEPATHS['out']}\n")
        submit_file.write(f"error = {CONDOR_FILEPATHS['err']}\n")
        submit_file.write(f"log = {CONDOR_FILEPATHS['log']}\n")
        submit_file.write(f"request_memory = {memory}\n")
        submit_file.write("getenv = True\n")
        submit_file.write(f'+JobFlavour = "{queue}"\n')
        submit_file.write(f"should_transfer_files = YES\n")
        submit_file.write(f"Transfer_Input_Files = {conda_filepath}\n")
        submit_file.write(f"Transfer_Output_Files = output_files.txt\n")
        submit_file.write(f'when_to_transfer_output = ON_EXIT\n')

        submit_file.write('on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)\n')
        submit_file.write('max_retries = 0\n')
        # submit_file.write('requirements = Machine =!= LastRemoteHost\n')
        submit_file.write(f"queue {n_folds}\n")
    
    # Submits the condor jobs
    if CWD.startswith("/eos"):
        # see https://batchdocs.web.cern.ch/troubleshooting/eos.html#no-eos-submission-allowed
        output = subprocess.run(["condor_submit", "-spool", CONDOR_FILEPATHS['submission']], capture_output=True, text=True, check=True)
    else:
        output = subprocess.run("condor_submit {}".format(CONDOR_FILEPATHS['submission']), capture_output=True, text=True, shell=True, check=True)

    cluster_id = int(re.search(r'\d+', output.stdout).group(0)[::-1])
    print(cluster_id)
    while True:
        output = subprocess.run(['condor_q', '-constraint', '\"ClusterId == ${CLUSTER_ID} && (JobStatus == 1 || JobStatus == 2)\"', '-af', 'ClusterId' '|' 'wc' '-l'], capture_output=True, text=True, check=True)
        if output.stdout == 0:
            print(f"Finished running condor jobs, running postprocessing.")
            postprocessing(output_dirpath, eos_dirpath)
            break
        else:
            time.sleep(60)

# Stdlib packages
import glob
import json
import logging
import os
import re
import subprocess
import time

# Condor job management
import htcondor2

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
CURRENT_TIME = None

################################


def make_condor_sub_dirpath(condor_sub_dirpath: str):
    CONDOR_JOBS_DIRPATH = os.path.join(CONDOR_DIRPATH, condor_sub_dirpath, 'jobs', '')
    os.makedirs(CONDOR_JOBS_DIRPATH)

    CONDOR_INPUT_DIRPATH = os.path.join(CONDOR_DIRPATH, condor_sub_dirpath, 'inputs', '')
    os.makedirs(CONDOR_INPUT_DIRPATH)

    for ext in CONDOR_FILEPATHS.keys():
        CONDOR_FILEPATHS[ext] = CONDOR_FILEPATHS[ext](CONDOR_JOBS_DIRPATH)

def postprocessing(output_dirpath: str):
    cluster_filepaths = glob.glob(CWD, f"{CURRENT_TIME}*BDT*")
    for filepath in cluster_filepaths:
        subprocess.run(['mv', filepath, os.path.join(output_dirpath, filepath.split('/')[-1])])

def submit(
    dataset_dirpath: str, output_dirpath: str, 
    eos_dirpath: str,  n_folds: int, 
    eval_result_filename: str, model_filename: str,
    memory: str="10GB", queue: str='workday'
):
    """
    A method to submit all the jobs in the jobs_dir to the cluster
    """
    output_dirpath = os.path.join(output_dirpath, '')
    CURRENT_TIME = output_dirpath.split('/')[-2]
    # Commits and pushes newest version of model to github so condor can pull directly from github
    try:
        subprocess.run(['git', 'commit', '-a', '-m', f'Commit before training at {CURRENT_TIME}'], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'push'], check=True)
    except subprocess.CalledProcessError as e:
        if 'Your branch is up to date with'.lower() not in e.stdout.lower(): 
            logger.error(f"Committing and pushing to git failed")
            raise e
        else: 
            logger.log(1, f"Git commit failed because branch is already up to date on remote. Continuing with batch submission")

    # Zips the datset for easy transfer to condor nodes
    dataset_dirname = dataset_dirpath.split('/')[-2]
    dataset_tarfilepath = os.path.join(GIT_REPO, f"{dataset_dirname}.tar.gz")
    EOS_redirector = eos_dirpath.split('/store')[0]
    dataset_EOStarfilepath = os.path.join(eos_dirpath, f"{dataset_dirname}.tar.gz")
    try:
        subprocess.run(['xrdfs', EOS_redirector, 'ls', dataset_EOStarfilepath.replace(EOS_redirector, '')], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if 'No such file or directory'.lower() not in e.stderr.lower(): 
            logger.error(f"Looking for tar file on EOS failed")
            raise e
        else:
            logger.log(1, f"{dataset_dirname} tar file not on EOS yet, making tar file and trasferring now")
            subprocess.run(['tar', '-zcf', dataset_tarfilepath, dataset_dirpath], check=True, capture_output=True, text=True)
            subprocess.run(['xrdcp', dataset_tarfilepath, eos_dirpath], check=True, capture_output=True, text=True)
            subprocess.run(['rm', dataset_tarfilepath], check=True, capture_output=True, text=True)

    # Makes directories on submitter machine for reviewing outputs/errors
    make_condor_sub_dirpath('/'.join(output_dirpath.split('/')[-5:]))

    # srv dirpaths
    dataset_srvdirpath = os.path.join('/srv', dataset_dirpath)
    output_srvdirpath = os.path.join('/srv', output_dirpath)

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
        executable_file.write("conda activate && rm /srv/Miniforge3.sh\n")
        executable_file.write("echo \"-------------------------------------\"\n")

        executable_file.write("echo \"Pulling training code from github\"\n")
        executable_file.write(f"git clone https://github.com/tcoulvert/HHtobbyy\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write("ls -la\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write("cd\n")

        executable_file.write("echo \"Pulling training dataset from EOS\"\n")
        executable_file.write(f"xrdcp {dataset_EOStarfilepath} .\n")
        executable_file.write(f"tar -xzf {dataset_EOStarfilepath} && rm {dataset_EOStarfilepath}\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write("ls -la\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write("cd\n")

        executable_file.write("echo \"Building conda env\"\n")
        executable_file.write(f"conda env create -f HHtobbyy/environment.yml -n train_env\n")
        executable_file.write("conda activate train_env\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        
        executable_file.write("echo \"Running training processing\"\n")
        executable_file.write("echo \"-------------------------------------\"\n")
        executable_file.write("cd /srv/HHtobbyy/training\n")
        for job_idx in range(n_folds):
            executable_file.write(f"if [ $1 -eq {job_idx} ]; then\n")
            executable_file.write(f"    python run_training.py {dataset_srvdirpath} --fold {job_idx}\n")
            executable_file.write(f"    mv {os.path.join(output_srvdirpath, f'{model_filename}{job_idx}.model')} /srv\n")
            executable_file.write(f"    mv {os.path.join(output_srvdirpath, f'{eval_result_filename}{job_idx}.json')} /srv\n")
            executable_file.write("fi\n")
        executable_file.write("cd /srv\n")
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
        submit_file.write(f"+JobFlavour = \"{queue}\"\n")
        submit_file.write(f"should_transfer_files = YES\n")
        submit_file.write(f"Transfer_Input_Files = \n")
        submit_file.write(f"Transfer_Output_Files = \n")
        submit_file.write(f'when_to_transfer_output = ON_EXIT\n')

        submit_file.write("on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)\n")
        submit_file.write("max_retries = 0\n")
        # submit_file.write('requirements = Machine =!= LastRemoteHost\n')
        submit_file.write(f"queue {n_folds}\n")
    
    # Submit the condor jobs
    schedd = htcondor2.Schedd()
    submit_dict = {}
    with open(CONDOR_FILEPATHS['submission'], "r") as submit_file:
        for line in submit_file:
            try:
                key, value = line.split(' = ')[0].strip(), line.split(' = ')[1].strip()
                submit_dict[key] = value
            except IndexError as e:
                if "queue" not in line: 
                    logger.error(f"Making submission dictionary failed with line {line}")
                    raise e
    # see https://batchdocs.web.cern.ch/troubleshooting/eos.html#no-eos-submission-allowed
    submit_result = schedd.submit(htcondor2.Submit(submit_dict), spool=CWD.startswith("/eos"), queue=str(n_folds))

    while True:
        jobs = schedd.query(constraint=f"ClusterId == {submit_result.cluster()}", projection=["ClusterId", "ProcId", "JobStatus", "RequestMemory"])
        if len(jobs) == 0:
            logger.log(1, f"Finished running condor jobs, running postprocessing.")
            postprocessing(output_dirpath, eos_dirpath)
            break
        else:
            if any(job['JobStatus'] == 5 for job in jobs):
                old_memory = jobs[0]["RequestMemory"]
                new_memory = int(
                    int(re.search(r'\d+', old_memory).group()) * 1.5
                )
                memory_units = old_memory[re.search(r'\d+', old_memory).end():]
                schedd.edit(f"ClusterId == {submit_result.cluster()}", "RequestMemory", f"\"{new_memory}{memory_units}\"")
            time.sleep(60)

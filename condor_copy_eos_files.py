import os
import subprocess

import logging
logger = logging.getLogger(__name__)

import argparse
parser = argparse.ArgumentParser(description="Transfers files from one EOS space to another.")

# Transfer procedure args
parser.add_argument("origin_filepath", help="Full filepath (including redirector) for location of directory to transfer from.")
parser.add_argument("destination_filepath", help="Full filepath (including redirector) for location of directory to transfer to.")
parser.add_argument("--grep_str", default="...", help="String with which to select (using grep) which files to transfer. Only files whose filepath contains this string at the origin EOS space will be transfered.")
parser.add_argument("--filetype_str", default="...", help="Filetype of files to transfer, format as \".<filetype>\". Only files whose filetype matches this string will be transfered.")
parser.add_argument("--force", action="store_true", help="Boolean which enables forcing the xrdcp (overwriting files if already located at the destination).")
parser.add_argument("--output_dir", default=os.path.join(os.getcwd(), ".condor_copy_eos", ""), help="Directory for condor files to be dumped into.")
# Condor args
parser.add_argument("--queue", default="longlunch", help="Queue with which to submit the condor job.")
parser.add_argument("--memory", default="10GB", help="RAM with which to submit the condor job.")

def main():
    args = parser.parse_args()

    origin_redirector = args.origin_filepath[:args.origin_filepath.find("/store")]
    origin_filepath = os.path.join(args.origin_filepath[args.origin_filepath.find("/store"):], "")
    destination_redirector = args.destination_filepath[:args.destination_filepath.find("/store")]
    destination_filepath = os.path.join(args.destination_filepath[args.destination_filepath.find("/store"):], "")
    
    jobs_dir = os.path.join(args.output_dir, subprocess.getoutput("date +%Y%m%d_%H%M%S"), "")
    if not os.path.exists(jobs_dir): os.makedirs(jobs_dir)

    # Making a temporary file containing a list of all the files that need to be trsnferred from one EOS space to another
    if args.grep_str != "...":
        os.system(f"xrdfs {origin_redirector} ls -R {origin_filepath} | grep {args.grep_str} > temp.txt")
    else:
        os.system(f"xrdfs {origin_redirector} ls -R {origin_filepath} > temp.txt")

    # Skimming output and keeping only real files (that have the right filetype, if given)
    files_to_copy = []
    with open("temp.txt", "r") as f:
        for line in f:
            formatted_line = line.rstrip()
            end_of_filepath = formatted_line.split("/")[-1]
            if (
                args.filetype_str != "..." 
                and end_of_filepath.endswith(args.filetype_str)
            ) or (
                args.filetype_str == "..." 
                and end_of_filepath.find(".") != -1
            ): 
                files_to_copy.append(formatted_line[formatted_line.find(origin_filepath)+len(origin_filepath):])

    # Remove already trasnferred files if not forcing
    if not args.force:
        skimmed_files_to_copy = []
        for file_to_copy in files_to_copy:
            stat, out = subprocess.getstatusoutput(f"xrdfs {destination_redirector} ls {destination_filepath}{file_to_copy}")
            if stat != 0: skimmed_files_to_copy.append(file_to_copy)
        files_to_copy = skimmed_files_to_copy
    if len(files_to_copy) < 1:
        return 1
    print(files_to_copy)

    # Deleting the temp file
    os.system(f"rm temp.txt")

    # Get proxy information (required in executable script for this method of running)
    try:
        stat, out = subprocess.getstatusoutput("voms-proxy-info -e --valid 5:00")
    except:
        logger.exception(
            "voms proxy not found or validity less that 5 hours:\n%s",
            out
        )
        raise
    try:
        stat, out = subprocess.getstatusoutput("voms-proxy-info -p")
        out = out.strip().split("\n")[-1]
    except:
        logger.exception(
            "Unable to voms proxy:\n%s",
            out
        )
        raise
    proxy = out

    # Setup the filepaths for the input and output files
    base_name = "copy_eos"
    job_file_executable = os.path.join(jobs_dir, f"{base_name}.sh")
    job_file_submit = os.path.join(jobs_dir, f"{base_name}.sub")
    job_file_out = os.path.join(jobs_dir, f"{base_name}.$(ClusterId).$(ProcId).out")
    job_file_err = os.path.join(jobs_dir, f"{base_name}.$(ClusterId).$(ProcId).err")
    job_file_log = os.path.join(jobs_dir, f"{base_name}.$(ClusterId).log")
    n_jobs = len(files_to_copy)

    # Write the executable file
    with open(job_file_executable, "w") as executable_file:
        # Shabang and x509 proxy
        executable_file.write("#!/bin/bash\n")
        executable_file.write(f"export X509_USER_PROXY={'/srv'+proxy[proxy.rfind('/'):]}\n")

        # Transfer files
        executable_file.write("echo \"Start of job $1\"\n")
        executable_file.write("echo \"-------------------------------------\"\n")

        for i, file_to_copy in enumerate(files_to_copy):
            executable_file.write(f"if [ $1 -eq {i} ]; then\n")
            # executable_file.write(f"    echo \"Transfering {file_to_copy}\"\n")
            executable_file.write(f"    xrdcp{' -f' if args.force else ''} {origin_redirector}{origin_filepath}{file_to_copy} {destination_redirector}{destination_filepath}{file_to_copy}\n")
            executable_file.write("fi\n")

        executable_file.write("echo \"Finished job $1\"\n")
        executable_file.write("echo \"-------------------------------------\"\n")
    os.system(f"chmod 775 {job_file_executable}")

    # Write the submit file
    with open(job_file_submit, "w") as submit_file:
        submit_file.write(f"executable = {job_file_executable}\n")
        submit_file.write("arguments = $(ProcId)\n")
        submit_file.write(f"output = {job_file_out}\n")
        submit_file.write(f"error = {job_file_err}\n")
        submit_file.write(f"log = {job_file_log}\n")
        submit_file.write(f"request_memory = {args.memory}\n")
        submit_file.write("getenv = True\n")
        submit_file.write(f'+JobFlavour = "{args.queue}"\n')
        submit_file.write(f"should_transfer_files = YES\n")
        submit_file.write(f"Transfer_Input_Files = {proxy}\n")
        submit_file.write(f"Transfer_Output_Files = \"\"\n")
        submit_file.write(f'when_to_transfer_output = ON_EXIT\n')

        submit_file.write('on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)\n')
        submit_file.write('max_retries = 0\n')
        submit_file.write(f"queue {n_jobs}\n")

    # Submit the condor jobs
    if os.getcwd().startswith("/eos"):
        # see https://batchdocs.web.cern.ch/troubleshooting/eos.html#no-eos-submission-allowed
        subprocess.run(["condor_submit", "-spool", job_file_submit])
    else:
        subprocess.run("condor_submit {}".format(job_file_submit), shell=True)

if __name__ == "__main__":
    main()

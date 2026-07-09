import os
import re
import sys
import time

if len(sys.argv) == 3:
    total_run_time = 12 * 60 * 60  # 12 hours
elif len(sys.argv) == 4:
    total_run_time = float(sys.argv[-1]) * 60 * 60  # argument given in hours
else:
    raise Exception(f"Only 2 or 3 arguments allowed")
ClusterId = sys.argv[1]
ScheddName = sys.argv[2]

start_time = time.time()
run_time = lambda: time.time() - start_time
tmp_filename = "tmp_info.txt"
while run_time() < total_run_time:
    os.system(f"condor_q -n {ScheddName} -constraint \"ClusterId == {ClusterId}\" -af ProcId JobStatus RequestMemory > {tmp_filename}")
    if os.path.getsize(tmp_filename) == 0:
        print(f"Finished running condor jobs.")
        break
    with open(tmp_filename, 'r') as f:
        for line in f:
            job_info = line.strip().split()
            ProcId, JobStatus, RequestMemory = int(job_info[0]), int(job_info[1]), job_info[2]
            if JobStatus == 5:
                new_RequestMemory = str(int( int(re.search(r'\d+', RequestMemory).group(0)) * 2 ))+str(RequestMemory[re.search(r'\d+', RequestMemory).end():])
                if int( int(re.search(r'\d+', RequestMemory).group(0)) * 2 ) >= 16_384: continue
                os.system(f"condor_qedit {ClusterId}.{ProcId} RequestMemory=\"{new_RequestMemory}\" -n {ScheddName}")
                os.system(f"condor_release {ClusterId}.{ProcId} -n {ScheddName}")
                print(f"JobId {ClusterId}.{ProcId} held, requesting 2x memory and resubmitting.")
    time.sleep(300)
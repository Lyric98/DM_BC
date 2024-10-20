import submitit

def run_job():
    # Your job script here
    import os
    os.system("python sr.py -p train -c config/sr_wave_64_512FFHQ.json")

def main():
    # Create a submitit executor
    executor = submitit.AutoExecutor(folder="experiments/logs")  # folder to store logs

    # Set SLURM job parameters
    executor.update_parameters(
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        # gpus_per_node=2,  # equivalent to --gres=gpu:2
        mem_gb=30,
        slurm_gres='gpu:rtx8000:1',
        time=60*12,  # minutes, equivalent to --time=48:00:00
        job_name="DMBC",
        account="pr_174_general"
    )

    # Submit the job
    job = executor.submit(run_job)
    print(f"Job submitted with ID: {job.job_id}")

if __name__ == "__main__":
    main()

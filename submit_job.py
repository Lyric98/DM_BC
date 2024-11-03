import submitit

def run_job():
    # Your job script here
    import os
    os.system("python sr.py -p train -c config/sr_wave_64_512FFHQ.json -enable_wandb -log_wandb_ckpt -log_eval")
    # os.system("python CBIS_dataset.py")
    # os.system("python data/prepare_data.py --path dataset/CBIS_full/full_image_RGB --out CBIS_test --size 64,512")

def main():
    # Create a submitit executor
    executor = submitit.AutoExecutor(folder="experiments/logs")  # folder to store logs

    # Set SLURM job parameters
    executor.update_parameters(
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1, #20
        gpus_per_node=2,  # equivalent to --gres=gpu:2
        mem=50,
        slurm_gres='gpu:2',
        # slurm_gres='gpu:rtx8000:1',
        slurm_time=60*48,  # minutes, equivalent to --time=48:00:00
        slurm_job_name="DMBC",
        slurm_account="biostats",
        # account="pr_174_general"
    )

    # Submit the job
    job = executor.submit(run_job)
    print(f"Job submitted with ID: {job.job_id}")

if __name__ == "__main__":
    main()

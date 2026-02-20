import time
from src.dashboard.deploy_runner import submit_job, get_job, get_results, get_grid

# A small bbox in Germany (e.g. around Nuremberg)
# Longitude / Latitude
BBOX = [11.0, 49.4, 11.1, 49.5]
YEARS = [2020, 2021]

def main():
    print(f"Submitting deploy job for bbox={BBOX}, years={YEARS}")
    job_id = submit_job(BBOX, YEARS)
    print(f"Job ID: {job_id}")

    while True:
        job = get_job(job_id)
        if not job:
            print("Job not found!")
            break
            
        print(f"Status: {job.status} | Stage: {job.stage} | Progress: {job.progress:.1f}%")
        
        if job.status in ["complete", "error"]:
            print(f"Final status: {job.status}")
            if job.error:
                print(f"Error: {job.error}")
            break
            
        time.sleep(2)
        
    print("\nChecking results...")
    for yr in YEARS:
        res = get_results(job_id, yr)
        if res:
            print(f"Year {yr}: {len(res)} cells of predictions/labels")
        else:
            print(f"Year {yr}: NO DATA")
            
    grid = get_grid(job_id)
    if grid:
        print(f"Grid: {len(grid['features'])} cells")
    else:
        print("Grid: NO DATA")

if __name__ == "__main__":
    main()

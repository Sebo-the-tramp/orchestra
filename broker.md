idea of the broker

1. add jobs to a queue 
    a. check if model exists
    b. check if request format is correct for that particular model
    c. finally add to the queue

2. dispatch job
    a. if worker is ready dispatch job 
    b. if worker is loading continue 
    c. if worker not present -> spawn process put in loading and continue

3. purge_jobs (how?)

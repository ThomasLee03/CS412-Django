Please note that:

For PCA imputation (as well as PCA imputation preprocessing), you will get a runtime error: at=error code=H12 desc="Request timeout"  
- This is due to the fact that it takes around 2-3 minutes, which is not permitted in the time window for the basic heroku package.

For Total variation in Painting, you will get a memory error: 
  Process running mem=1553M(303.4%)
  Error R15 (Memory quota vastly exceeded)
  Stopping process with SIGKILL
  - This is due to the fact that total variation in painting takes up a ton of memory to process the image, which is not permitted with the basic heroku package.

If you attempt to run the TV in Painting or the PCA imputation methods, they should take around 2-5 minutes to complete based on the image.

To do PCA preprocessing, you must select an imputation method and then check the PCA preprocessing. Note that PCA preprocessing only works with mean, median, and mode, and will just take you to the same to choose an imputation method. 

To log into the two different accounts I showed:

- Username: thomas
- Password: Th0mas123

- Username: jinhee
- Password: Th0mas123

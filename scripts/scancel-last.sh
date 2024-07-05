scancel $(<.dawgz/$(cat .dawgz/workflows.csv | awk -F',' '{ print $2 }' | tail -n1)/jobids)

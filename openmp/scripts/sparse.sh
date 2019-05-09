#/bin/bash

# ./sparse ./matrix/2cubes_sphere.mtx 2048
# ./sparse ./matrix/cage12.mtx 1024
# ./sparse ./matrix/consph.mtx 2048
# ./sparse ./matrix/cop20k_A.mtx 2048
# ./sparse ./matrix/filter3D.mtx 2048
# ./sparse ./matrix/hood.mtx 1024
# ./sparse ./matrix/m133-b3.mtx 1024
# ./sparse ./matrix/mac_econ_fwd500.mtx 1024
# ./sparse ./matrix/majorbasis.mtx 1024
# ./sparse ./matrix/mario002.mtx 512
# ./sparse ./matrix/mc2depi.mtx 512
# ./sparse ./matrix/offshore.mtx 1024
# ./sparse ./matrix/patents_main.mtx 1024
# ./sparse ./matrix/pdb1HYS.mtx 4096
# ./sparse ./matrix/poisson3Da.mtx 16384
# ./sparse ./matrix/pwtk.mtx 1024
# ./sparse ./matrix/rma10.mtx 4096
# ./sparse ./matrix/scircuit.mtx 1024
# ./sparse ./matrix/shipsec1.mtx 1024
# ./sparse ./matrix/webbase-1M.mtx 256 

test_cases=("./matrix/2cubes_sphere.mtx 2048" "./matrix/cage12.mtx 1024" "./matrix/consph.mtx 2048" \
              "./matrix/cop20k_A.mtx 2048" "./matrix/filter3D.mtx 2048" "./matrix/hood.mtx 1024" \
              "./matrix/m133-b3.mtx 1024" "./matrix/mac_econ_fwd500.mtx 1024" "./matrix/majorbasis.mtx 1024" \
              "./matrix/mario002.mtx 512" "./matrix/mc2depi.mtx 512" "./matrix/offshore.mtx 1024" \
              "./matrix/patents_main.mtx 1024" "./matrix/pdb1HYS.mtx 4096" "./matrix/poisson3Da.mtx 16384" \
              "./matrix/pwtk.mtx 1024" "./matrix/rma10.mtx 4096" "./matrix/scircuit.mtx 1024" \
              "./matrix/shipsec1.mtx 1024" "./matrix/webbase-1M.mtx 256")
threads=(1 2 3 4 5 6)

echo "${test_cases[0]}"
echo "${test_cases[1]}"

test_cnt=0
thread_cnt=0

while [ $test_cnt -lt 20 ]; do
  echo "--------------------------------------------------------------------"
  echo "test for : ${test_cases[test_cnt]}" 

  ((thread_cnt = 0))
  while [ $thread_cnt -lt 6 ]; do
    echo "--------------------------------------------------------------------"
    echo "# of thread : ${threads[thread_cnt]}"
    ./sparse ${test_cases[test_cnt]} ${threads[thread_cnt]}
    echo "--------------------------------------------------------------------"
    ((thread_cnt++))
  done
  ((test_cnt++))
done

# for test_case in ${test_cases[@]}; do
#   echo "--------------------------------------------------------------------"
#   echo "test for : "${test_case}"" 
#   for thread in ${threads[@]}; do
#     echo "--------------------------------------------------------------------"
#     echo "# of thread : ${thread}"
#     ./sparse "${test_case}" "${thread}"
#     echo "--------------------------------------------------------------------"
#   done
#   echo "--------------------------------------------------------------------"
# done
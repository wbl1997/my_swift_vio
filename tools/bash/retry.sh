#!/usr/bin/env bash
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <"bash_command"> [try_times(=10)]" 
    exit 1
fi
TRY_TIMES=10
if [[ $# -eq 2 ]]; then
   TRY_TIMES=$2
fi

CMD=$1
n=0
until [ $n -ge $TRY_TIMES ]
do
   $CMD && break  # substitute your command here
   n=$[$n+1]
   sleep 2
done



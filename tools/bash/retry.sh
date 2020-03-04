
n=0
until [ $n -ge 10 ]
do
   $CMD && break  # substitute your command here
   n=$[$n+1]
   sleep 10
done

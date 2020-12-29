# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

read -r N
for (( i=0; i<$N; i++ )); do
    read -r Pi
    horses[$i]=$Pi
done
sorted=( $( printf "%s\n" "${horses[@]}" | sort -n ) )
min=${sorted[0]}
for (( i=0; i<$((N-1)); i++ )); do
    d=$((sorted[(($i+1))] - sorted[$i]))
    (( d < min )) && min=$d
done
echo $min

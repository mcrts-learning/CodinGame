# game loop

while true; do
    H=0
    idx=0
    for (( i=0; i<8; i++ )); do
        # mountainH: represents the height of one mountain.
        read mountainH
        if [ "$H" -lt "$mountainH" ]
        then
            H=$mountainH
            idx=$i
        fi
    done

    # Write an action using echo
    # To debug: echo "Debug messages..." >&2

    echo "$idx" # The index of the mountain to fire on.
done

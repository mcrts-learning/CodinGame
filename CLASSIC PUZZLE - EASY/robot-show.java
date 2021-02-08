import java.util.*;
import java.io.*;
import java.math.*;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
class Solution {

    public static void main(String args[]) {
        Scanner in = new Scanner(System.in);
        int L = in.nextInt();
        int N = in.nextInt();
        List<Integer> bots =  new ArrayList<Integer>() ;
        for (int i = 0; i < N; i++) {
            bots.add(in.nextInt());
        }
        int left = Collections.min(bots);
        int right = Collections.max(bots);
        int t = Math.max(L - left, right);

        System.out.println(t);
    }
}

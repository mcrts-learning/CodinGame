import java.util.*;
import java.io.*;
import java.math.*;

class Solution {
    public static void main(String args[]) {
        Scanner in = new Scanner(System.in);
        long n = in.nextInt();
        in.close();
        for (long a=n+1; a<= 2*n; a++) {
            long b = (n*a) / (a-n);
            long q = (n*a) % (a-n);
            if (q == 0) {
                String s = String.format("1/%d = 1/%d + 1/%d", n, b, a);
                System.out.println(s);
            }
        }
    }
}

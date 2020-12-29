import java.util.*;
import java.io.*;
import java.math.*;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
class Player {

    public static void main(String args[]) {
        Scanner in = new Scanner(System.in);
        int width = in.nextInt();
        int height = in.nextInt();
        char[][] map = new char[height][width];

        for (int i = 0; i < height; i++) {
            String line = in.next();
            map[i] = line.toCharArray();
        }
        char[][] modmap = new char[height][width];
        for (int j=0; j < height; j++) {
            for (int i=0; i< width; i++) {
                if (map[j][i] == '0') {
                    int count = 0;
                    if (j-1 >= 0 && map[j-1][i] == '0' ) {
                        count++;
                    }
                    if (j+1 < height && map[j+1][i] == '0') {
                        count++;
                    }
                    if (i-1 >= 0 && map[j][i-1] == '0') {
                        count++;
                    }
                    if (i+1 < width && map[j][i+1] == '0') {
                        count++;
                    }
                    modmap[j][i] = (char)(count + 48);
                } else {
                    modmap[j][i] = '#';
                }
            }
            System.out.println(modmap[j]);




            // Write an action using System.out.println()
            // To debug: System.err.println("Debug messages...");
        }
    }
}

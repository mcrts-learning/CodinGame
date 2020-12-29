using System;
using System.Linq;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
class Solution
{
    static void Main(string[] args)
    {
        var input_string = Console.ReadLine();
        var split_string = input_string.Split('0');
        var counter = new List<int>();
        foreach (string s in split_string) {
            counter.Add(s.Length);
        }
        var cumul_counter = new List<int>();
        for (int i = 0; i < counter.Count - 1; i++) {
            cumul_counter.Add(counter[i] + counter[i+1]);
        }
        var max = cumul_counter.Max();
        Console.WriteLine(max + 1);
    }
}

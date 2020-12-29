/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/

const C: number = 5;
const N: number = 10;

const r: number = parseInt(readline());
const v: number = parseInt(readline());
var robbers = Array<number>(r).fill(0);

for (let i = 0; i < v; i++) {
    var inputs: string[] = readline().split(' ');
    var c: number = parseInt(inputs[0]);
    var n: number = parseInt(inputs[1]);
    var time: number = C**(c-n) * N**n;
    var robber = robbers.indexOf(Math.min(...robbers));
    robbers[robber] += time;
}

// Write an answer using console.log()
// To debug: console.error('Debug messages...');

console.log(Math.max(...robbers));

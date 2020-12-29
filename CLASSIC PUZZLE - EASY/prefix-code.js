const n = parseInt(readline());
var mapping = new Map();

for (let i = 0; i < n; i++) {
    var inputs = readline().split(' ');
    const b = inputs[0];
    const c = String.fromCharCode(parseInt(inputs[1]));
    mapping.set(b, c)
}
const s = readline();
var buffer = new String();
var code = new String();
s.split('').forEach(c => {
    buffer += c;
    if (mapping.get(buffer)) {
        code += mapping.get(buffer)
        buffer = new String();
    }
}
)
if (buffer.length > 0) {
    console.log('DECODE FAIL AT INDEX ' + (s.length - buffer.length));
} else {
    console.log(code);
}

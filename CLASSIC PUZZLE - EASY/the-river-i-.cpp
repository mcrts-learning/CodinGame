#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

long nextvalue(long number){
    long result = number;
    std::string s = std::to_string(number);

	for (std::string::size_type i = 0; i < s.size(); i++) {
		result += s[i] - '0';
	}
    return result;
}


int main()
{
    long long r1;
    cin >> r1; cin.ignore();
    long long r2;
    cin >> r2; cin.ignore();

    while(r1 != r2){
        if (r1 < r2) {
            r1 = nextvalue(r1);
        }
        else {
            r2 = nextvalue(r2);
        }
    }

    cout << r1 << endl;
}

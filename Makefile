all: trmf

trmf: trmf.cpp
	g++ -std=c++11 -O3 trmf.cpp -o trmf

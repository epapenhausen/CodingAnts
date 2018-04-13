#!/usr/bin/python

import sys, getopt, math

def parseFile(filename):
    readArray = False
    array = []
    with open(filename) as f:   
        for line in f:
            if "end" in line:
                readArray = False

            if readArray:
                array.append([float(x) for x in line.split()])

            if "begin dump:" in line:
                readArray = True
    return array

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
             
    arrayOrig = parseFile(inputfile);
    arrayPPCG = parseFile(outputfile);
    

    for a1, a2 in zip(arrayOrig, arrayPPCG):
        num = 0
        den = 0
        # get the l2 norm   
        for v1, v2 in zip(a1, a2):
            num += ((v1 - v2) * (v1 - v2))
            den += (v2 * v2)
        num = math.sqrt(num);
        den = math.sqrt(den);
        if den == 0:
            den += 0.1 # epsilon

        if (num / den) >= 0.05:
            print a1
            print a2
            print (num / den)
            print 'Fail'
            sys.exit(1)
                                        
    return True

if __name__ == "__main__":
    main(sys.argv[1:])

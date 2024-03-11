import sys


def main():
    #Keep track of each independent run to load models for later
    if len(sys.argv) >2:
        fname = sys.argv[1]
        outname = sys.argv[2]
    else: #If I forgot to send in a version, try and generate a unique Vnum.
        print('Supply a file name to read from, and a output file to write to')
        return
    
    with open(fname, 'r') as f:
        averages = []
        for row in f:
            avgRun = row.split()
            averages.append(sum(avgRun)/len(avgRun))
    with open(outname, 'w') as f:
        for avgVal in averages:
            outname.write(str(avgVal)+'\n')
main()

#coding: utf-8
import matplotlib.pyplot as plt

epochs = []
errors = []

if __name__ == '__main__':
    argvs = sys.argv
    argc = len(argvs)
    if(argc != 4):
        print 'Usage: python %s <log file path>' % argvs[0]
        quit()
    logfile = argvs[1]
    
    fp = open(logfile)
    
    for line in fp:
        if line == "":
            continue
        line = line.strip().split()
        epochs.append(int(line[0]))
        errors.append(float(line[1]))
    fp.close()

    plt.plot(epochs, errors, "r-")
    plt.xlabel("epoch")
    plt.ylabel("error (%)")
    plt.grid()
    plt.tight_layout()
    plt.show()


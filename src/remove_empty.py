import  sys




def main():
    f1 = sys.argv[1]
    f2 = sys.argv[2]
     
    file1 = open(f1,'r')
    file2 = open(f2,'r')

    file1_ = open(f1+'.out','w')
    file2_ = open(f2+'.out','w')

    for l1,l2 in zip(file1,file2):
        l1 = l1.strip()
        l2 = l2.strip()
        if len(l1) >0 and len(l2) >0:
            file1_.write(l1+'\n')
            file2_.write(l2+'\n')
    file1_.close()
    file2_.close()
main()
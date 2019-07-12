import numpy as np

with open('book_tags.csv','r') as f:
    with open('tag_array.csv','w') as f2:
        firstline = True
        ID = 0
        tags =[]
        counts = []
        f2.write('goodreads_book_id,tag_id,count\n')
        for line in f:
            if firstline:
                firstline = False
                continue
            #map(int,line.strip().split(','))
            dat = line.strip().split(',')
            dat = [int(dat[0]),int(dat[1]),int(dat[2])]
            #print(dat)
            if dat[0] == ID:
                tags.append(dat[1])
                counts.append(dat[2])
            else:
                f2.write(str(ID)+',\"'+str(tags)+'\",\"'+str(counts)+'\"\n')
                tags = [dat[1]]
                counts = [dat[2]]
                ID = dat[0]



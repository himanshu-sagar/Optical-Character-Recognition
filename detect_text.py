import copy
import cv2

def hk(img, bg = 255, nh = 8):
    
    def neighborhood4(i,j):
        lst = []
        if(i==0):
            if(j==0):
                return lst
            else:
                lst.append((i, j-1))
                return lst
        elif(j==0):
            lst.append((i-1, j))
            return lst
        else:
            lst.append((i-1,j))
            lst.append((i, j-1))
            return lst
    
    def neighborhood8(i,j, ncols):
        lst = []
        if (i == 0):
            if (j == 0):
                return lst
            else:
                lst.append((i, j - 1))
                return lst
        elif (j == 0):
            lst.append((i - 1, j))
            lst.append((i-1, j+1))
            return lst
        elif(j == ncols-1):
            lst.append((i - 1, j))
            lst.append((i, j - 1))
            lst.append((i-1, j-1))
            return lst
        else:
            lst.append((i, j-1))
            lst.append((i-1, j-1))
            lst.append((i-1, j))
            lst.append((i-1, j+1))
            return lst

    def combineObjs(objList):
        rmin = min([ele[0] for ele in objList])
        rmax = max([ele[1] for ele in objList])
        cmin = min([ele[2] for ele in objList])
        cmax = max([ele[3] for ele in objList])
        return (rmin,rmax,cmin,cmax)

    def printMat(imag):
        for row in imag:
            for col in row:
                print(col, end= " ", file=text_file)
            print(file=text_file)
        print(file=text_file)
        print(file=text_file)

    label = 1
    objs = []
    find = []

    def findCluster(x):
        while(find[x-1]!=x):
            x = find[x-1]
        return x

    img = copy.deepcopy(img)
    nrows,ncols = img.shape
    for i in range(nrows):
        for j in range(ncols):
        
            if(img[i,j]!= bg):
                if(nh == 4):
                    ngbrs = neighborhood4(i,j)
                else:
                    ngbrs = neighborhood8(i,j, ncols)
                ngbrs = [ele for ele in ngbrs if(img[ele[0], ele[1]]!= bg)]
                
                if(len(ngbrs)==0):
                    img[i, j] = label
                    find.append(label)
                    objs.append((i,i, j,j))
                    label+=1

                else:
                    finds = list(set([findCluster(img[ele[0], ele[1]]) for ele in ngbrs]))
                    relObjs = [objs[ele-1] for ele in finds if objs[ele-1] is not None]
                    leastVal = min(finds)
                    img[i, j] = leastVal
                    relObjs.append((i,i,j,j))
                    combObj = combineObjs(relObjs)
                    for ele in finds:
                        find[ele-1] = leastVal

                    for ele in finds:
                        objs[ele-1] = None
                    objs[leastVal-1] = combObj

    for i in range(nrows):
        for j in range(ncols):
            if(img[i,j]!=bg):
                img[i,j] = findCluster(img[i,j])
   

    objs = [i for i in objs if i is not None]
    objs = list(filter(lambda x: x[0]!=x[1] and x[2]!=x[3], objs))
    return objs




from scipy.sparse import csr_matrix
import numpy as np
import csv
import sys
from random import randint
from pyspark import SparkConf, SparkContext


W = []
H = []

#maps a userid and movieid to the stratum
def getStratum(x,no_workers,user_blocks,movie_blocks):
	userId, movieId = x[0], x[1]
	strata = (movieId/movie_blocks) - (userId/user_blocks)
	strata = strata%no_workers

	return strata

#creates a sparse matrix for the ratings block
def getSparse(V,rowSize,colSize,row_start,col_start):
	rows = []
	cols = []
	ratings = []
	for triple in V:
		rows.append(triple[1][0]-row_start)
		cols.append(triple[1][1]-col_start)
		ratings.append(triple[1][2])
		#ratings.append(-1 if triple[1][2]==0 else triple[1][2])
	
	return csr_matrix((ratings, (rows,cols)), shape=(rowSize, colSize))

def SGD(x):

    	T0 = 100
    	n = x[7]

    	#get number of non zero updates
    	max_iter=V[:,:].nonzero()[0].size

    	W = x[1]
    	H = x[2]
	
    	V = getSparse(x[0],W.shape[0],H.shape[1],x[3],x[5])
    	no_rows = V.shape[0]
    	no_cols = V.shape[1]

    	iter=0
    
    	epsilon = pow(T0+n,-1*beta_value)

    	while(True):
    		#get random row id and col id
        	row = randint(0,no_rows-1) 
        	col = randint(0,no_cols-1)
        	flag = 0

        	#perform updates for all non zero entries
        	if V[row,col] > 0:
                    
			#update learning rate parameter
	        	W1 = W[row,: ]
        		H1 = H[:,col]

         		temp = V[row,col] - W1.dot(H1)                     
		
			#perform gradient updates
			W_update = -2.0*(temp)*H1.transpose() + 2.0*(lambda_value/ V[row,:].nonzero()[0].size)*W1
	        	H_update = -2.0*(temp)*W1.transpose() + 2.0*(lambda_value/ V[:,col].nonzero()[0].size)*H1

	       		W[row,:] = W[row,:] - epsilon * W_update
			H[:,col] = H[:,col] - epsilon * H_update

			iter+=1

        if iter>max_iter:
            break

    	return (W,H,n + iter,x[3],x[4],x[5],x[6])

def LoadNetSparse():
        
        rows,cols,vals =[],[],[]
        for key,value in textfiles:
                tuples = value.split("\n")
                key = tuples[0].split(":")[0]

                no_tuples =  len(tuples)
                for i in range(1,no_tuples-1):
                        tokens = tuples[i].split(",")
                        user = tokens[0]
                        rating =  tokens[1]
                        rows.append(int(user))
                        cols.append(int(key))
                        vals.append(float(rating))

        return  (vals,rows,cols)

def LoadSparseMatrix():
        val = []
        row = []
        col = []
        select = []
        f = open(trainfile)
        reader = csv.reader(f)
        for line in reader:
                row.append( int(line[0])-1 )
                col.append( int(line[1])-1 )
                val.append( float(line[2]) )
                select.append( (int(line[0])-1, int(line[1])-1) )
        return csr_matrix( (val, (row, col)) ), select

def combineLists(x,y):
        return (x[0]+y[0],x[1]+y[1],x[2]+y[2])

def writeMatrix(arr,outputFile):
        np.savetxt(outputFile,arr,delimiter=",")

mseOutput = open("mseoutput.txt", "w")

def writeMSE(V, pred, iteration):
        global mseOutput
        error = 0
        
	error = map(lambda x:(x[2] - pred[x[0]][x[1]])*(x[2] -pred[x[0]][x[1]]), V.collect())
	mse = reduce(lambda x,y: x+y,error )
        mseOutput.write("Iteration {0} {1}".format(iteration, mse/len(V.collect())))

def main():
	conf = SparkConf().setAppName("DSGD")#.setMaster("local[3]")
    	sc = SparkContext(conf=conf)

    	#load rating from trainfile
    	rdd = sc.textFile(trainfile).map(lambda x: x.split(",")).map(lambda x: (int(x[0])-1, int(x[1])-1, float(x[2])))
        
    	#get max users
    	users = rdd.map(lambda x: x[0]).max() + 1
    	#get max movies
    	movies = rdd.map(lambda x: x[1]).max() + 1

    	curr_block = []
        
    	prev_loss = -1.0
    	tol=0.00001

    	#initialize W and H to random non zero matrices
    	W =  np.random.rand(users,no_factors)
    	H = np.random.rand(no_factors,movies)

    	index_V = np.zeros((no_workers,4*no_workers))

    	user_blocks = users / no_workers
    	movie_blocks = movies / no_workers

    	if users%no_workers > 0:
		user_blocks = users / no_workers +1

    	if movies%no_workers > 0:
		movie_blocks = movies/ no_workers +1
       
    	#block partition logic
    	#stroes the boundary indices for each blcok
    	#iterate through each strate
    	for i in range (0,no_workers):
            	
		#print 'strata {0}'.format(k)
   		for j in range(0,no_workers):
			row_start = j*user_blocks 
			row_end = row_start + user_blocks
			col_start = j*movie_blocks + i*movie_blocks 
			col_end = col_start + movie_blocks
			
			if col_start > movie_blocks:
				col_start = col_start % movies -1
				col_end = col_start + movie_blocks

			if col_end > movies:
				col_end = movies

			if row_end > users:
				row_end = users

			index_V [i,4*j] = row_start
			index_V [i,4*j+1] = row_end
			index_V [i,4*j+2] = col_start
			index_V [i,4*j+3] = col_end

        
	# creating keys for each Partitions/block
        partitions = rdd.keyBy(lambda x: getStratum(x,no_workers,user_blocks,movie_blocks))

	#epochs
    	for k in range(0,no_iter):
        	n=0
		#strata
		for i in range(0,no_workers):
	
			#filter all blocks belonging to 1 stratum
	    		blocks = partitions.filter(lambda x: x[0] == i)  #stratum
	
	    		data = []
			
			#get V,W,H for each block in the stratum
		    	for j in range(0,no_workers):
			#filter ratings for 1 block of the stratum
			curr_V = blocks.filter(lambda x: x[1][0]/user_blocks ==j)
		        curr_W = W[index_V[i,4*j]:index_V[i,4*j+1],:]
		        curr_H = H[:,index_V[i,4*j+2]:index_V[i,4*j+3]]
		        data.append((curr_V.collect(),curr_W,curr_H,index_V[i,4*j],index_V[i,4*j+1],index_V[i,4*j+2],index_V[i,4*j+3],n))
		
				
		    	r = sc.parallelize(data,no_workers).map(lambda x : SGD(x)).collect()
		                   	
			#iterate through results from SGD to sum on the iterations that is passed 
			#to the next strata
			for tuple in r:
				n += tuple[2]
				row_start = tuple[3]
				row_end = tuple[4]
				col_start = tuple[5]
				col_end = tuple[6]
	
				W[row_start:row_end,:] = tuple[0]
				H[:,col_start:col_end] = tuple[1]
	
	        nz_rows, nz_cols = V.nonzero()
		temp =  W.dot(H)
	
			#calculate error to check for convergence
		LNZSL = V[nz_rows,nz_cols] - temp[nz_rows,nz_cols]
		LNZSL = np.sum(LNZSL*LNZSL.T)
	
		loss = LNZSL + lambda_value * (np.sum(W*W) + np.sum(H*H))
	
		if np.fabs(loss - prev_loss) < tol:
			break
		else:
			prev_loss = loss
		

    	#function to calculate the MSE error       
    	#writeMSE(rdd, W.dot(H), k + 1)
    	writeMatrix(W,"W.csv")
    	writeMatrix(H,"H.csv")
    	#global mseOutput
    	#mseOutput.close()

	#stop spark context
    	sc.stop()



#get no of factors
no_factors =  int(sys.argv[1])

#get number of workers
no_workers =  int(sys.argv[2])

#get number of iterations
no_iter =  int(sys.argv[3])

#get beta value for learning rate
beta_value  = float(sys.argv[4])

#get lambda value for regularization paramter
lambda_value =  float(sys.argv[5])
trainfile = sys.argv[6]
main()


